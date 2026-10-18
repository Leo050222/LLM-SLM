#!/usr/bin/env python3
"""Load downloaded visual benchmarks without contacting Hugging Face.

Dependencies: ``pip install pyarrow`` (Parquet only). ``verify.py`` additionally
uses Pillow. No packages are installed by this module.

Examples::

    python load_benchmark.py --root /data/vision_benchmarks --list
    python load_benchmark.py --root /data/vision_benchmarks --benchmark BLINK \
        --task Relative_Depth --split val --limit 2

    from load_benchmark import iter_benchmark
    rows = iter_benchmark('/data/vision_benchmarks', 'MathVista', split='testmini')

Parquet rows retain their original schema, including embedded image bytes.
MMBench rows expose question, choices, answer, and decoded ``image`` bytes;
``raw`` retains every original TSV field, including the original image value.
"""

import argparse
import base64
import csv
import json
from pathlib import Path
import re
import sys


def read_plan(root):
    root = Path(root).expanduser().resolve()
    with (root / 'plan.json').open(encoding='utf-8') as handle:
        plan = json.load(handle)
    if not isinstance(plan.get('datasets'), list):
        raise ValueError('plan.json must contain a datasets list')
    return plan


def data_path(root, dataset_name, relative_path):
    """Plan paths are relative to root/<dataset name>/; accept root-relative plans too."""
    root = Path(root).expanduser().resolve()
    relative = Path(relative_path)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('A plan file path must be relative and remain inside root')
    candidates = [root / dataset_name / relative, root / relative]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('A data file resolves outside root')
            return resolved
    return candidates[0]


def tsv_rows(path):
    # Image cells can be several megabytes, exceeding csv's small default limit.
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not reader.fieldnames or 'image' not in reader.fieldnames:
            raise ValueError('MMBench TSV has no image column')
        for row in reader:
            if None in row:
                raise ValueError('TSV row has more cells than the header')
            yield row


def nonempty(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ('', 'nan', 'none', 'null')
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def image_format(blob):
    """Recognize common encoded image headers without decoding every image."""
    if blob.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    if blob.startswith(b'\xff\xd8\xff'):
        return 'jpeg'
    if blob.startswith((b'GIF87a', b'GIF89a')):
        return 'gif'
    if blob.startswith(b'RIFF') and blob[8:12] == b'WEBP':
        return 'webp'
    if blob.startswith((b'II*\x00', b'MM\x00*')):
        return 'tiff'
    if blob.startswith(b'BM'):
        return 'bmp'
    if blob.startswith(b'\x00\x00\x01\x00'):
        return 'ico'
    if len(blob) >= 12 and blob[4:8] == b'ftyp':
        return 'iso-bmff-image'
    return None


def decode_image_cell(value):
    value = str(value).strip()
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or ';base64' not in header:
            raise ValueError('Unsupported image data URL')
    encoded = ''.join(value.split())
    try:
        blob = base64.b64decode(encoded + '=' * (-len(encoded) % 4), validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError('Image cell is neither a known index nor valid base64') from error
    if not image_format(blob):
        raise ValueError('Decoded image cell has no recognized image header')
    return blob


class MMBenchImages:
    """Resolve base64 images and chains of references to other question indices."""

    def __init__(self, path):
        self.cells = {}
        for row in tsv_rows(path):
            index = str(row.get('index', '')).strip()
            cell = str(row.get('image') or '').strip()
            if index and nonempty(cell):
                previous = self.cells.get(index)
                if previous is not None and previous != cell:
                    raise ValueError('Duplicate question index has conflicting image cells')
                self.cells[index] = cell

    def resolve(self, cell):
        value = str(cell or '').strip()
        if not nonempty(value):
            raise ValueError('Empty MMBench image cell')
        seen = set()
        referenced = False
        while value in self.cells:
            if value in seen:
                raise ValueError('Cyclic MMBench image index reference')
            seen.add(value)
            value = self.cells[value]
            referenced = True
        return decode_image_cell(value), referenced


def _matches(path, selector):
    if not selector:
        return True
    # MathVerse publishes these as distinct configurations and splits.
    stem = Path(path).stem.lower()
    if stem in ('testmini', 'testmini_text_only') and selector.lower() in ('testmini', 'testmini_text_only'):
        return stem == selector.lower()
    # Match test separately from testmini; support task names with spaces/hyphens.
    normalize = lambda value: re.sub(r'[^a-z0-9]', '', value.lower())
    wanted = normalize(selector)
    if any(normalize(part) == wanted for part in Path(path).parts):
        return True
    return bool(re.search(r'(?<![a-z0-9])' + re.escape(selector.lower()) +
                          r'(?![a-z0-9])', str(path).lower()))


def selected_files(root, benchmark, split=None, task=None):
    plan = read_plan(root)
    matched = [dataset for dataset in plan['datasets']
               if dataset['name'].lower() == benchmark.lower()]
    if not matched:
        raise ValueError('Unknown benchmark: ' + benchmark)
    dataset = matched[0]
    files = [item for item in dataset['files']
             if Path(item['path']).suffix.lower() in ('.parquet', '.tsv')
             and _matches(item['path'], split) and _matches(item['path'], task)]
    if not files:
        raise ValueError('No data files match the benchmark/split/task selection')
    return dataset, files


def iter_benchmark(root, benchmark, split=None, task=None, batch_size=128):
    """Yield offline rows. Split/task selectors match repository file paths."""
    dataset, files = selected_files(root, benchmark, split, task)
    for item in files:
        path = data_path(root, dataset['name'], item['path'])
        if path.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                yield from batch.to_pylist()
        else:
            images = MMBenchImages(path)
            for raw in tsv_rows(path):
                blob, _ = images.resolve(raw.get('image'))
                row = dict(raw)
                row.update(question=raw.get('question'),
                           choices={key: raw[key] for key in ('A', 'B', 'C', 'D', 'E', 'F')
                                    if nonempty(raw.get(key))},
                           answer=raw.get('answer'), image=blob, raw=dict(raw))
                yield row


def _printable(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {'bytes': len(value)}
    if isinstance(value, dict):
        return {key: _printable(child) for key, child in value.items() if key != 'raw'}
    if isinstance(value, (list, tuple)):
        return [_printable(child) for child in value]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + '…'
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--benchmark')
    parser.add_argument('--split')
    parser.add_argument('--task')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--limit', type=int, default=1, help='Rows to display; default: 1')
    args = parser.parse_args()
    if args.list:
        for dataset in read_plan(args.root)['datasets']:
            paths = [item['path'] for item in dataset['files']
                     if Path(item['path']).suffix.lower() in ('.parquet', '.tsv')]
            print(json.dumps({'benchmark': dataset['name'], 'files': paths}, ensure_ascii=False))
        return
    if not args.benchmark:
        parser.error('--benchmark is required unless --list is used')
    if args.limit < 1:
        parser.error('--limit must be positive')
    import itertools
    for row in itertools.islice(iter_benchmark(args.root, args.benchmark, args.split, args.task),
                               args.limit):
        print(json.dumps(_printable(row), ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
