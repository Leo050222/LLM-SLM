#!/usr/bin/env python3
"""Verify an offline benchmark download against root/plan.json.

Dependencies: ``pip install pyarrow Pillow``. This script installs nothing and
never uses the network. Run ``python verify.py --root /data/vision_benchmarks``.
The machine-readable report is root/VERIFIED.json; any failed check exits 1.
Optional source hashes are checked when present; otherwise SHA-256 is recorded
and the report explicitly identifies that only the source size was compared.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

from load_benchmark import (MMBenchImages, data_path, decode_image_cell, nonempty,
                            read_plan, tsv_rows)


def image_blobs(value):
    """Yield encoded bytes embedded in an Arrow image value, never read image paths."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        if value:
            yield bytes(value)
    elif isinstance(value, dict):
        if value.get('bytes'):
            yield from image_blobs(value['bytes'])
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from image_blobs(child)
    elif isinstance(value, str) and nonempty(value):
        try:
            yield decode_image_cell(value)
        except ValueError:
            pass  # A file name or URL is not an embedded image.


def pil_verify(blob):
    from PIL import Image
    with Image.open(io.BytesIO(blob)) as image:
        image.verify()


def verify_parquet(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)  # Reading the footer validates the file structure.
    schema = parquet.schema_arrow
    names = schema.names
    image_columns = [name for name in names
                     if re.fullmatch(r'(?:decoded_image|image(?:_\d+)?)', name, re.I)]
    answer_columns = [name for name in names
                      if name.lower() in ('answer', 'answers', 'label', 'correct_answer', 'gt_answer')]
    result = {
        'kind': 'parquet', 'rows': parquet.metadata.num_rows,
        'row_groups': parquet.metadata.num_row_groups,
        'schema': [{'name': field.name, 'type': str(field.type), 'nullable': field.nullable}
                   for field in schema],
        'images': {name: {'embedded_nonempty_rows': 0, 'embedded_images': 0,
                          'path_nonempty_rows': 0} for name in image_columns},
        'answers_nonempty': {name: 0 for name in answer_columns},
        'rows_with_embedded_image': 0, 'pil_samples_verified': 0,
    }
    read_rows = 0
    errors = []
    for batch in parquet.iter_batches(batch_size=256, columns=image_columns + answer_columns):
        read_rows += batch.num_rows
        for row in batch.to_pylist():
            has_image = False
            for name in image_columns:
                value = row.get(name)
                blobs = list(image_blobs(value))
                stats = result['images'][name]
                stats['embedded_images'] += len(blobs)
                stats['embedded_nonempty_rows'] += bool(blobs)
                if isinstance(value, dict) and nonempty(value.get('path')):
                    stats['path_nonempty_rows'] += 1
                elif isinstance(value, str) and nonempty(value) and not blobs:
                    stats['path_nonempty_rows'] += 1
                has_image = has_image or bool(blobs)
                for blob in blobs:
                    if result['pil_samples_verified'] >= 3:
                        break
                    pil_verify(blob)
                    result['pil_samples_verified'] += 1
            result['rows_with_embedded_image'] += has_image
            for name in answer_columns:
                result['answers_nonempty'][name] += nonempty(row.get(name))
    result['rows_read'] = read_rows
    if read_rows != result['rows']:
        errors.append('Parquet rows read do not match footer row count')
    footer_rows = sum(parquet.metadata.row_group(index).num_rows
                      for index in range(parquet.metadata.num_row_groups))
    if footer_rows != result['rows']:
        errors.append('Parquet row-group counts do not match footer row count')
    result['errors'] = errors
    return result


def verify_mmbench(path):
    images = MMBenchImages(path)
    result = {'kind': 'mmbench_tsv', 'rows': 0, 'unique_question_indices': 0,
              'missing_question_indices': 0, 'answers_nonempty': 0,
              'images_resolved': 0, 'image_reference_rows': 0,
              'pil_samples_verified': 0, 'errors': []}
    indices = set()
    first_error = None
    for row in tsv_rows(path):
        result['rows'] += 1
        index = str(row.get('index') or '').strip()
        if index:
            indices.add(index)
        else:
            result['missing_question_indices'] += 1
        result['answers_nonempty'] += nonempty(row.get('answer'))
        try:
            blob, referenced = images.resolve(row.get('image'))
            result['images_resolved'] += 1
            result['image_reference_rows'] += referenced
            if result['pil_samples_verified'] < 3:
                pil_verify(blob)
                result['pil_samples_verified'] += 1
        except Exception as error:
            if first_error is None:
                first_error = type(error).__name__ + ': ' + str(error)
    result['unique_question_indices'] = len(indices)
    if result['images_resolved'] != result['rows']:
        result['errors'].append('Some MMBench images could not be resolved or decoded')
    if first_error:
        result['errors'].append(first_error)
    return result


def verify_file(root, dataset_name, item):
    result = {'path': item['path'], 'ok': False, 'errors': []}
    path = data_path(root, dataset_name, item['path'])
    result['actual_path'] = str(path)
    if not path.is_file():
        result['errors'].append('File missing')
        return result
    actual_size = path.stat().st_size
    result['size'] = actual_size
    if 'size' not in item:
        result['errors'].append('No expected size in plan')
    elif actual_size != int(item['size']):
        result['errors'].append('File size differs from plan')
    sha256 = hashlib.sha256()
    md5 = hashlib.md5() if item.get('md5') else None
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            sha256.update(block)
            if md5:
                md5.update(block)
    result['sha256'] = sha256.hexdigest()
    expected_sha256 = str(item.get('sha256') or '').lower().removeprefix('sha256:')
    if expected_sha256 and result['sha256'] != expected_sha256:
        result['errors'].append('SHA-256 differs from plan')
    if md5:
        result['md5'] = md5.hexdigest()
        if result['md5'] != str(item['md5']).lower():
            result['errors'].append('MD5 differs from plan')
    result['source_hash_comparison'] = ('sha256' if expected_sha256 else
                                        'md5' if md5 else 'unavailable_size_only')
    if not result['errors']:
        suffix = path.suffix.lower()
        if suffix == '.parquet':
            result.update(verify_parquet(path))
        elif suffix == '.tsv':
            result.update(verify_mmbench(path))
        else:
            result['kind'] = 'ancillary_file'
    result['ok'] = not result['errors']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    report = {'root': str(root), 'checked_at': datetime.now(timezone.utc).isoformat(),
              'ok': False, 'datasets': [], 'errors': []}
    try:
        plan = read_plan(root)
        if not plan['datasets']:
            raise ValueError('The plan contains no datasets')
        for dataset in plan['datasets']:
            entry = {'name': dataset['name'], 'files': [], 'ok': False}
            if not dataset.get('files'):
                entry['errors'] = ['Dataset has no files in the plan']
            for item in dataset.get('files', []):
                try:
                    checked = verify_file(root, dataset['name'], item)
                except Exception as error:
                    checked = {'path': item.get('path'), 'ok': False,
                               'errors': [type(error).__name__ + ': ' + str(error)]}
                entry['files'].append(checked)
                print('{} {} {}'.format('OK' if checked['ok'] else 'FAIL',
                                       dataset['name'], item.get('path')), flush=True)
            entry['ok'] = bool(entry['files']) and all(item['ok'] for item in entry['files'])
            report['datasets'].append(entry)
        report['ok'] = all(dataset['ok'] for dataset in report['datasets'])
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    temporary = root / 'VERIFIED.json.tmp'
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        os.replace(temporary, root / 'VERIFIED.json')
    except Exception as error:
        print('Could not write VERIFIED.json: ' + str(error), file=sys.stderr)
        return 1
    print('Verification {}: {}'.format('passed' if report['ok'] else 'failed',
                                      root / 'VERIFIED.json'))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
