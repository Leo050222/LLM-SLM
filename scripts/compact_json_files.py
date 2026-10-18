#!/usr/bin/env python3
"""Compact directories of per-record JSON files into validated JSONL shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


KEEP_NAMES = {
    "summary.json", "manifest.json", "config.json", "protocol.json",
    "launch.json", "scoring_launch.json", "monitor_state.json",
    "local_verification.json", "data_manifest.json", "audit.json",
    "analysis.json", "intervention_analysis.json", "run_summary.json",
}


def candidates(directory: Path, max_source_bytes: int) -> list[Path]:
    result = []
    for path in directory.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        if path.name.lower() in KEEP_NAMES or path.name.startswith(".jsonl-"):
            continue
        try:
            if path.stat().st_size <= max_source_bytes:
                result.append(path)
        except FileNotFoundError:
            pass
    return sorted(result, key=lambda p: p.name)


def compact_directory(
    directory: Path,
    sources: list[Path],
    shard_bytes: int,
    dry_run: bool,
) -> tuple[int, int]:
    total_bytes = sum(path.stat().st_size for path in sources)
    if dry_run:
        print(f"PLAN\t{len(sources)}\t{total_bytes}\t{directory}", flush=True)
        return len(sources), total_bytes

    prefix = ".jsonl-compaction"
    temps: list[Path] = []
    finals: list[Path] = []
    written = 0
    shard_no = -1
    output = None
    current_bytes = 0

    try:
        for source in sources:
            before = source.stat()
            with source.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            record = {"_source_file": source.name, "data": value}
            encoded = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            payload = encoded.encode("utf-8")

            if output is None or (current_bytes and current_bytes + len(payload) > shard_bytes):
                if output is not None:
                    output.flush()
                    os.fsync(output.fileno())
                    output.close()
                shard_no += 1
                temp = directory / f"{prefix}-{shard_no:05d}.jsonl.tmp"
                final = directory / f"records-{shard_no:05d}.jsonl"
                if temp.exists() or final.exists():
                    raise RuntimeError(f"output already exists: {temp} or {final}")
                temps.append(temp)
                finals.append(final)
                output = temp.open("wb")
                current_bytes = 0

            output.write(payload)
            current_bytes += len(payload)
            written += 1
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"source changed while reading: {source}")

        if output is not None:
            output.flush()
            os.fsync(output.fileno())
            output.close()
            output = None

        checked = 0
        seen: set[str] = set()
        for temp in temps:
            with temp.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    record = json.loads(line)
                    name = record.get("_source_file")
                    if not isinstance(name, str) or name in seen or "data" not in record:
                        raise RuntimeError(f"invalid compacted record: {temp}:{line_no}")
                    seen.add(name)
                    checked += 1
        expected = {path.name for path in sources}
        if checked != written or seen != expected:
            raise RuntimeError(f"verification mismatch in {directory}")

        for temp, final in zip(temps, finals):
            os.replace(temp, final)

        manifest = directory / "jsonl_compaction_manifest.json"
        manifest_tmp = directory / ".jsonl_compaction_manifest.json.tmp"
        with manifest_tmp.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": "one JSON object per line: {_source_file, data}",
                    "source_count": len(sources),
                    "source_bytes": total_bytes,
                    "shards": [path.name for path in finals],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_tmp, manifest)

        for source in sources:
            source.unlink()
        print(f"DONE\t{len(sources)}\t{total_bytes}\t{len(finals)}\t{directory}", flush=True)
        return len(sources), total_bytes
    except Exception:
        if output is not None:
            output.close()
        for temp in temps:
            temp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--min-files", type=int, default=100)
    parser.add_argument("--max-source-mib", type=int, default=16)
    parser.add_argument("--shard-mib", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total_files = 0
    total_bytes = 0
    for target_name in args.targets:
        target = (args.root / target_name).resolve()
        if not target.is_dir() or args.root.resolve() not in target.parents:
            print(f"ERROR\tinvalid target\t{target}", file=sys.stderr)
            return 2
        for directory, child_dirs, _ in os.walk(target):
            child_dirs[:] = [name for name in child_dirs if not name.startswith(".jsonl-")]
            path = Path(directory)
            files = candidates(path, args.max_source_mib * 1024 * 1024)
            if len(files) < args.min_files:
                continue
            count, byte_count = compact_directory(
                path, files, args.shard_mib * 1024 * 1024, args.dry_run
            )
            total_files += count
            total_bytes += byte_count
    print(f"TOTAL\t{total_files}\t{total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
