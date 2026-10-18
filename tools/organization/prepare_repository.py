#!/usr/bin/env python3
"""Export a small, credential-checked source snapshot without importing Git history.

Preview: python3 tools/organization/prepare_repository.py --dry-run
Export:  python3 tools/organization/prepare_repository.py

Excluded directories are recorded once rather than enumerating their children.
Only included regular files are hashed. Secrets are never printed or written;
the local review log contains locations and detector names only.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo


MAX_BYTES = 5 * 1024 * 1024
MANIFEST_JSON = "docs/project/PUBLICATION_MANIFEST.json"
MANIFEST_CSV = "docs/project/PUBLICATION_MANIFEST.csv"
MANIFEST_PATHS = {MANIFEST_JSON, MANIFEST_CSV}
SOURCE_SUFFIXES = {
    ".py", ".pyi", ".sh", ".bash", ".zsh", ".slurm", ".sbatch",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".css", ".scss",
    ".ipynb", ".r", ".jl", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".cu", ".cuh", ".mmd", ".tex", ".bib", ".sql", ".patch", ".diff",
}
DOCUMENT_SUFFIXES = {".md", ".rst", ".txt"}
FIGURE_SUFFIXES = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".pdf"}
BINARY_SUFFIXES = FIGURE_SUFFIXES - {".svg"}
FORBIDDEN_DIR_NAMES = {
    ".git", ".local", ".aris", ".agents", ".codex", ".openai", ".next",
    "node_modules", ".pnpm-store", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".cache", ".mplcache", "cache", "dist", "build", "venv",
    ".venv", "data", "checkpoints", "weights", "models",
    "cases", "runs", "shards", "logs", "slurm_logs", "judge_cache",
    "raw", "raw_data", "samples", "rollouts", "traces", "completed",
    "data_legacy_json", "data_json", "data_js", "tokens", "logits", "scores",
    "items", "teacher_prefixes", "claims",
}
FORBIDDEN_ROOTS = {"bridge-rollout-browser", "downloads", "tmp", "refine-logs"}
FORBIDDEN_PREFIXES = {"artifacts/legacy_tmp"}
ALWAYS_PRIVATE_DIRS = {
    ".git", ".local", ".aris", ".agents", ".codex", ".openai", "node_modules",
    ".pnpm-store", ".venv", "venv", "__pycache__", ".cache", ".next", "dist",
}
KNOWN_SENSITIVE = {
    "ocrd_pipeline/anchor_selector/.secrets.env",
    "ocrd_pipeline/anchor_selector/.secrets.openlux.env",
    "reports/deepmath_student_top16_trace_difference.html",
    "reports/deepmath_baseline_sample_gap/data_legacy_json/case_0004.js",
    "reports/deepmath_baseline_sample_gap/data_legacy_json/case_0069.js",
}
FORBIDDEN_SUFFIXES = {
    ".jsonl", ".ndjson", ".parquet", ".arrow", ".feather", ".sqlite", ".db",
    ".bin", ".pt", ".pth", ".safetensors", ".ckpt", ".npy", ".npz",
    ".pkl", ".pickle", ".h5", ".hdf5", ".zst", ".gz", ".bz2", ".xz",
    ".zip", ".tar", ".tgz", ".7z", ".rar", ".log", ".err", ".out",
    ".pem", ".p12", ".pfx", ".key", ".pyc", ".pyo", ".map",
}
RAW_STEM = re.compile(
    r"(?:^|[_-])(?:per_question|per_sample|per_token|paired(?:_scores|_results)?|"
    r"candidate_records|prompts?|responses?|answers?|generations?|trajectories|"
    r"rollouts?|intervention_points|token_records|sample_records|examples?|"
    r"benchmark_inputs|verified_judge_counterexamples|student_only_correctness)"
    r"(?:$|[_-])", re.I,
)
SUMMARY_STEM = re.compile(
    r"summary|summaries|manifest|metric|comparison|accuracy|aggregate|statistics|"
    r"counts?|table|pass[0-9]|audit|status|config|metadata|complete|submission|"
    r"submitted|launch|request|plan|resume|resubmission|recovery|verification|"
    r"score_rank|job_ids?|settings|hash|benchmark|association|distribution|"
    r"feasibility|results?|analysis|coverage|catalog|records|series|remote_sources",
    re.I,
)
SECRET_PATTERNS = {
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "api_token_sk": re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "npm_token": re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "aws_access_id": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\bBearer\s+([A-Za-z0-9_./+=-]{20,})"),
}
HARDCODED = re.compile(
    r"\b(api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|client[_-]?secret|"
    r"password|passwd|(?:bce|aws)[_-]?access[_-]?key[_-]?id|"
    r"(?:bce|aws)[_-]?secret[_-]?access[_-]?key)"
    r"[\s\"']*[:=]\s*[\"']([^\"'\s\\]{6,})[\"']", re.I,
)
PLACEHOLDER = re.compile(
    r"(?:your[_-]|replace[_-]|example|dummy|placeholder|fake[_-]|test[_-]|"
    r"not[_-]?a[_-]?real|redacted|changeme|insert[_-]|<|\$\{|os\.environ|process\.env)",
    re.I,
)
SPECIAL_NAMES = {
    ".gitignore", ".gitattributes", ".editorconfig", ".npmrc", "Makefile",
    "Dockerfile", "LICENSE", "LICENSE.txt", "COPYING", "NOTICE",
    "requirements.txt", "requirements-dev.txt", "package.json", "package-lock.json",
    "pnpm-lock.yaml", "yarn.lock", "uv.lock", "poetry.lock", "Pipfile", "Pipfile.lock",
    "environment.yml", "environment.yaml", "pyproject.toml", "setup.py", "setup.cfg",
    "tsconfig.json", "tsconfig.app.json", "tsconfig.node.json", "components.json",
}
AGGREGATE_EXCEPTIONS = {
    "reports/quartile_support_k_results_20260919/paired_changes_vs_k0.csv",
    "reports/all_benchmark_comparison_20260911/single_rollout_results.csv",
    "reports/all_benchmark_comparison_20260911/single_rollout_results.json",
}


def under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def exclusion(path: str, is_dir: bool, extra: list[str]) -> str | None:
    p = Path(path)
    parts = p.parts
    low = p.name.lower()
    if path in MANIFEST_PATHS:
        return "manifest_self_reference"
    if any(fnmatch.fnmatchcase(path, pattern) for pattern in extra):
        return "explicit_exclusion"
    if path in KNOWN_SENSITIVE:
        return "known_sensitive_file"
    if any(part in ALWAYS_PRIVATE_DIRS for part in parts if part != p.name or is_dir):
        return "private_metadata_dependency_or_cache_directory"
    if parts and parts[0] in FORBIDDEN_ROOTS:
        return "local_duplicate_or_temporary_root"
    if any(under(path, prefix) for prefix in FORBIDDEN_PREFIXES):
        return "archived_temporary_material"
    curated_summary = under(path, "results/westlake") or under(path, "results/legacy")
    if not curated_summary and any(part in FORBIDDEN_DIR_NAMES for part in parts if part != p.name or is_dir):
        return "dependency_cache_dataset_or_raw_run_directory"
    if "datasets" in parts and parts[0] != "datasets":
        return "dataset_directory"
    if low == ".ds_store" or low.startswith((".env", ".secret", ".credential")):
        return "credential_or_local_metadata"
    if low in {"id_rsa", "id_ed25519", "id_ecdsa", "credentials", "authorized_keys", "known_hosts"}:
        return "credential_or_local_metadata"
    compressed_catalog = under(path, "docs/experiments") and low.endswith((".csv.gz", ".json.gz"))
    if not is_dir and p.suffix.lower() in FORBIDDEN_SUFFIXES and not compressed_catalog:
        return "raw_data_model_archive_or_log"
    return None


def source_reference_paths(root: Path) -> set[str]:
    """Keep compact source summaries referenced by the experimental documentation."""
    refs = set()
    for doc in (root / "docs" / "experiments").glob("*.md"):
        text = doc.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\(([^)]+)\)", text):
            value = match.group(1).split("#", 1)[0]
            if not value or "://" in value or value.startswith("#"):
                continue
            try:
                resolved = (doc.parent / value).resolve()
                refs.add(resolved.relative_to(root).as_posix())
            except (ValueError, RuntimeError, OSError):
                continue
    return refs


def choose(path: str, size: int, refs: set[str]) -> tuple[bool, str]:
    p = Path(path)
    ext, name = p.suffix.lower(), p.name
    if size > MAX_BYTES:
        return False, "larger_than_5_mib"
    if under(path, "docs/experiments"):
        return True, "experiment_documentation"
    if under(path, "results/westlake") or under(path, "results/legacy"):
        return True, "curated_shared_storage_summary"
    if path == "artifacts/storage_inventory/westlake_inventory_20261019.json":
        return True, "shared_storage_provenance_inventory"
    if under(path, "datasets"):
        return (True, "dataset_location_documentation") if ext in DOCUMENT_SUFFIXES or "manifest" in p.stem else (False, "dataset_payload")
    if ext in SOURCE_SUFFIXES or name in SPECIAL_NAMES or name.startswith("requirements"):
        if ext == ".js" and (name == "data.js" or re.match(r"case_\d+\.js$", name)):
            return False, "generated_sample_payload"
        return True, "source_or_configuration"
    if ext in {".html", ".htm"} and path.startswith(("code/", "src/", "tools/")):
        return True, "web_visualization_source"
    if ext in DOCUMENT_SUFFIXES:
        if RAW_STEM.search(p.stem):
            return False, "per_sample_text"
        return True, "documentation_or_compact_text"
    if ext in FIGURE_SUFFIXES:
        return True, "scientific_figure_or_document"
    if ext in {".json", ".csv", ".tsv"}:
        if path in AGGREGATE_EXCEPTIONS or re.search(r"(?:summary|manifest|status|complete)$", p.stem, re.I):
            return True, "aggregate_summary_manifest_or_completion_marker"
        if RAW_STEM.search(p.stem):
            return False, "per_sample_or_token_table"
        if under(path, "docs") or under(path, "code") or path in refs or SUMMARY_STEM.search(p.stem):
            return True, "compact_summary_configuration_or_document_source"
        return False, "unclassified_structured_data"
    return False, "not_publication_material"


def placeholder(value: str) -> bool:
    return bool(PLACEHOLDER.search(value)) or value.lower() in {
        "password", "password123", "api_key", "api-key", "access_token", "token",
        "none", "null", "undefined", "secret", "secret-key", "secret_key",
    }


def secret_locations(data: bytes, path: str) -> list[dict]:
    if b"\x00" in data[:8192]:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        kinds = set()
        for kind, regex in SECRET_PATTERNS.items():
            if any(not placeholder(m.group(0)) for m in regex.finditer(line)):
                kinds.add(kind)
        for match in HARDCODED.finditer(line):
            label, value = match.groups()
            if placeholder(value):
                continue
            if label.lower() in {"password", "passwd"} or len(value) >= 16:
                if not value.startswith(("http", "/", "./", "../")):
                    kinds.add("hardcoded_credential")
        if kinds:
            found.append({"path": path, "line": number, "detectors": sorted(kinds)})
    return found


def file_record(path: str, kind: str, size: int | None, reason: str, included=False) -> dict:
    return {
        "path": path, "type": kind, "status": "included" if included else "excluded",
        "reason": reason, "size_bytes": size, "sha256": None,
        "hash_status": "not_read_excluded",
    }


def inventory(root: Path, extra: list[str]) -> tuple[list[dict], list[dict]]:
    refs = source_reference_paths(root)
    records, links, findings = [], [], []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        retained = []
        for name in dirs:
            item = Path(directory) / name
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                links.append((relative, item))
                continue
            reason = exclusion(relative, True, extra)
            if reason:
                records.append(file_record(relative, "directory", None, reason))
            else:
                retained.append(name)
        dirs[:] = retained
        for name in files:
            item = Path(directory) / name
            relative = item.relative_to(root).as_posix()
            if item.is_symlink():
                links.append((relative, item))
                continue
            try:
                metadata = item.stat()
            except OSError:
                records.append(file_record(relative, "file", None, "unreadable_file"))
                continue
            if not stat.S_ISREG(metadata.st_mode):
                records.append(file_record(relative, "special", None, "not_regular_file"))
                continue
            reason = exclusion(relative, False, extra)
            selected = False
            if not reason:
                selected, reason = choose(relative, metadata.st_size, refs)
            record = file_record(relative, "file", metadata.st_size, reason, selected)
            if selected:
                data = item.read_bytes()
                detected = [] if item.suffix.lower() in BINARY_SUFFIXES else secret_locations(data, relative)
                if detected:
                    findings.extend(detected)
                    record.update(status="excluded", reason="sensitive_pattern_requires_explicit_exclusion")
                else:
                    record.update(sha256=hashlib.sha256(data).hexdigest(), hash_status="computed")
            records.append(record)
    selected_paths = {r["path"] for r in records if r["status"] == "included"}
    for relative, item in links:
        target = os.readlink(item)
        record = file_record(relative, "symlink", len(os.fsencode(target)), "", False)
        record["link_target"] = target
        reason = exclusion(relative, item.is_dir(), extra)
        if relative in {"refine-logs", "downloads"} and reason == "local_duplicate_or_temporary_root":
            reason = None
        if not reason and os.path.isabs(target):
            reason = "absolute_symlink"
        if not reason:
            try:
                resolved = item.resolve(strict=True).relative_to(root).as_posix()
                if resolved not in selected_paths and not any(p.startswith(resolved + "/") for p in selected_paths):
                    reason = "symlink_target_not_in_snapshot"
            except (ValueError, RuntimeError, OSError):
                reason = "external_or_dangling_symlink"
        if reason:
            record["reason"] = reason
        else:
            record.update(status="included", reason="internal_relative_symlink",
                          sha256=hashlib.sha256(os.fsencode(target)).hexdigest(), hash_status="link_text_computed")
        records.append(record)
    return sorted(records, key=lambda r: r["path"]), findings


def manifest_document(root: Path, records: list[dict], findings: list[dict]) -> dict:
    included = [r for r in records if r["status"] == "included"]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "target_repository": "https://github.com/Leo050222/LLM-SLM",
        "source_root": str(root),
        "policy": {
            "maximum_file_bytes": MAX_BYTES,
            "follow_directory_symlinks": False,
            "excluded_directory_coverage": "A directory record covers all descendants, which are not enumerated.",
            "excluded_file_hashes": "Excluded payloads are not read; sha256 is null.",
            "manifest_self_reference": "The two publication manifests are copied but excluded from their own hash inventory.",
            "secret_logging": "Locations and detector names only; no credential values are retained.",
        },
        "summary": {
            "included_files": sum(r["type"] == "file" for r in included),
            "included_symlinks": sum(r["type"] == "symlink" for r in included),
            "included_file_bytes": sum(r["size_bytes"] or 0 for r in included if r["type"] == "file"),
            "excluded_records": len(records) - len(included),
            "excluded_reasons": dict(sorted(Counter(r["reason"] for r in records if r["status"] == "excluded").items())),
            "sensitive_file_count": len({r["path"] for r in findings}),
        },
        "entries": records,
    }


def write_manifests(root: Path, document: dict, findings: list[dict]) -> None:
    path = root / MANIFEST_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["path", "type", "status", "reason", "size_bytes", "sha256", "hash_status", "link_target"]
    with (root / MANIFEST_CSV).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(document["entries"])
    local = root / ".local" / "review" / "publication_secrets.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps({"findings": findings}, indent=2) + "\n", encoding="utf-8")
    local.chmod(0o600)


def check_experiment_links(root: Path, records: list[dict]) -> list[dict]:
    included = {r["path"] for r in records if r["status"] == "included"}
    missing = []
    for relative in sorted(included):
        if not under(relative, "docs/experiments") or not relative.endswith(".md"):
            continue
        doc = root / relative
        for match in re.finditer(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            value = match.group(1).split("#", 1)[0]
            if not value or "://" in value or value.startswith("#"):
                continue
            try:
                resolved = (doc.parent / value).resolve().relative_to(root).as_posix()
            except (ValueError, RuntimeError, OSError):
                missing.append({"document": relative, "target": value, "reason": "outside_source_root"})
                continue
            if (resolved not in included and resolved not in MANIFEST_PATHS
                    and not any(path.startswith(resolved + "/") for path in included)):
                missing.append({"document": relative, "target": value, "reason": "target_not_published"})
    return missing


def export(root: Path, output: Path, records: list[dict]) -> None:
    if output == root or root not in output.resolve().parents:
        raise ValueError("Output must be a separate directory inside the source root.")
    if output.exists() and (output / ".git").exists():
        raise ValueError("Output contains .git; use a fresh --output directory to preserve its Git history.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".publication-", dir=output.parent))
    try:
        for record in records:
            if record["status"] != "included":
                continue
            destination = staging / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if record["type"] == "symlink":
                destination.symlink_to(record["link_target"])
            else:
                shutil.copy2(root / record["path"], destination)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != record["sha256"]:
                    raise ValueError("Source changed while exporting: " + record["path"])
        for manifest in MANIFEST_PATHS:
            destination = staging / manifest
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / manifest, destination)
            if destination.stat().st_size > MAX_BYTES:
                raise ValueError("Publication manifest exceeds maximum file size: " + manifest)
        dangling = [r["path"] for r in records if r["status"] == "included" and r["type"] == "symlink"
                    and not (staging / r["path"]).exists()]
        if dangling:
            raise ValueError("Symlink targets missing in snapshot: " + ", ".join(dangling))
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(".local/publish/LLM-SLM"))
    parser.add_argument("--dry-run", action="store_true", help="Write manifests/review log; do not modify export directory.")
    parser.add_argument("--exclude", action="append", default=[], help="Additional source-relative path/glob exclusion.")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    records, findings = inventory(root, args.exclude)
    document = manifest_document(root, records, findings)
    missing_links = check_experiment_links(root, records)
    document["summary"]["unpublished_experiment_document_links"] = len(missing_links)
    write_manifests(root, document, findings)
    link_log = root / ".local" / "review" / "publication_links.json"
    link_log.write_text(json.dumps(missing_links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": "preview" if args.dry_run else "export", "output": str(output),
                      **document["summary"]}, ensure_ascii=False, indent=2))
    for finding in findings:
        print(f"SENSITIVE {finding['path']}:{finding['line']} ({', '.join(finding['detectors'])})", file=sys.stderr)
    if findings:
        print("Export blocked. Review the local location log and explicitly exclude sensitive files.", file=sys.stderr)
        return 2
    if not args.dry_run:
        export(root, output, records)
        print("Snapshot exported; no Git metadata was created or copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
