#!/usr/bin/env python3
"""Resume and verify public benchmark downloads described by a JSON plan.

Requires Python 3.9+ and requests. Nothing is installed or configured globally.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import requests


WORKERS = 4
CHUNK_SIZE = 1024 * 1024
UPDATE_SECONDS = 10
PROXY = "http://127.0.0.1:7890"
CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)\Z")
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def acquire_root_lock(root):
    handle = (root / ".download.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write("pid=%d started_at=%s\n" % (os.getpid(), utc_now()))
        handle.flush()
        return handle
    except BlockingIOError:
        handle.close()
        raise RuntimeError("another downloader is already running under " + str(root))
    except BaseException:
        handle.close()
        raise


def load_plan(path, override):
    with Path(path).open(encoding="utf-8") as handle:
        plan = json.load(handle)
    root_value = override or plan.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("plan.root or --root is required")
    root = Path(root_value).expanduser().resolve()
    datasets = plan.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("plan.datasets must be a nonempty list")
    jobs = []
    names = set()
    for dataset in datasets:
        name = dataset.get("name")
        if (not isinstance(name, str) or name in ("", ".", "..")
                or any(character in name for character in "/\\\n\r\0")):
            raise ValueError("invalid dataset name")
        if name in names:
            raise ValueError("duplicate dataset: " + name)
        names.add(name)
        entries = dataset.get("files")
        if not isinstance(entries, list) or not entries:
            raise ValueError(name + ": files must be a nonempty list")
        paths = set()
        for entry in entries:
            relative = entry.get("path")
            if (not isinstance(relative, str) or not relative
                    or any(character in relative for character in "\\\n\r\0")):
                raise ValueError(name + ": invalid file path")
            parsed_path = PurePosixPath(relative)
            if parsed_path.is_absolute() or ".." in parsed_path.parts or not parsed_path.parts:
                raise ValueError(name + ": file path must stay inside dataset")
            relative = str(parsed_path)
            if relative in paths or relative in ("SOURCE.json", "SHA256SUMS"):
                raise ValueError(name + ": duplicate or reserved path: " + relative)
            paths.add(relative)
            size = entry.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(name + "/" + relative + ": invalid expected size")
            url = entry.get("url")
            if not isinstance(url, str) or urlsplit(url).scheme != "https":
                raise ValueError(name + "/" + relative + ": public HTTPS URL required")
            destination = (root / name / relative).resolve()
            if not destination.is_relative_to(root / name):
                raise ValueError("destination escapes dataset: " + relative)
            expected = {}
            for algorithm, length in (("sha256", 64), ("md5", 32)):
                digest = entry.get(algorithm)
                if digest is not None:
                    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{%d}" % length, digest):
                        raise ValueError("invalid " + algorithm + " for " + relative)
                    expected[algorithm] = digest.lower()
            jobs.append({"key": name + "/" + relative, "dataset": name,
                         "path": relative, "url": url, "size": size,
                         "expected": expected, "destination": destination})
    return root, datasets, jobs


class Progress:
    def __init__(self, root, datasets, jobs):
        self.root = root
        self.datasets = datasets
        self.jobs = jobs
        self.lock = threading.RLock()
        self.state = {"state": "RUNNING", "started_at": utc_now(),
                      "updated_at": utc_now(), "workers": WORKERS,
                      "files": {job["key"]: {"state": "PENDING", "bytes": 0,
                                "size": job["size"], "url": job["url"]}
                                for job in jobs}}
        self.last_write = 0.0
        self.write(force=True)

    def write(self, force=False):
        with self.lock:
            now = time.monotonic()
            if force or now - self.last_write >= UPDATE_SECONDS:
                self.state["updated_at"] = utc_now()
                self.state["downloaded_bytes"] = sum(item["bytes"] for item in self.state["files"].values())
                self.state["expected_bytes"] = sum(item["size"] for item in self.state["files"].values())
                atomic_json(self.root / "status.json", self.state)
                self.last_write = now

    def update(self, job, force=False, **changes):
        with self.lock:
            self.state["files"][job["key"]].update(changes)
            self.write(force=force)

    def metadata(self, name):
        with self.lock:
            dataset = next(item for item in self.datasets if item["name"] == name)
            records = []
            sums = []
            for job in self.jobs:
                if job["dataset"] != name:
                    continue
                current = self.state["files"][job["key"]]
                record = {"path": job["path"], "url": job["url"],
                          "size": job["size"], "expected_hashes": job["expected"],
                          "state": current["state"]}
                for field in ("sha256", "md5", "download_url", "transport", "error"):
                    if field in current:
                        record[field] = current[field]
                records.append(record)
                if current["state"] == "COMPLETE":
                    sums.append(current["sha256"] + "  " + job["path"])
            # Preserve pinned repository/version and provenance fields verbatim.
            source = {key: value for key, value in dataset.items() if key != "files"}
            source.update({"name": name, "updated_at": utc_now(), "files": records})
            atomic_json(self.root / name / "SOURCE.json", source)
            atomic_text(self.root / name / "SHA256SUMS", "\n".join(sums) + ("\n" if sums else ""))

    def finish(self, errors):
        with self.lock:
            complete = all(item["state"] == "COMPLETE" for item in self.state["files"].values())
            self.state["state"] = "COMPLETE" if complete and not errors else "ERROR"
            self.state["finished_at"] = utc_now()
            if errors:
                self.state["errors"] = errors
            self.write(force=True)
            for dataset in self.datasets:
                self.metadata(dataset["name"])
            return self.state["state"] == "COMPLETE"


def hashes(path, job, progress):
    digests = {"sha256": hashlib.sha256()}
    if "md5" in job["expected"]:
        digests["md5"] = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            for digest in digests.values():
                digest.update(block)
            progress.write()
    result = {algorithm: digest.hexdigest() for algorithm, digest in digests.items()}
    for algorithm, expected in job["expected"].items():
        if result[algorithm] != expected:
            raise ValueError(algorithm + " mismatch")
    return result


def verified(path, job, progress):
    if path.stat().st_size != job["size"]:
        raise ValueError("size mismatch")
    progress.update(job, force=True, state="VERIFYING", bytes=job["size"])
    return hashes(path, job, progress)


def routes(url):
    addresses = [url]
    parsed = urlsplit(url)
    if parsed.hostname == "huggingface.co":
        addresses.append(urlunsplit((parsed.scheme, "hf-mirror.com", parsed.path,
                                     parsed.query, parsed.fragment)))
    return [(address, proxy) for address in addresses for proxy in (True, False)]


def transfer(session, job, part, address, proxy, progress):
    offset = part.stat().st_size if part.exists() else 0
    if offset > job["size"]:
        part.unlink()
        offset = 0
    headers = {"Accept-Encoding": "identity", "Cache-Control": "no-cache"}
    if offset:
        headers["Range"] = "bytes=%d-" % offset
    transport = "proxy" if proxy else "direct"
    progress.update(job, force=True, state="DOWNLOADING", bytes=offset,
                    download_url=address, transport=transport, error=None)
    proxies = {"http": PROXY, "https": PROXY} if proxy else {}
    with session.get(address, stream=True, headers=headers, proxies=proxies,
                     timeout=(10, 60), allow_redirects=True) as response:
        response.raise_for_status()
        if response.status_code == 206:
            match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if not match or int(match.group(1)) != offset:
                raise ValueError("invalid Content-Range for resume")
            if match.group(3) != "*" and int(match.group(3)) != job["size"]:
                raise ValueError("remote total size changed")
            mode = "ab" if offset else "wb"
        elif response.status_code == 200:
            # A server may ignore Range; never append a fresh full response.
            mode = "wb"
            offset = 0
        else:
            raise ValueError("unexpected HTTP status %d" % response.status_code)
        with part.open(mode) as handle:
            progress.update(job, force=True, bytes=offset)
            for block in response.iter_content(chunk_size=CHUNK_SIZE):
                if not block:
                    continue
                if offset + len(block) > job["size"]:
                    raise ValueError("response exceeds expected file size")
                handle.write(block)
                offset += len(block)
                progress.update(job, bytes=offset)
            handle.flush()
            os.fsync(handle.fileno())
        progress.update(job, force=True, bytes=offset)
        if offset != job["size"]:
            raise ValueError("incomplete response: %d/%d bytes" % (offset, job["size"]))


def download(job, progress):
    destination = job["destination"]
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        # Explicit per-request settings prevent global proxy/environment changes.
        session.trust_env = False
        if destination.is_file():
            try:
                result = verified(destination, job, progress)
            except ValueError as error:
                log("RECHECK " + job["key"] + ": " + str(error))
            else:
                progress.update(job, force=True, state="COMPLETE", bytes=job["size"],
                                skipped=True, **result)
                progress.metadata(job["dataset"])
                log("OK " + job["key"] + " (existing)")
                return
        last_error = "no download attempt"
        for cycle in range(4):
            for address, proxy in routes(job["url"]):
                try:
                    if part.exists() and part.stat().st_size == job["size"]:
                        try:
                            result = verified(part, job, progress)
                        except ValueError:
                            part.unlink()
                        else:
                            os.replace(str(part), str(destination))
                            progress.update(job, force=True, state="COMPLETE", bytes=job["size"],
                                            skipped=False, **result)
                            progress.metadata(job["dataset"])
                            log("OK " + job["key"] + " (resumed)")
                            return
                    transfer(session, job, part, address, proxy, progress)
                    try:
                        result = verified(part, job, progress)
                    except ValueError:
                        part.unlink()
                        raise
                    os.replace(str(part), str(destination))
                    progress.update(job, force=True, state="COMPLETE", bytes=job["size"],
                                    skipped=False, **result)
                    progress.metadata(job["dataset"])
                    log("OK " + job["key"])
                    return
                except (requests.RequestException, OSError, ValueError) as error:
                    # Log no signed redirect URL, token, or verbose response body.
                    if isinstance(error, requests.HTTPError):
                        last_error = "HTTP %s" % error.response.status_code
                    elif isinstance(error, requests.RequestException):
                        last_error = type(error).__name__
                    else:
                        last_error = str(error)[:240]
                    count = part.stat().st_size if part.exists() else 0
                    progress.update(job, force=True, state="RETRYING", bytes=count,
                                    error=last_error)
                    host = urlsplit(address).hostname
                    log("RETRY %s %s/%s: %s" % (job["key"],
                        "proxy" if proxy else "direct", host, last_error))
            if cycle < 3:
                time.sleep(min(2 ** (cycle + 1), 15))
        progress.update(job, force=True, state="ERROR", error=last_error)
        progress.metadata(job["dataset"])
        raise RuntimeError(job["key"] + ": " + last_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON download plan")
    parser.add_argument("--root", help="override plan.root")
    args = parser.parse_args()
    progress = None
    lock_handle = None
    try:
        root, datasets, jobs = load_plan(args.plan, args.root)
        root.mkdir(parents=True, exist_ok=True)
        lock_handle = acquire_root_lock(root)
        progress = Progress(root, datasets, jobs)
        for dataset in datasets:
            progress.metadata(dataset["name"])
        log("DOWNLOAD_START %d files, %.2f GiB, %d workers" % (
            len(jobs), sum(job["size"] for job in jobs) / 1024 ** 3, WORKERS))
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(download, job, progress): job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    future.result()
                except Exception as error:
                    message = str(error)[:500]
                    progress.update(job, force=True, state="ERROR", error=message)
                    errors.append(message)
        if not progress.finish(errors):
            log("ERROR: %d download(s) failed; rerun the same plan to resume" % len(errors))
            return 1
        log("DOWNLOAD_COMPLETE")
        return 0
    except BaseException as error:
        if progress is not None:
            try:
                progress.finish([type(error).__name__ + ": " + str(error)[:500]])
            except Exception:
                pass
        log("ERROR: %s: %s" % (type(error).__name__, str(error)[:500]))
        return 1
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


if __name__ == "__main__":
    sys.exit(main())
