#!/usr/bin/env python3
"""Serve the local gap page and lazily cache missing per-case data over SSH."""

from __future__ import annotations

import argparse
import http.server
import re
import subprocess
import threading
from pathlib import Path


CASE_RE = re.compile(r"^/data/case_(\d{4})\.js$")
REMOTE_DATA = Path(
    "/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/"
    "ge6_test1000_qwen3_1_7b_baseline_seed20260811/"
    "sampled_token_gap_visualization_compact/data"
)
SSH = [
    "ssh", "-o", "ConnectTimeout=120", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=12", "-p", "10024",
    "wangxin@172.16.78.35",
]


class GapHandler(http.server.SimpleHTTPRequestHandler):
    root: Path
    locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root), **kwargs)

    @classmethod
    def lock_for(cls, name: str) -> threading.Lock:
        with cls.locks_guard:
            return cls.locks.setdefault(name, threading.Lock())

    def do_GET(self) -> None:
        match = CASE_RE.fullmatch(self.path.split("?", 1)[0])
        if match:
            name = f"case_{match.group(1)}.js"
            local = self.root / "data" / name
            if not local.is_file():
                with self.lock_for(name):
                    if not local.is_file():
                        try:
                            result = subprocess.run(
                                [*SSH, "cat", str(REMOTE_DATA / name)],
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                timeout=240,
                            )
                            if not result.stdout.startswith(b'window.__dmGapReceive("'):
                                raise RuntimeError("unexpected remote case payload")
                            local.parent.mkdir(parents=True, exist_ok=True)
                            temporary = local.with_suffix(".js.tmp")
                            temporary.write_bytes(result.stdout)
                            temporary.replace(local)
                        except Exception as error:
                            self.send_error(503, f"Remote case cache unavailable: {error}")
                            return
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(root / "index.html")
    GapHandler.root = root
    server = http.server.ThreadingHTTPServer((args.host, args.port), GapHandler)
    print(f"DEEPMATH_GAP_SERVER=http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
