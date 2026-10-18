#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from build_viewer_data import load_token_decoder


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".rollout_cache"
TOKENIZER = HERE.parent.parent / "tmp/anchor_smoke_local/tokenizer/tokenizer.json"
SSH = [
    "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", "-p", "10024",
    "wangxin@172.16.78.35",
]
REMOTE_BASE = "/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix"
ROOTS = {
    "low": f"{REMOTE_BASE}/prefix_takeover_caps_20260917",
    "high": f"{REMOTE_BASE}/prefix_takeover_caps5_10_20260917",
}
decode_token = load_token_decoder(TOKENIZER)


def fetch_rollout(sample_index: int, cap: int, replicate: int) -> dict:
    cache_key = f"{sample_index:05d}_c{cap}_r{replicate}.json"
    cache_path = CACHE / cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    root = ROOTS["low" if cap <= 4 else "high"]
    filename = f"{sample_index:05d}.json"
    completed = f"{root}/cap{cap}_r{replicate}/completed/{filename}"
    score = f"{root}/cap{cap}_r{replicate}/scores/{filename}"
    remote_code = (
        "import json;"
        f"x=json.load(open({completed!r}));s=json.load(open({score!r}));"
        "keep={k:x.get(k) for k in ['dataset','dataset_index','problem_id','problem','solution','cap','replicate','response','response_token_ids','token_sources','events','actual_takeovers','continuation_token_count','finish_reason']};"
        "keep['correct']=s.get('correct');"
        "keep['events']=[{k:e.get(k) for k in ['round','anchor_response_position','bridge_token_ids','forced_initial_anchor']} for e in x.get('events',[]) if 'bridge_token_ids' in e];"
        "keep['prefix_length']=len(x['anchor']['response_prefix_token_ids']);"
        "print(json.dumps(keep,separators=(',',':')))"
    )
    result = subprocess.run(
        SSH + ["python -c " + shlex.quote(remote_code)],
        text=True, capture_output=True, timeout=75, check=True,
    )
    row = json.loads(result.stdout)
    ids = row.pop("response_token_ids")
    row["token_ids"] = ids
    row["token_pieces"] = [decode_token(value) for value in ids]
    assert len(row["token_pieces"]) == len(row["token_sources"])
    CACHE.mkdir(exist_ok=True)
    temp = cache_path.with_suffix(".tmp")
    temp.write_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    os.replace(temp, cache_path)
    return row


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/cap-rollout":
            return super().do_GET()
        try:
            query = urllib.parse.parse_qs(parsed.query)
            sample_index = int(query["sample_index"][0])
            cap = int(query["cap"][0])
            replicate = int(query["replicate"][0])
            if not (0 <= sample_index < 800 and 0 <= cap <= 10 and 0 <= replicate < 8):
                raise ValueError("参数超出范围")
            body = json.dumps(fetch_rollout(sample_index, cap, replicate), ensure_ascii=False).encode()
            self.send_response(200)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode()
            self.send_response(502)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    os.chdir(HERE)
    ThreadingHTTPServer(("127.0.0.1", 9012), Handler).serve_forever()
