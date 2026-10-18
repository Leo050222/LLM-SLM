"""Pin the three requested official model repositories without downloading weights."""
import concurrent.futures
import json
from pathlib import Path
from urllib.parse import quote

import requests


NAMES = ["Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Thinking", "Qwen3-VL-8B-Thinking"]


def resolve(name):
    repo = "Qwen/" + name
    response = requests.get(f"https://huggingface.co/api/models/{repo}", timeout=(15, 45))
    response.raise_for_status()
    revision = response.json()["sha"]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}"
    params = {"recursive": "true", "expand": "false", "limit": 1000}
    files = []
    while url:
        response = requests.get(url, params=params, timeout=(15, 45))
        response.raise_for_status()
        for entry in response.json():
            if entry.get("type") != "file":
                continue
            path = entry["path"]
            item = {"path": path, "size": entry["size"],
                    "url": f"https://huggingface.co/{repo}/resolve/{revision}/{quote(path, safe='/')}"}
            if entry.get("lfs", {}).get("oid"):
                item["sha256"] = entry["lfs"]["oid"]
            files.append(item)
        url = response.links.get("next", {}).get("url")
        params = None
    assert any(item["path"].endswith(".safetensors") for item in files), name
    assert any(item["path"] == "config.json" for item in files), name
    return {"name": name, "repo_id": repo, "revision": revision,
            "source": "https://huggingface.co/" + repo, "precision": "original BF16",
            "files": sorted(files, key=lambda item: item["path"])}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        datasets = list(executor.map(resolve, NAMES))
    plan = {
        "root": "/storage/yukaichengLab/share/model/.staging/qwen3vl_20261016",
        "publish_root": "/storage/yukaichengLab/share/model",
        "state_dir": "/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/model_downloads/qwen3vl_20261016",
        "datasets": datasets,
    }
    target = Path(__file__).with_name("plan.json")
    target.write_text(json.dumps(plan, indent=2) + "\n")
    for model in datasets:
        print(model["name"], model["revision"], "files", len(model["files"]),
              "bytes", sum(item["size"] for item in model["files"]), flush=True)
    print("TOTAL_BYTES", sum(item["size"] for model in datasets for item in model["files"]))


if __name__ == "__main__":
    main()
