"""Use checksum-pinned public HTTP sources, then verify and publish the models."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


RUN = Path(__file__).resolve().parent
PYTHON = "/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python"


def main():
    plan = json.loads((RUN / "plan.json").read_text())
    spec = importlib.util.spec_from_file_location("resumable_download", RUN / "download.py")
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    original_routes = downloader.routes
    upstream = {file["url"]: file.get("upstream_url")
                for model in plan["datasets"] for file in model["files"]}

    def routes(url):
        result = original_routes(url)
        original = upstream.get(url)
        if original:
            result += original_routes(original)
        return result

    downloader.routes = routes
    downloader.WORKERS = 6
    sys.argv = [str(RUN / "download.py"), "--plan", str(RUN / "plan.json")]
    result = downloader.main()
    if result:
        return result
    return subprocess.run([PYTHON, "-u", str(RUN / "verify_models.py"),
                           "--plan", str(RUN / "plan.json")]).returncode


if __name__ == "__main__":
    sys.exit(main())
