"""Resolve pinned, public evaluation files; does not download dataset payloads."""
import concurrent.futures
import fnmatch
import json
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = "/storage/yukaichengLab/wangxin/test_eiloeo/benchmarks/vision"
SPECS = [
    ("MathVista", "AI4Math/MathVista", "2b6ad69445fbb5695c9b165475e8decdbeb97747", ["data/test*.parquet"],
     "testmini: 1000 labeled questions; test: 5141 questions with hidden answers. Image bytes are embedded."),
    ("MathVision", "MathLLMs/MathVision", "2837ddb3f13abaf6b3997c12d80753e5470bd46a", ["data/test*.parquet"],
     "test: all 3040 labeled questions; testmini: overlapping 304-question subset. Image bytes are embedded."),
    ("MathVerse", "AI4Math/MathVerse", "3bc86196678bad115a923d2851c6821dbe235939", ["testmini.parquet", "testmini_text_only.parquet"],
     "All publicly released evaluation data: 3940 visual instances (788 base problems x 5 versions) plus 788 Text Only instances. The larger paper dataset is not publicly released."),
    ("CV-Bench", "nyu-visionx/CV-Bench", "bc284db50d036958861cb60cdd7b77612052ce0d", ["test_2d.parquet", "test_3d.parquet"],
     "Complete 2D and 3D evaluation (2638 questions). Images and answers are embedded."),
    ("BLINK", "BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040", ["*/val-*.parquet", "*/test-*.parquet"],
     "All 14 tasks, val and test. Val answers are public; test answers are hidden. Report actual parquet row counts (the upstream prose and metadata differ)."),
    ("MMStar", "Lin-Chen/MMStar", "bc98d668301da7b14f648724866e57302778ab27", ["mmstar.parquet"],
     "Complete 1500-question val split, including image bytes and answers."),
    ("RealWorldQA", "xai-org/RealworldQA", "17e7f75e092e47169732462ea3cdfebe911105dd", ["data/test-*.parquet"],
     "Complete 765-question test split, including image bytes and public answers."),
]
MMBENCH = [
    ("DEV_EN", 52285836, "30c05be8f2f347a50be25aa067248184"),
    ("DEV_CN", 52199109, "593f9b5f6bea453d870a798b34ae4f37"),
    ("TEST_EN", 72968653, "8d1a704e791df6f5d9fb3ef782b65365"),
    ("TEST_CN", 72832414, "156e6569f9547afb16603eca16e85373"),
]
CURRENT_TEST_RELEASE = {
    "TEST_EN": {
        "md5": "aab598a927c183aa6657de9097e2be09",
        "sha256": "c8fb8b7ead6e52e5c09f805049bc3f16740a2f8664d3a8d71e7294943584516a",
        "last_modified": "Tue, 24 Feb 2026 09:01:19 GMT",
    },
    "TEST_CN": {
        "md5": "4cba097fd698201a68632d31f07afdc2",
        "sha256": "9eff3496b2d050e2807ebb6abeaee40a1bd51c5b8636f0ce588881ad8a37bde1",
        "last_modified": "Tue, 24 Feb 2026 09:01:16 GMT",
    },
}


def pin_current_mmbench(dataset):
    """Record the actual official public-answer release and registry discrepancy."""
    discrepancies = {}
    for item in dataset["files"]:
        variant = item["path"].removeprefix("MMBench_").removesuffix("_V11.tsv")
        if variant not in CURRENT_TEST_RELEASE:
            continue
        current = CURRENT_TEST_RELEASE[variant]
        registered = next(row[2] for row in MMBENCH if row[0] == variant)
        item["md5"] = current["md5"]
        item["sha256"] = current["sha256"]
        discrepancies[item["path"]] = {
            "registered_md5": registered,
            "actual_md5": current["md5"],
            "actual_sha256": current["sha256"],
            "last_modified": current["last_modified"],
            "content_hash_origin": "Whole-file download of official OSS object, cross-checked with an independent download on another host.",
            "note": "Current official files include public answers. Their byte hashes differ from the VLMEvalKit registry; the registry values have not been represented as matching.",
        }
    dataset["registry_hash_discrepancy"] = discrepancies


def resolve(spec):
    name, repo, revision, patterns, notes = spec
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}"
    entries = []
    params = {"recursive": "true", "expand": "false", "limit": 1000}
    while url:
        response = requests.get(url, params=params, timeout=(15, 45))
        response.raise_for_status()
        entries.extend(response.json())
        url = response.links.get("next", {}).get("url")
        params = None
    selected = []
    docs = {"README.md", "LICENSE", "LICENSE.txt", "LICENSE.md", ".gitattributes"}
    for entry in entries:
        path = entry["path"]
        if entry.get("type") != "file":
            continue
        if path not in docs and not any(fnmatch.fnmatch(path, pattern) for pattern in patterns):
            continue
        item = {"path": path, "size": entry["size"],
                "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path, safe='/')}"}
        lfs = entry.get("lfs") or {}
        if lfs.get("oid"):
            item["sha256"] = lfs["oid"]
        selected.append(item)
    for pattern in patterns:
        if not any(fnmatch.fnmatch(item["path"], pattern) for item in selected):
            raise ValueError(f"No files matched {name}: {pattern}")
    if name == "BLINK":
        parquet = [x for x in selected if x["path"].endswith(".parquet")]
        assert len(parquet) == 28 and len({x["path"].split('/')[0] for x in parquet}) == 14
    return {"name": name, "repo_id": repo, "revision": revision,
            "source": f"https://huggingface.co/datasets/{repo}", "notes": notes,
            "files": sorted(selected, key=lambda x: x["path"])}


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        datasets = list(executor.map(resolve, SPECS))
    files = []
    for variant, size, md5 in MMBENCH:
        path = f"MMBench_{variant}_V11.tsv"
        files.append({"path": path, "size": size, "md5": md5,
                      "url": f"https://opencompass.oss-cn-shanghai.aliyuncs.com/utils/benchmarks/MMBench/{path}"})
    datasets.append({"name": "MMBench", "version": "1.1", "source": "https://github.com/open-compass/MMBench",
                     "checksum_source": "https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_mcq.py",
                     "notes": "Official v1.1 TSVs, EN/CN DEV/TEST. Image fields may reference other row indices; use the supplied offline loader. Answer availability is measured from the actual current files.",
                     "files": files})
    pin_current_mmbench(datasets[-1])
    order = ["MathVista", "MathVision", "MathVerse", "CV-Bench", "BLINK", "MMBench", "MMStar", "RealWorldQA"]
    datasets.sort(key=lambda item: order.index(item["name"]))
    result = {"root": ROOT, "purpose": "Complete publicly available evaluation data for eight requested visual benchmarks", "datasets": datasets}
    target = Path(__file__).with_name("plan.json")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for dataset in datasets:
        print(dataset["name"], len(dataset["files"]), sum(f["size"] for f in dataset["files"]), flush=True)
    print("TOTAL_BYTES", sum(f["size"] for d in datasets for f in d["files"]))


if __name__ == "__main__":
    main()
