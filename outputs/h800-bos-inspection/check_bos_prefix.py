"""Read-only BOS prefix check. Credentials stay in memory and are never logged."""
import base64
import argparse
import functools
import json
import logging
import pathlib
import ssl
import subprocess
import sys

sys.path.insert(0, "/private/tmp/codex-bos-readonly-deps")
import certifi
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.protocol import HTTPS
from baidubce.retry.retry_policy import NoRetryPolicy
from baidubce.services.bos.bos_client import BosClient

logging.disable(logging.CRITICAL)
ssl._create_default_https_context = functools.partial(
    ssl.create_default_context, cafile=certifi.where()
)


def safe_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def safe_error(error):
    return {
        "exception_type": type(error).__name__,
        "http_status": safe_scalar(getattr(error, "status_code", None)),
        "error_code": safe_scalar(getattr(error, "code", None)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", choices=["awomo-data", "auwomo-data"], default="awomo-data")
    bucket = parser.parse_args().bucket
    response = subprocess.run(
        [
            "kubectl", "--kubeconfig", "/Users/leolee/Downloads/auwomo-h800-team",
            "--context", "cluster1-team-auwomo", "--request-timeout=15s",
            "-n", "dcloud-auwomo", "get", "secret",
            "jfs-private-prod-awomo-self-hosted-secret", "-o", "json",
        ],
        capture_output=True, text=True, check=True,
    )
    secret = json.loads(response.stdout)["data"]
    access_key = base64.b64decode(secret["access-key"])
    secret_key = base64.b64decode(secret["secret-key"])
    client = BosClient(BceClientConfiguration(
        credentials=BceCredentials(access_key, secret_key),
        endpoint="https://bj.bcebos.com", protocol=HTTPS,
        connection_timeout_in_mills=10000, retry_policy=NoRetryPolicy(),
    ))
    report = {
        "checked_at_client_date": "2026-10-11",
        "bucket": bucket,
        "endpoint": "https://bj.bcebos.com",
        "credential_source": "existing team BOS configuration; credentials not saved",
        "read_only": True,
        "checks": [],
    }
    for prefix, delimiter in [("wangxin/test_eiloeo/", None), ("wangxin/", "/")]:
        check = {"prefix": prefix, "delimiter": delimiter, "max_keys": 100}
        try:
            result = client.list_objects(
                bucket, prefix=prefix, delimiter=delimiter, max_keys=100
            )
            check.update({
                "success": True,
                "is_truncated": bool(getattr(result, "is_truncated", False)),
                "objects": [
                    {"key": safe_scalar(o.key), "size_bytes": o.size,
                     "last_modified": safe_scalar(getattr(o, "last_modified", None))}
                    for o in (getattr(result, "contents", None) or [])
                ],
                "common_prefixes": [
                    safe_scalar(o.prefix) for o in (getattr(result, "common_prefixes", None) or [])
                ],
            })
        except Exception as error:
            check.update({"success": False, **safe_error(error)})
        report["checks"].append(check)
        # An authorization failure cannot establish whether either prefix exists.
        if not check["success"]:
            break
    target = report["checks"][0]
    if target.get("success") and not target.get("objects"):
        check = {"exact_object_key": "wangxin/test_eiloeo", "operation": "HEAD"}
        try:
            client.get_object_meta_data(bucket, "wangxin/test_eiloeo")
            check["success"] = True
        except Exception as error:
            check.update({"success": False, **safe_error(error)})
        report["checks"].append(check)
    output = pathlib.Path(__file__).with_name(bucket + "-wangxin-test_eiloeo-check-20261011.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for check in report["checks"]:
        display = {k: v for k, v in check.items() if k not in {"objects", "common_prefixes"}}
        if "objects" in check:
            display["object_count_this_page"] = len(check["objects"])
            display["sample_objects"] = check["objects"][:15]
        if "common_prefixes" in check:
            display["common_prefix_count_this_page"] = len(check["common_prefixes"])
            display["sample_common_prefixes"] = check["common_prefixes"][:30]
        print(json.dumps(display, ensure_ascii=False), flush=True)
    print("LOCAL_REPORT=" + str(output))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"fatal_error": safe_error(error)}))
        raise SystemExit(1)
