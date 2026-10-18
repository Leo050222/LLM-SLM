"""Check only the requested BOS prefix against documented regional endpoints."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import pathlib
import subprocess

from check_bos_prefix import (
    BceCredentials, BceClientConfiguration, BosClient, HTTPS,
    NoRetryPolicy, safe_error, safe_scalar,
)

response = subprocess.run([
    "kubectl", "--kubeconfig", "/Users/leolee/Downloads/auwomo-h800-team",
    "--context", "cluster1-team-auwomo", "--request-timeout=15s",
    "-n", "dcloud-auwomo", "get", "secret",
    "jfs-private-prod-awomo-self-hosted-secret", "-o", "json",
], capture_output=True, text=True, check=True)
secret = json.loads(response.stdout)["data"]
credentials = BceCredentials(
    base64.b64decode(secret["access-key"]), base64.b64decode(secret["secret-key"])
)


def check(bucket, region):
    result = {"bucket": bucket, "region": region,
              "prefix": "wangxin/test_eiloeo/", "max_keys": 1}
    client = BosClient(BceClientConfiguration(
        credentials=credentials, endpoint="https://" + region + ".bcebos.com",
        protocol=HTTPS, connection_timeout_in_mills=8000,
        retry_policy=NoRetryPolicy(),
    ))
    try:
        response = client.list_objects(
            bucket, prefix="wangxin/test_eiloeo/", max_keys=1
        )
        result.update({"success": True,
                       "is_truncated": bool(getattr(response, "is_truncated", False)),
                       "sample_objects": [{"key": safe_scalar(o.key), "size_bytes": o.size}
                                          for o in (getattr(response, "contents", None) or [])]})
        parent = client.list_objects(bucket, prefix="wangxin/", delimiter="/", max_keys=1)
        result["wangxin_parent"] = {
            "object_count_this_page": len(getattr(parent, "contents", None) or []),
            "common_prefix_count_this_page": len(getattr(parent, "common_prefixes", None) or []),
            "is_truncated": bool(getattr(parent, "is_truncated", False)),
        }
    except Exception as error:
        result.update({"success": False, **safe_error(error)})
        cause = getattr(error, "last_error", None)
        if cause is not None:
            result["cause_type"] = type(cause).__name__
    return result


results = []
with ThreadPoolExecutor(max_workers=3) as pool:
    regions = ["bj", "bd", "su", "gz", "cd", "yq", "hkg", "fwh", "fsh"]
    futures = [pool.submit(check, bucket, region)
               for bucket in ["awomo-data", "auwomo-data"] for region in regions]
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
output = pathlib.Path(__file__).with_name("bos-data-all-regions-check-20261011.json")
output.write_text(json.dumps({"checked_at_client_date": "2026-10-11",
                             "region_source": "https://cloud.baidu.com/doc/BOS/s/akrqd2wcx",
                             "extra_legacy_region": "fsh",
                             "checks": sorted(results, key=lambda r: (r['bucket'], r['region']))},
                             indent=2, ensure_ascii=False) + "\n")
print("LOCAL_REPORT=" + str(output))
