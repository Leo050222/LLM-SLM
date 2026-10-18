#!/usr/bin/env python3
"""Check the configured external judge before starting a long GPU run."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchor_selector.openlux_client import OpenLuxClient, load_env_file
from online_rollout.core import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--secrets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    secrets = load_env_file(args.secrets)
    client = OpenLuxClient(cfg['api_url'], secrets.get('OPENLUX_API_KEY', ''), cfg['judge_model'],
                           temperature=cfg.get('judge_temperature', 0),
                           max_tokens=cfg.get('judge_max_tokens', 1200),
                           timeout_seconds=cfg.get('judge_timeout_seconds', 180), max_retries=0)
    started = time.monotonic()
    try:
        judgment, metadata = client.judge('Connectivity and schema check only. Mathematical prefix: "2 + 2 =". Student sampled token: "5". Teacher top token: "4". Classify this immediate contrast with your required JSON schema.')
    except Exception as error:
        # Keep credentials/provider bodies out of diagnostics.
        raise SystemExit('JUDGE_PROBE_FAILED ' + type(error).__name__) from None
    result = dict(status='JUDGE_PROBE_OK', judgment=judgment, metadata=metadata,
                  elapsed_seconds=time.monotonic() - started)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
