# Qwen2.5-VL-32B-Instruct download

Requested official original BF16 checkpoint; GPU inference remains paused.

- Source: https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- Pinned revision: `7cfb30d71a1f4f49a57592323337a4a4727301da`
- Official files: 32; safetensors shards: 18
- Total repository bytes: 68,299,245,118
- Final remote directory: `/storage/yukaichengLab/share/model/Qwen2.5-VL-32B-Instruct`
- Staging root: `/storage/yukaichengLab/share/model/.staging/qwen25vl32b_20261016`
- Remote run directory: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_data/model_downloads/qwen25vl32b_20261016`
- Started process: 440724 (2026-10-16 14:45:20 Asia/Shanghai)

## State

COMPLETE and VERIFIED at 2026-10-16 16:08:21 Asia/Shanghai. All 32 files (68,299,245,118 bytes), 18 weight shards, weight SHA-256, safetensors header/index, and offline CPU processor checks passed. Final model directory is published. Evidence: `reports/COMPLETE.json`, `reports/VERIFIED.json`, and `reports/status.json`.

## Execution

The detached CPU-only runner uses 8 HTTP workers, resumable `.part` files, fixed upstream revisions and expected SHA-256 hashes. Every completed file is size-checked and hashed. The final verifier checks all weights, safetensors headers/index, model config and offline CPU image processing before publishing the real model directory. Shared-storage publication uses exclusive creation and hardlinks when no-replace directory rename is unsupported. Existing models are never replaced.

`status.json` in the run directory describes overall pipeline state; `status.json` in staging describes byte progress. `run.log` records retries and validation. Run `run.py` again only after confirming no previous download process is alive; its locks prevent duplicate writers.

## Read-only progress command

```bash
ssh gpu-wangxin 'python3 /storage/yukaichengLab/wangxin/test_eiloeo/opd_data/model_downloads/qwen25vl32b_20261016/progress.py'
```

The 7B checkpoint was used only for a CPU processor compatibility precheck; that passed. The new 32B checkpoint will still receive its own full validation after transfer. Independent Hugging Face / ModelScope metadata cross-check is in `sources.json`; 18/18 weight sizes and hashes match.
