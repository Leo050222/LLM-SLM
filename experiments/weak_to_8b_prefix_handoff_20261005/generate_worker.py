"""Generate strong-model greedy continuations for frozen weak-model prefixes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

ROOT = Path("/storage/yukaichengLab/wangxin/test_eiloeo")
OUT = ROOT / "opd_eval_prefix_handoff_weak_to_8b_20261005"
INPUTS = OUT / "inputs.jsonl"
MODEL = "/storage/yukaichengLab/share/model/Qwen3-8B"
MAX_CONTEXT = 40960
MAX_RESPONSE = 40264
PERCENTAGES = (10, 20, 30, 40, 50)
ARMS = (("weak17_to_8b", "weak17_response_token_ids"), ("weak4_to_8b", "weak4_response_token_ids"))


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def load_inputs():
    raw = INPUTS.read_bytes()
    manifest = json.loads((OUT / "INPUTS_COMPLETE.json").read_text())
    assert sha(raw) == manifest["inputs_sha256"]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == manifest["rows"] == 4571
    return rows, manifest


def main():
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    rows, manifest = load_inputs()
    signature = sha(json.dumps({"inputs": manifest["inputs_sha256"], "model": MODEL, "percentages": PERCENTAGES, "sampling": manifest["generation"], "version": 1}, sort_keys=True).encode())

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    torch.manual_seed(20261005 + rank)
    witness = torch.randn(128, 128, device="cuda")
    value = float((witness @ witness).sum())
    torch.cuda.synchronize()
    print(f"CUDA_WITNESS_OK rank={rank} local_rank={local_rank} device={torch.cuda.get_device_name()} value={value}", flush=True)
    del witness
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    engine = LLM(
        model=MODEL,
        tokenizer=MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=20261005,
        max_model_len=MAX_CONTEXT,
        max_num_seqs=8,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        disable_log_stats=True,
    )
    print(f"ENGINE_READY rank={rank} vllm={vllm.__version__}", flush=True)

    tasks = []
    task_number = 0
    for arm, weak_field in ARMS:
        for percentage in PERCENTAGES:
            for row in rows:
                if task_number % world_size == rank:
                    weak_ids = row[weak_field]
                    prefix_length = math.floor(len(weak_ids) * percentage / 100)
                    prefix = weak_ids[:prefix_length]
                    prompt = row["prompt_token_ids"] + prefix
                    maximum = min(MAX_RESPONSE - len(prefix), MAX_CONTEXT - len(prompt) - 1)
                    assert maximum > 0, (row["canonical_index"], arm, percentage, len(prompt), len(prefix))
                    destination = OUT / arm / f"cap{percentage}" / "completed" / f"{row['canonical_index']:05d}.json"
                    tasks.append((destination, arm, weak_field, percentage, row, prefix, prompt, maximum, signature))
                task_number += 1
    print(f"TASKS_ASSIGNED rank={rank} count={len(tasks)} world_size={world_size}", flush=True)

    def existing(task):
        destination = task[0]
        if not destination.exists():
            return False
        saved = json.loads(destination.read_text())
        assert saved["generation_signature"] == signature
        assert saved["canonical_index"] == task[4]["canonical_index"]
        assert saved["prefix_percentage"] == task[3]
        return True

    pending = [task for task in tasks if not existing(task)]
    print(f"TASKS_PENDING rank={rank} count={len(pending)}", flush=True)
    batch_size = 8
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [TokensPrompt(prompt_token_ids=task[6]) for task in batch]
        params = [SamplingParams(max_tokens=task[7], temperature=0.0, top_p=1.0, top_k=-1, n=1, seed=20261005) for task in batch]
        outputs = engine.generate(prompts, params, use_tqdm=False)
        assert len(outputs) == len(batch)
        for task, output in zip(batch, outputs, strict=True):
            destination, arm, weak_field, percentage, row, prefix, prompt, maximum, _ = task
            generated = output.outputs[0]
            continuation = list(generated.token_ids)
            assert len(continuation) <= maximum
            full_ids = prefix + continuation
            response = tokenizer.decode(full_ids, skip_special_tokens=True)
            record = {
                "canonical_index": row["canonical_index"],
                "dataset": row["dataset"],
                "problem_id": row["problem_id"],
                "solution": row.get("solution"),
                "metadata": row.get("metadata"),
                "arm": arm,
                "strong_model": "Qwen3-8B",
                "weak_model": "Qwen3-1.7B" if weak_field.startswith("weak17") else "Qwen3-4B",
                "prefix_percentage": percentage,
                "prefix_rule": "floor(weak_response_token_count * percentage / 100)",
                "weak_response_token_count": len(row[weak_field]),
                "prefix_token_count": len(prefix),
                "prefix_token_ids": prefix,
                "continuation_token_ids": continuation,
                "response_token_ids": full_ids,
                "response": response,
                "finish_reason": generated.finish_reason,
                "prompt_token_count": len(row["prompt_token_ids"]),
                "weak_correct": row["weak17_correct"] if weak_field.startswith("weak17") else row["weak4_correct"],
                "generation_signature": signature,
                "generated_at": time.time(),
            }
            atomic(destination, record)
        done = min(start + batch_size, len(pending))
        if done % 80 == 0 or done == len(pending):
            print(f"GENERATION_PROGRESS rank={rank} done={done} total={len(pending)}", flush=True)

    assert all(existing(task) for task in tasks)
    marker = OUT / f"worker_{rank:02d}_COMPLETE.json"
    atomic(marker, {"rank": rank, "world_size": world_size, "tasks": len(tasks), "signature": signature, "completed_at": time.time()})
    print(f"WORKER_COMPLETE rank={rank} tasks={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
