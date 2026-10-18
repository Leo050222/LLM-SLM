# Online multi-takeover test runner

Defaults: student chunk 64, strict absolute natural-log probability gap > 2,
teacher top-16 evidence, judge `select_anchor=true`, teacher bridge 8 tokens,
unlimited repeat takeovers within the response length budget. One rollout per
question by default. All 1000 test questions are included, including the 42
previously excluded for having no selected anchor.

Only the current prefix is shown to the judge. Candidates are evaluated in order;
a yes discards that candidate token and the remainder of the student draft.
Teacher's first token is forbidden from matching that student token (as in the
old bridge pipeline). After the bridge, generate a fresh student chunk. EOS ends
the answer; a hard context/response boundary can shorten the bridge. The old
bridge's ignore-EOS behavior is deliberately not used for online completion.

## Architecture and current limits

Two vLLM workers remain resident on separate GPUs. A networked login-node
coordinator calls the existing external judge API. A shared filesystem carries
atomic, content-addressed requests and cached results, so compute nodes need no
outbound internet. Each worker batches up to eight pending requests. The
coordinator works on eight questions concurrently. Only one worker per role and
one coordinator per output directory are permitted (file locks enforce this).

This first implementation prioritizes correctness: prefix caching is disabled
because prompt logprob scoring needs full alignment. Both models re-score the
student draft to obtain raw chosen-token probabilities; this avoids relying on
generation logprob processing. Teacher scores every draft position, not just its
8-token bridges. Long prefixes therefore incur substantial repeated prefill
cost; no throughput claim has been validated yet. RPC cache files retain full
prefixes and top-16 scores and can grow large. Keep them for resumability until
the experiment is complete. Judge prompts use the full prefix without silent
truncation. Sampling uses temperature=1 and top_p=1.

Each committed chunk/bridge is saved atomically. Restarting unchanged inputs and
configuration resumes; completed GPU and judge calls are reused. A code/config
change requires a new output directory. A model change requires a new queue.
GPU errors are cached as failures, never interpreted as a negative judge vote.
To retry a transient GPU error, stop the coordinator, fix/restart workers, and
remove only the relevant error JSON under queue/ROLE/results before resuming.
Judge failures leave the question incomplete; rerun unchanged after recovery.
Workers stop when queue/STOP exists or the Slurm job ends.

## Prepare the fixed test set

Run these commands in the cluster's validated environment. Set OPD_CODE_ROOT to
the actual deployed repository directory and OPD_PYTHON to its environment's
Python executable. All run outputs must live below test_eiloeo.

```bash
export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_seed20260911
mkdir -p "$OPD_ONLINE_RUN/logs"
"$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/prepare.py" \
  --input /storage/yukaichengLab/wangxin/test_eiloeo/opd_data/deepmath_ge6_opd/test_1000.parquet \
  --output "$OPD_ONLINE_RUN/prompts.jsonl" \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B
```

Original nested parquet and flat parquet are accepted. For exact reuse of an
existing baseline's prompts, pass its complete 1000-row rollout JSONL instead;
only prompt fields and metadata are retained, never its generated response.
Prepared input records source/output hashes and checks count, IDs and tokens.

## Launch workers and coordinator

```bash
sbatch --output="$OPD_ONLINE_RUN/logs/%j.out" \
  --error="$OPD_ONLINE_RUN/logs/%j.err" \
  "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/launch_workers.slurm"
```

Wait for both workers' queue/ROLE/model.json files on the first launch and verify
worker logs. From the networked node, expose the existing OPENLUX_API_KEY securely
in the environment if the selected endpoint requires it; no key belongs in code
or command history. Reuse the existing judge configuration:

```bash
"$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/run.py" \
  --input "$OPD_ONLINE_RUN/prompts.jsonl" \
  --output-dir "$OPD_ONLINE_RUN/results" --queue "$OPD_ONLINE_RUN/queue" \
  --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B \
  --judge-config "$OPD_CODE_ROOT/ocrd_pipeline/anchor_selector/config_gpt56_luna_openlux.json"
```

Before the full run, use a separate smoke directory/queue and a few prompt rows
with `--expected-rows N --max-tokens 256`. Confirm real GPU prompt-logprob
alignment, first-token ban, EOS handling and at least two handoffs. Local mocked
tests cannot establish GPU/API compatibility or throughput.

## Evaluate and inspect

After all questions complete, run scoring in the existing verl environment:

```bash
"$OPD_PYTHON" "$OPD_CODE_ROOT/ocrd_pipeline/online_rollout/evaluate.py" \
  --run-dir "$OPD_ONLINE_RUN/results"
```

The evaluator reuses full_rollout/analyze_teacher_rollout_shard.py's current
boolean-normalized scorer. It refuses to summarize an incomplete cohort. Results
include accuracy, takeover count, teacher tokens, draft/discarded tokens and
judge calls; provider usage lives in judge_cache metadata. This is single-rollout
accuracy, not the old 64-rollout pass@64 metric. Compare to the same 1000-question
population with identical scoring. `completed/` contains final responses and
per-token student/teacher source masks; `traces/` contains checkpoint/event data.
`run_status.json` records incomplete questions. No real GPU run is implied by
successful local unit tests.

```bash
python3 -m unittest discover -s ocrd_pipeline/online_rollout/tests -v
```
