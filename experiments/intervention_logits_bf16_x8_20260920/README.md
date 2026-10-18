# Full BF16 intervention logits, k1_t2, eight rollouts

## Submission status

Final: both model jobs completed successfully; verifier107892 COMPLETED 0:0 in2m45s. ALL_LOGITS_VERIFIED.json confirms399601events/404shards per role, full BF16[399601,151936] coverage, all file checksums and event IDs verified. Each role's production archive is121430981336bytes; combined242861962672bytes. Local receipt saved alongside this README. Historical launch progress below is superseded by this completion.

Launch verified:107906 and107907 both RUNNING on gnho008. Both passed CUDA/model smoke, historical calibration, longest-prefix replay and BF16 round-trip gates; formal archive writes observed (student>=12,182 events/12 shards, teacher>=4,093 events/4 shards). No longer merely queued.

Current recovery: removing008 exclusion let107900/107901 start on008, but both exited before torch initialization because device cgroups renumber the sole visible GPU while SLURM_JOB_GPUS retains global IDs. Fixed binding to use the sole visible UUID inside a single-GPU cgroup (still reject unhealthy UUID). Replacement normal-queue jobs: student107906, teacher107907, verifier107892 depends on both. No required node;008 is eligible. No model outputs were created by failed jobs. LAUNCH.json contains updated source hashes.

Latest correction: removed gnho008 from existing queued jobs107900/107901's exclusion lists using scontrol (job IDs and queue submission times preserved). No required node; remaining exclusions005/009/018. Before any outputs existed, added gnho008-only single-GPU UUID binding based on SLURM_JOB_GPUS and rejection of the known faulty physicalGPU2 UUID. Scientific replay unchanged; runtime source hash updated in LAUNCH.json. This supersedes the retained-whole-node-exclusion statement below.

Current (supersedes migrations below): user withdrew the gnho008 placement request because that node became occupied. Restored ordinary scheduling with separate single-H800 student **107900** and teacher **107901**, no required node, original unhealthy-node exclusions retained. Cancelled pending107898. CPU verifier107892 now depends on afterok107900:107901. Frozen data, output directory and scientific settings remain unchanged.

Latest migration: user requested gnho008. Combined two-H800 job **107898** targets gnho008; old pending jobs107890/107891 were cancelled, and CPU verifier107892 now depends on afterok107898. Initial migration snapshot: PENDING Priority. `launch_008.py` uses SLURM_JOB_GPUS physical IDs to select only allocated GPU UUIDs; it rejects the failing physicalGPU2 UUID. This avoids CUDA ordinal drift established by diagnostic107895. The scientific replay script and frozen inputs are unchanged. Historical submission details below are retained for provenance; consult LAUNCH.json for current jobs.

Submitted student job **107890**, teacher job **107891**, and afterok CPU verifier **107892**. Initial snapshot: both GPU jobs PENDING Priority; verifier PENDING Dependency. GPU smoke and formal shard production have not yet been observed. Preparation completed: 404 immutable shards, 18,599,681 input tokens per model, maximum actual prefix 36,644 tokens, 64 calibration prefixes. Local and independent remote documented CPU tests both passed all eight cases. Environment specification hash remains `5ae61e04e36b0afd56a5b580951741b3f4b16813c0a1acb521dfa0691c3e8107`; no installation or environment changes. See LAUNCH.json and INPUTS_COMPLETE.json for receipts.

User authorized storing full BF16 logits for both models at every intervention in the eight main rollouts. This is **fixed-prefix replay**, not recovery of unsaved historical generation tensors and not new generation. Historical vLLM outputs are read-only. The replay uses the unchanged warm verl-opd environment, Transformers SDPA, BF16 Qwen3-1.7B / Qwen3-14B, accepted token IDs and causal KV-cached chunks of 512 tokens. The backend differs from historical vLLM, and calibration errors are retained explicitly.

Scope: 36,568 traces, 399,601 intervention points, 8 datasets, r0–r7, exactly k=1 and threshold=2. The 816 traces without intervention are inventoried separately. Timeout partial traces and terminal/EOS interventions are included. Each event has one full 151,936-dimensional raw pre-softmax vector per model, before temperature or first-token banning. Model compute and archive dtype are BF16. Two-model tensor payload is 242,855,110,144 bytes (~242.86 GB decimal); inputs, checksums and metadata add modest overhead.

Output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_analysis/intervention_logits_bf16_x8_20260920`.
Scripts: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_logits_bf16_x8_20260920`.

The immutable input manifest stores hashes of original traces and their manifests. JSONL input shards snapshot exact prefix IDs and event metadata, including global event IDs, original trigger and replacement token, zero-based response position, prediction hidden-state position and SHA256 of uint32 little-endian prefix IDs. For prefix length n, hidden state n-1 predicts the intervention token; neither that token nor the rejected student token enters its own prefix. Event vectors use the same order for both models.

One non-exclusive H800 Slurm job per model, 8 CPUs/96 GiB each, unlimited allocation duration, no periodic resubmission. Known unhealthy nodes 005/008/009/018 are excluded. Each worker checks available memory and a seeded CUDA kernel; verifies tokenizer equality; then tests historical calibration on 64 first-event prefixes spanning all datasets and replicates, a real longest-context replay, independent fresh-prefill alignment controls, and a bitwise BF16 safetensors round trip. Calibration thresholds are gross-error gates, not claims of exact historical equivalence. The paired historical gap can be reconstructed from per-role calibration rows by event_id.

Each shard contains approximately <=1024 events and <=250,000 replay input tokens, with a whole-trace exception to avoid repeated prefills. Files are written atomically, read back for shape/dtype validation, and checksummed. Resume validates completed shard checksums and refuses changed model/runtime identity. The final CPU job verifies every checksum, BF16 shape and contiguous event-ID coverage for both models before writing `ALL_LOGITS_VERIFIED.json`.

## Invocations

Local CPU alignment tests (no packages required):

```bash
cd /Users/leolee/Documents/OPD/experiments/intervention_logits_bf16_x8_20260920
python3 -m unittest -v test_run.py
```

Remote non-launching CPU tests after sync:

```bash
ssh -o BatchMode=yes gpu-wangxin 'cd /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/intervention_logits_bf16_x8_20260920 && /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python -m unittest -v test_run.py'
```

Prepare once: `python run.py prepare`. Submit `gpu.slurm student` and `gpu.slurm teacher` with `sbatch`, then `verify.slurm` with `afterok:<student_job>:<teacher_job>`. Consult `LAUNCH.json` before submitting to avoid duplicate work. Only final verification is completion, not successful submission.

## Loading an event

```python
from safetensors import safe_open
with safe_open('/absolute/output/student/00000.safetensors', framework='pt', device='cpu') as f:
    event_ids = f.get_tensor('event_ids')
    logits = f.get_slice('logits')[0]  # BF16, complete model output vocabulary
    logprob = logits.float().log_softmax(-1)
```

Join row zero to the first event in `inputs/00000.jsonl`; subsequent tensor rows follow flattened trace/event order. Vocabulary includes all 151,936 model output dimensions, including unused tokenizer slots. Do not silently trim dimensions or normalize only top-k.
