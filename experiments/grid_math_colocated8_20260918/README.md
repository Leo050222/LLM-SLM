# Math/reasoning grid continuation: 8 colocated pairs

Submitted job **106176** at 2026-09-18 10:29:36 Asia/Shanghai. Initial state PENDING (Resources). One exclusive node, 8 H800, 64 CPU, 768 GiB host RAM, UNLIMITED job time. No periodic monitor enabled.

Each GPU loads Qwen3-14B teacher and Qwen3-1.7B student in BF16, with explicit teacher/student KV caches 18/10 GiB. Batch size 4 per engine; 32 active traces per pair, 256 total. Teachers initialize before students. This configuration passed two-GPU benchmark job 106064; scaling to 8 colocated pairs has not yet run. Real startup runs CUDA witnesses and service smoke for all 8 pairs.

Resume existing math_reasoning_grid_x8_20260916: 175 additional-replicate config groups, 2,254 rows per group (MATH500, OlympiadBench, BBH). Preserve 87,339 completed additional-rollout outputs and 38 VERIFIED groups, plus original 56,350 baseline outputs outside this root. All existing trace/timing checkpoints retained.

Only deployment settings and worker provenance change. Thresholds, k, seeds, prompts, model/tokenizer/context, sampling and takeover rules unchanged. Existing cumulative 900-second active-wall-time per-trace timeout remains; timeout counts as incorrect and is not retried. Hardware/concurrency can affect which trajectories hit this deadline, and batching can affect numerical sampling: do not treat this as an unchanged deployment or use it for a clean speed comparison.

Preflight: shell syntax and Python compile passed; 16-process mock verified 8 colocated bindings, startup readiness and cleanup; all 175 actual source manifests matched original scientific settings and inputs; migration tested for idempotency and rejects changes to threshold or model path. Migration provenance saved remotely under recovery_colocated8_20260918/migration.json. No live manifests or outputs modified by preflight.

Coordinator retains allocation and engines across 5-hour dispatcher passes until all remaining generation completes; runtime failures stop with logs and retained checkpoints. This submission generates rollouts; no new external grading requests are initiated.
