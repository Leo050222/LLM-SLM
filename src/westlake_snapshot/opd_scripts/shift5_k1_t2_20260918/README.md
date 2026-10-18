# Local position shift ablation: k=1, pi=2, +/-1..5 tokens

One rollout for each of all 4,571 samples across eight datasets. Student Qwen3-1.7B, teacher Qwen3-14B. Same R1 generation seed 20260912, original dataset indices, non-thinking prompts, temperature/top-p 1/1, 64-token drafts, 40,264 output and 40,960 context limits. No intervention count cap and no anchor judge. Existing 900-second per-trace cumulative budget preserved. Generation only; answer scoring is separate.

Each round detects the earliest token with signed log P_student(token) minus log P_teacher(token) strictly greater than 2. Draw an independent reproducible uniform requested offset from {-5,-4,-3,-2,-1,+1,+2,+3,+4,+5}, using shift seed 2026091802 and sample/round identity. Replace the token at the shifted entry with a single teacher token, banning the original student token **at that shifted entry**, then resume student generation and detect new thresholds on the changed trajectory. Unlike the earlier quartile control, this experiment does not replay source positions or match source counts.

Left shifts can roll back committed student tokens across draft boundaries, but cannot erase a previous teacher intervention or prompt tokens: clamp to immediately after the latest teacher token, or response start. The resulting actual shift can be zero. Both requested/actual shifts and clipping are logged. Right shifts can cross a draft boundary: extend student generation only far enough to obtain the shifted entry token. If normal EOS occurs before reaching it, finish without a teacher call and record the skipped intervention. Output/context limits are handled identically. EOS from the teacher ends generation normally. A student EOS at the entry position can be replaced, as in the existing intervention protocol.

Trace events retain trigger candidate/gap, trigger and shifted positions, requested/actual shifts, left floor/clipping, shifted entry token/ban, bridge token, right extension, rollback and discard counts. Prior teacher tokens are preserved, so final teacher positions equal intervention-event positions. Rollback and handoff commit atomically; failed RPCs retain the previous consistent checkpoint.

Resources: one node, eight H800 GPUs, 64 CPUs, 512 GiB, UNLIMITED allocation. Four persistent student/teacher pairs, 128 concurrent traces. Warm verl-opd environment reused without changes. GPU-free-memory preflight, per-GPU seeded CUDA witness, four service smokes and four short actual shifted-rollout smokes precede formal dispatch. No periodic monitor is enabled.

Validation: 12 tests pass locally and remotely, including +/-5 across chunks, prompt/previous-teacher boundaries, EOS before target/in extension, max length, strict threshold, deterministic offsets, checkpoint resume, failed-teacher atomicity, full dispatcher startup and identical-manifest resume. Python and shell syntax checks passed.

Frozen prompt SHA256: 2c21e5857b8efbc41e1c5bb445d714e0dad41df39e0a7f28e93432aa8d9d64a1.
Remote code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/shift5_k1_t2_20260918
Remote output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/shift5_k1_t2_20260918
Group: r1_shift5_k1_t2
