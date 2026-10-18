# Existing-reasoning experiment: two-node continuation, 2000-second budget

User requested stopping 107636 and moving to two nodes with a 2000-second trace timeout. Old job was cancelled at 3h03m34s. New job 107811 requests two nodes, eight H800/64 CPU/512 GiB per node; eight student/teacher pairs and 256 traces share one locked dispatcher. Both nodes must pass GPU preflight, all eight service smokes, and all eight forced-trigger reprompt smokes before formal dispatch.

The same output root is continued: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_ablation/weak_reprompt_t2_k1_20260920`. Original manifest, 900-second timeout completions/checkpoints/timings and retained-output hashes are archived in `recovery_16gpu_2000s`. 2197 completed non-timeout outputs retained byte-for-byte, 1095 timeout checkpoints reopened, 2374 total remaining. Existing cumulative time is preserved: a trace that consumed 900 seconds has approximately 1100 additional seconds, not a fresh 2000 seconds. In-flight checkpoints resume; scheduler downtime excluded.

Models, full-existing-reasoning prompt, frozen inputs, sample identity, seeds, token ban, generation/scoring workers, token limits and sampling unchanged. Strict manifest migration permits only replica/concurrency/code identity and timeout updates. Nine local and remote tests passed, including cumulative timeout boundary, migration idempotency and rejection of seed changes. Model/prompt/sampling source files are byte-identical to the original job. Separate no-draft experiment 107647 remains unchanged.

Initial status: PENDING Priority. GPU runtime validation is gated within the allocation and has not yet run.
