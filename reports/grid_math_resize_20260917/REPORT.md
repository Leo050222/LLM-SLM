# Math/reasoning GPU allocation correction

The user's earlier request to reduce math/reasoning to4H800 was interrupted. On September17 at00:46, inspection found job105164 still using8H800. Code job105134 continued unaffected.

Stopped105164 and resumed as105199 with non-exclusive4H800, two student/teacher pairs,64 concurrent traces. Both service smoke checks passed; formal TRACE_DONE records confirmed generation. All35348 existing completed files passed a SHA256 preservation check. The user then corrected the request to keep8H800.

Stopped105199; its brief run added102 outputs. Resumed as105201 ongnho032 with8H800, four student/teacher pairs,128 concurrent traces. Slurm confirmed8GPU allocated and RUNNING. The successor script also requests8GPU. Source manifests and35450 completed-file hashes are archived under remote `recovery_8gpu_20260917`.

The original FIFO inference worker, dispatcher, queue RPC, cumulative900s budget, worker binding and node launcher are byte-identical. Resource-only migration passed checks for175 group manifests, repeated application, and rejection of seed changes. Model identities, thresholds, k, seeds, sampling and output limits remain unchanged. Trace and timing files were not edited. Resource changes can affect wall-time censoring; the short4GPU phase is recorded in per-group manifests and migration history.

Final verification: job105201 passed startup and generated new formal TRACE_DONE outputs across four replicas. All35450 pre-switch completed files passed SHA256 recheck with0 changed. Eight-GPU continuation is operational.
