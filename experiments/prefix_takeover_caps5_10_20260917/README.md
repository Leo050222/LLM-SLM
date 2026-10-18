# Fixed-prefix intervention caps 5–10

Submitted Slurm job105237 on2026-09-17. Requests one node8H800,64CPU,512GiB, four student/teacher pairs,128 traces. The standard exclusionsgnho009/018 remain. Existing three jobs untouched.

Exact byte-identical800 frozen prompts/prefixes and eight paired seeds from `prefix_takeover_caps_20260917`; input SHA2563404d4b6e760c37e18844dae80d793cc60574553d49d7b37f086e533d174eae7. Caps5,6,7,8,9,10, each800×8=6400 outputs:38400 total. Threshold2,k1, initial forced teacher intervention counts toward the cap. Stop invoking/scoring teacher at the cap. No900-second scientific cutoff; EOS and existing40264/40960 response/context limits. Core state machine and workers unchanged.

Local and remote tests cover0–10 cap adherence, no scoring after cap, fixed initial prefix, same initial teacher seed across caps, EOS, resume without duplicated intervention, and changed-input rejection.

Remote output: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/prefix_takeover_caps5_10_20260917`.
Remote runtime: `/storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/prefix_takeover_caps5_10_20260917`.

Automatic continuation uses8GPU segments. At completion,24000 code outputs are scored with the original isolated harness.14400 math/reasoning outputs are prepared for GPT-4o-mini with the original extraction and prompt, but external submission remains gated: automatic review rejected extending the previous12000-row authorization and requested explicit approval for this new14400-row batch. No external relay/approval marker was created before that approval.

User subsequently explicitly approved the additional14400-row OpenLux GPT-4o-mini scoring. Its separate authenticated relay is running; three synthetic probe cases passed. The approval marker is now present, so the final GPU segment automatically submits full scoring (24000 code +14400 math/reasoning). Prior blocked status above is resolved.

Submission verification:105237 PENDING(Resources),1node8H800. Slurm's provisional start estimate at02:19 was2026-09-17 03:57:11 ongnho031; this is not a guaranteed start time. No models have started for this queued job yet. Original three tasks were not interrupted.

## gnho018 recheck after user question
2026-09-17: Slurm confirmed gnho018 IDLE with no allocated CPU/GPU resources. A short one-GPU Slurm preflight rechecked actual CUDA execution in the unchanged verl-opd environment. nvidia-smi reported driver555.42.02 and0 volatile uncorrected ECC errors; PyTorch2.11.0+cu130 (CUDA13.0) failed CUDA initialization with `NVIDIA driver ... too old (found version12050)`. Preflight exited1 and released the temporary allocation. The historical018 exclusion remains necessary for this environment; task105237 was not retargeted or duplicated.


## User-requested two-node allocation
The one-node queued job105237 was cancelled before any generation started. Runtime now requests gnho020+gnho036,4H800/32CPU/256GiB per node, no exclusive-node reservation. Global ranks0–3 on node0 and4–7 on node1 map to four distinct queue replicas; node0 alone runs the shared128-trace dispatcher and verification. Scientific Python code, frozen prompts, seeds and scoring remain unchanged. Future segments retain this2×4 topology.

Replacement job105242 started immediately at2026-09-17 02:25:05 ongnho020 andgnho036. Slurm confirmed2nodes,8GPU total,4H800 per node,64CPU and512GiB total,OverSubscribe=OK. Node bindings: node0/gnho020 local CUDA devices0,1,2,3; node1/gnho036 local CUDA devices0,1,2,3 (Slurm-local numbering). Prior105237 cancelled; existing other three experiments remained running. Pending03:57 estimate no longer applies.

Live2×4 launch verification:8 distinct GPU UUIDs,8 CUDA witnesses,4 service smoke pairs and code scoring smoke passed. First131 formal completions covered all six caps with0 TRACE_ERROR; each completed output passed immutable-prefix and intervention-limit checks. Evidence saved remotely in LAUNCH_VALIDATION.json.
