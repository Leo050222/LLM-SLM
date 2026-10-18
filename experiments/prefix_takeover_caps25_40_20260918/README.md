# Fixed-prefix caps25/30/35/40 with four colocated pairs

Job106421 submitted2026-09-18 11:59:44 Asia/Shanghai. One node4H800,32CPU384GiB,UNLIMITED. Each GPU loads BF16 Qwen3-14B and Qwen3-1.7B; teacher/student KV18/10GiB, batch4,32 traces per pair/128total. Same tested deployment parameters as colocation benchmark106064. No changes to existing experiments.

Same800prefixes (100 each of8datasets) and8paired replicate seeds as caps0–20; SHA2563404d4b6e760c37e18844dae80d793cc60574553d49d7b37f086e533d174eae7. Threshold2,k1. Four caps25,30,35,40 yield25600rollouts. Initial forced teacher intervention at the frozen first anchor counts toward cap; after reaching cap only student continues. No scientific wall-time cutoff.7-day infrastructure lease raises errors preserving checkpoints.

Tests: local and remote branch tests passed for all new caps, zero/one boundaries, no teacher/scoring after cap, initial-anchor no-repeat on resume, EOS and input identity rejection. Shell syntax/Python compilation passed. Launcher mock verified4GPUs/8engines with non-contiguous visible devices, staged readiness and cleanup. Actual CUDA/model/service/code-scoring smoke runs after allocation.

Code16000outputs automatically scored internally after generation. Math/reasoning9600outputs require new-batch external grading authorization; pending async user question. Until authorized no approval marker or relay is installed, and GPU job schedules code-only scoring. If authorized before generation finishes, install restricted relay and approval marker to use full automatic scorer. Judge prompt already staged. Deployment fingerprints saved remotely. No recurring monitor enabled.
