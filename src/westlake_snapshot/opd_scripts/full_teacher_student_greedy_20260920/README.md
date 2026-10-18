# Current launch: only currently available healthy GPUs

Job 107918 RUNNING on gnho008, two GPU resources, 16 CPUs and128 GiB RAM. Still one serial job: student then teacher then code scoring. Does not reserve the whole node. Older pending 107911/107916 cancelled. 107912 failed before model loading because nvidia-smi does not support minor_number as a CSV query field; fixed by reading XML. 107917 passed GPU kernels but vLLM cannot parse CUDA_VISIBLE_DEVICES UUID strings; fixed using PCI_BUS_ID numeric indices whose torch-reported UUID must match the selected NVML UUID before loading.

Only GPUs in SLURM_STEP_GPUS/SLURM_JOB_GPUS (matched against XML minor_number), with <500 MiB used and >=74000 MiB free, and passing an actual seeded CUDA BF16 kernel are used. Replica/concurrency automatically match the selected count. Current allocation global IDs4,6: GPU-050edb0f-9a6e-1ad5-ee48-c5ca77592845 and GPU-a62b0061-fce8-b11c-d392-df53adab095b. Both pass UUID identity and kernel checks, both student engines ready; both greedy seed-invariance smokes passed and dispatch writes completed outputs (verified 2026-09-21 00:03 Asia/Shanghai). nvidia-smi index within a restricted allocation may be renumbered; UUID and minor number are the recorded durable identifiers. All original greedy protocol settings unchanged.

Historical records below are superseded by this current launch.

# Current launch: gnho008 healthy seven GPUs

Job 107911 replaces cancelled pending job 107883 at user request. One serial Slurm job, student then teacher then code-only scoring. Pinned gnho008, reserves all eight GPU resources to ensure exclusive device ownership, but starts only seven replicas (224 active traces) on physical nvidia-smi indices 0,1,3,4,5,6,7 selected by UUID. Physical GPU2 is excluded. User supplied tests show CUDA initialization fails on physical GPU2; previous worker rank7 was a CUDA ordinal, not a reliable physical index. Seven-card GPU preflight, seeded kernels and greedy service smokes gate generation. Local and remote six tests passed. Models, frozen prompts, greedy sampling, token limits, output root and timeout policy unchanged. Runtime startup remains pending resource allocation; node currently has three GPUs allocated to another job.

Historical launch notes follow; the current job and mapping above supersede prior resource placement.

# Full-dataset student/teacher greedy baselines

Submitted serial job 107883 on 2026-09-20 after the user requested one task. One node, 8 H800, 64 CPU, 512 GiB, UNLIMITED; all student generation first, then all teacher generation, then code-only scoring within the same job. Previous pending generation jobs 107880/107881 and dependent score 107882 cancelled before starting. Initial jobs 107875/107876 failed on gnho008: rank7 CUDA initialization reported no CUDA GPUs despite clean nvidia-smi memory preflight; other ranks passed seeded kernels. Added gnho008 to exclusions alongside 005/009/018. No generation had started.

Reuses the unchanged warm verl-opd environment. GPU memory preflight, seeded CUDA witnesses and eight service greedy seed-invariance smokes are required at each role stage. Remote preparation and six unit tests passed. New serial job is queued; model startup has not been verified.

Original main frozen prompts and family-local indices are preserved byte-for-byte: CRUXEval-I 800, CRUXEval-O 800, HumanEval+ 164, MBPP+ 378, LiveCodeBench 175, MATH500 500, OlympiadBench 674, BBH 1080. Each role generates 4571 completions, 9142 total. Exactly one completion per prompt/role, temperature=0, top_p=1, top_k=-1. Original prompt token IDs retain the established chat/thinking format. No saved response prefix or sampled historical baseline is reused. BF16, context40960, response40264, chunk64, batch8, concurrency256/node, and original 900-second cumulative active-wall-time limit are retained; timeout outputs count as wrong.

Output: /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_baselines/full_teacher_student_greedy_20260920
Code: /storage/yukaichengLab/wangxin/test_eiloeo/opd_scripts/full_teacher_student_greedy_20260920

Role-specific dispatcher locks permit concurrent student and teacher execution without sharing queues. Input hashes, greedy settings, model fingerprints, source hashes, checkpoints, final identities and exact coverage are recorded/validated. Six local unit tests and Python/shell syntax checks passed. GPU and remote data checks have NOT run.

