# gnho008 CUDA check

2026-09-20: Slurm short diagnostic jobs 107894 (numeric CUDA selectors) and 107895 (GPU UUID selectors), each requesting all eight currently unallocated H800 GPUs for at most five minutes. No resets, driver changes, or other jobs were interrupted.

UUID-selected test confirms seven physical GPUs pass CUDA initialization, seeded FP32 and BF16 1024×1024 matrix multiplication, causal BF16 SDPA, finite-result checks and synchronization. Passing physical nvidia-smi indices: 0,1,3,4,5,6,7. Their checksums agree and available memory is approximately 78.65 GiB each. These are short kernel checks, not a model/long-duration stress test.

Failing physical nvidia-smi index: **2**, UUID `GPU-c33c45c6-61d6-36bc-e5ac-6d80e3bc4968`. CUDA initialization raises `RuntimeError: No CUDA GPUs are available`, even though nvidia-smi lists the device with zero memory use. Root hardware/driver cause has not been determined.

Important correction: numeric CUDA_VISIBLE_DEVICES=7 failed in the first diagnostic, but CUDA numeric enumeration is shifted relative to nvidia-smi here. Numeric selector6 maps to physical index7 and passes. Thus earlier references to faulty "GPU7" did not identify the physical device correctly. Use UUID identity and ensure any production worker uses only GPUs allocated to it by Slurm; never work around the problem by selecting unallocated healthy UUIDs. A whole-node allocation can safely choose its healthy cards, or the scheduler's unhealthy GRES device should be disabled by an administrator.

Machine-readable proof: result_uuid.json. Existing logits jobs were not modified by this diagnostic.
