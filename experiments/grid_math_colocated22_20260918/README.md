# Math/reasoning full x8 main experiment: 22 H800 continuation

User requested adding the 6 free GPUs on gnho020 to the existing 16 GPUs.
Jobs: 106827 (gnho020, 6 GPUs/48 CPUs/576 GiB), 106828 (gnho029 + gnho041, 8 GPUs/64 CPUs/768 GiB per node). Both UNLIMITED.
Same warm verl-opd environment, model worker byte-identical to the 16-GPU deployment. Each GPU holds teacher14B and student1.7B, KV18/10 GiB, batch4, 32 concurrent traces per GPU (704 total).

The old dispatcher106743 preloads all assigned work. Stop that job after helper preflight and before snapshot, then continue with immutable disjoint configuration ownership. The first new job snapshots all175 manifests and completed counts under setup/exclusive scheduler locks, assigning whole configs longest-remaining-first normalized by capacities8/8/6. Later jobs validate the snapshot without needing an exclusive scheduler lock held against active dispatchers.

Completed outputs, partial traces, random seeds, scientific settings and cumulative900-second timing files are preserved. No reset or retry of timeouts. Migration rejects changes outside execution/runtime/provenance. Whole-config ownership and per-shard exclusive locks prevent concurrent writers. Each allocation ends on its own shard completion; the last dispatcher writes the global completion marker after all175 configs verify.

Validation: shell syntax,175 actual-manifest migrations, repeat idempotency, rejection of changed seeds, worker-byte equality. Every allocated GPU receives free-memory preflight; all22 model pairs must pass generation, scoring and first-token-ban smoke checks. Service startup status is recorded separately from Slurm RUNNING.
