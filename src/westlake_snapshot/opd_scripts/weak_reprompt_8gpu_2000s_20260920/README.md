# Single-node continuation with 2000-second cumulative timeout

User requested replacing pending two-node job 107811 with a single-node eight-H800 job. 107811 was cancelled without ever starting (elapsed 00:00:00). Replacement 107835 requests one node, eight H800, 64 CPUs, 512 GiB; four model pairs and 128 concurrent traces.

Continue the original full-existing-reasoning output root and already reopened timeout checkpoints. Preserve cumulative timing, 2000-second trace limit, 2197 completed results, and 2374 remaining tasks including 1095 reopened timeouts. Do not rerun reopen_timeouts.py. Models, prompt, sampling and input settings remain unchanged. Nine CPU tests passed locally and remotely, including cumulative timing and strict migration rejection. In-job service/reprompt smokes gate full dispatch. Initial state PENDING Priority.
