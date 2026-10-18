# Four-way CPU scoring continuation

User authorized additional CPU nodes. Submitted array **106192_0..3**, 64 CPU / 128GiB each, 48 sandbox workers each, no GPUs, UNLIMITED. Automatic aggregation **106193**, 8CPU / 32GiB, afterok dependency on entire array. Replaced and cancelled only original scoring job106160 after successful submission; latest reported288000/463400 finished, actual atomic score files may include slightly more.

Wrapper uses unchanged original score.py and its original __file__, pinned by SHA256, so grading signature, test fixtures, extractors, sandbox, timeout handling and atomic cache remain identical. Original code remains untouched. Assign each of200 config/replicate groups by ordinal modulo4, 50 groups/115850 rows per shard. Each shard revalidates cached scores against source hash and scoring signature, then evaluates missing rows only. All shards hold shared original global scoring lock and unique shard locks; old exclusive process must release before any scoring proceeds. Source rollout files never changed.

Per-shard progress, errors and completion markers live under parallel_scoring_20260918 in existing output root. Aggregate requires all four markers with correct signature, processed count and zero infrastructure errors; then original1000-group/125-combination summary checks run. Original SCORING_PROGRESS.json is now historical; use per-shard progress instead. Cached counts are included in each shard processed count, not new work.

Validation: all four transformed shard programs and aggregation compile; disjoint assignment covers all200 groups. Real sandbox positive/negative smoke executes independently in every job. Math106176 remains user-cancelled; no monitor resumed.
