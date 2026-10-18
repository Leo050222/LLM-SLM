# Caps25/30/35/40 split-GPU continuation

User explicitly requested switching from colocation to teacher/student on separate GPUs. Submitted106430 with afterany:106421; then cancelled only106421. Four H800 remain requested, now2pairs (teacher14B and student1.7B each on its own GPU), batch8,64trace concurrency,32CPU384GiB,UNLIMITED. Existing node wrapper cannot safely hot-switch; allocation is reacquired through scheduler.

Root is unchanged:opd_eval_prefix/prefix_takeover_caps25_40_20260918. Migration snapshot preserved1774completions (actual count can be slightly higher at shutdown), plus trace/timing checkpoints. Previous rolling120s completions264 at switch, earlier429; these windows vary, not a controlled paired benchmark. Compare deployment throughput cautiously across different dataset/prefix stages.

Migration permits only replicas/concurrency/placement/batch/KV/code-provenance changes. Model identities, tokenizer, caps, seeds, prefix hash, threshold2,k1,max generation length and branch logic unchanged. Target code hashes pinned. Actual source manifest migration and idempotency tested; model path and cap corruption rejected. All four branch tests passed. Old source manifest is stored in root/switch_split_20260918.json; active manifest updated only when new dispatcher starts.

Original grading scripts remain used to keep grading signature unchanged. External9600math/reasoning authorization still pending; internal16000code scoring automatically scheduled after generation. No monitor enabled.
