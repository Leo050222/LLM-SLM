# KV reuse live validation

Wait for all4 model.json files in cached/reference student/teacher queues.
Run this exact command. All GPU work goes to existing job102555 via FileRPC;
this login process only coordinates requests. Do not stop workers until root
finishes validation. Report any failure and leave workers available for diagnosis.

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_kv_reuse_v2; export OPD_KV_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/kv_reuse_validation_20260911_v2; /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python "$OPD_KV_RUN/validate.py"'
```

Expected KV_REUSE_VALIDATED_WITH_BF16_DRIFT and nonzero KV/sidecar hits,
including an exercised block-boundary repair. The 0.5 logprob guard catches
large mismatches only: it is NOT a numerical-equivalence acceptance criterion.
The first version failed its original 0.05 tolerance (max observed 0.249775).
Report this limitation, the measured max error and the independent no-cache
prefill-vs-decode control. Do not label the numerical equivalence check passed.
The script tests append, rollback, teacher entry with a token ban, the first
uncached target on a 512-token aligned prefix, and a long-prefix continuation.
Full results stay in remote validation.json. Leave workers running for root.
