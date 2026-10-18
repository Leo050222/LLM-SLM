# Random k8 deployment witness

Wait for8 model.json files in the run's queue replicas0–3 (student+teacher).
Then execute verbatim; this only coordinates RPC requests into the existing GPU job.

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_random_k8_cap8; export OPD_ONLINE_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911; /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python "$OPD_ONLINE_RUN/verify_live.py"'
```
Expected RANDOM_LIVE_OK,1–8 handoffs at synthetic positions0,8,...56,
with at most8tokens each and0 judge calls. Early EOS is allowed.
The synthetic trace is not an evaluation answer. Report actual handoff/token counts.
