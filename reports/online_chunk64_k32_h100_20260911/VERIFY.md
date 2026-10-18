# H100 k32 live contract check

Reuse the frozen verl-opd environment. On the login host, wait until all eight
queue/replica_*/{student,teacher}/model.json files exist; do not launch GPU work
outside job102187. Then execute this exact command (the RPC work runs inside
that job; the command itself coordinates file requests only):

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'export OPD_CODE_ROOT=/storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8_2nodes; OPD_RUN=/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911; /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python "$OPD_RUN/validate_k32.py" --queue "$OPD_RUN/queue/replica_0" --input "$OPD_RUN/prompts1000.jsonl" --output "$OPD_RUN/live_contract_k32.json" --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B'
```

Expected LIVE_WORKER_CONTRACT_OK, bridge_tokens32, synthetic_takeovers2,
and alignment errors below0.05. These synthetic traces are not evaluation results.
Report deviations rather than altering the command or environment.
