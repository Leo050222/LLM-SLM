# Two-node launch verification (agent follows documented invocation)

Warm verl-opd environment, unchanged worker kernels; the new execution layout is
8 independent student/teacher pairs on 2 exclusive H800 nodes. Read-only readiness:

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'find /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911/queue -name model.json | wc -l; test -f /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911/results/manifest.json && echo COORDINATOR_READY'
```

Wait for 16 records and COORDINATOR_READY, then execute verbatim:

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 '/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911/verify_scale.py /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_test1000_2nodes_seed20260911'
```

Expected TWO_NODE_16GPU_VERIFIED. Checks all 16 per-rank H800 CUDA witnesses,
unique physical GPU UUID binding within each node (Slurm remaps every process-visible CUDA index to0; node inventory maps EngineCore PID to physical UUID), 8 replicas with matching model
identities, 256 question concurrency, no takeover count limit, and preservation
of all migrated completed responses. No new inference or other job mutation is
needed. On failure report divergence; do not patch or substitute the invocation.
