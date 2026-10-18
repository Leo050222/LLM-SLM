# Documented live service witness

Existing warm environment: `/storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd`;
PyTorch 2.11.0, vLLM 0.23.0, Transformers 5.6.0. No package rebuild.
Slurm job 102139 owns two H800s on gnho033, one model per GPU. Workers initialize
asynchronously. Read-only readiness command (both model.json paths must exist):

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'ls /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911/queue/student/model.json /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911/queue/teacher/model.json'
```

Once both paths exist, execute this invocation verbatim. It enqueues short
inference/score requests to the allocated workers, never loads CUDA on login.

```bash
ssh -o BatchMode=yes -p 10025 wangxin@172.16.78.35 'export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false; /storage/yukaichengLab/wangxin/miniconda3/envs/verl-opd/bin/python /storage/yukaichengLab/wangxin/test_eiloeo/opd_online_code/20260911_chunk64_k8/ocrd_pipeline/online_rollout/validate_service.py --queue /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911/queue --input /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911/prompts8.jsonl --output /storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k8_smoke20260911/live_contract.json --tokenizer /storage/yukaichengLab/share/model/Qwen3-1.7B'
```

Expected sentinel: `LIVE_WORKER_CONTRACT_OK`, alignment error < .05 for both
models, eight-token bridge with a changed first token, and two synthetic
handoffs. Synthetic handoffs validate orchestration, not judge quality or task
accuracy. Do not repair code or substitute commands if it fails; report the
actual divergence. The parent conducts a separate real-judge smoke.
