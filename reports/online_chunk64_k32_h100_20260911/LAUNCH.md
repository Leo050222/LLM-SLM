# DeepMath test1000: online teacher k32 on H100

User-authorized fresh experiment. Teacher support length32 tokens; unlimited
per-question takeovers. Other scientific settings match the completed k8 run:
student Qwen3-1.7B, teacher Qwen3-14B,64-token drafts, strict absolute natural
logprob gap>2, same external Luna judge/config, test1000 prompts, one rollout,
seed20260911, native non-thinking, sampling temperature/top_p1, response40264
and context40960. The first teacher token cannot equal the rejected student
entry token. H100 hardware and4 replicas replace the earlier H8008 replicas.

- GPU job102187: exclusive nv-h100-007,8 H100,960GiB RAM,24h.
- Four student/teacher pairs,128 concurrent questions.
- Coordinator3152897 on login02; up to6 complete checkpoint recovery passes.
  These passes retry failed questions and do not limit teacher takeovers.
- Score array102188:3 jobs each64 CPU cores/128GiB,32 parallel scores each.
- Aggregate102189:4 CPU cores, after successful score array.
- All prompts copied byte-for-byte from k8; no completed answers, traces,
  judge decisions or GPU caches were reused in this new experiment.
- Runtime snapshot remains opd_online_code/20260911_chunk64_k8_2nodes;
  --teacher-tokens32 configures the existing parameterized implementation.

Remote run root:
/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/online_chunk64_k32_test1000_h100_seed20260911

Software spec unchanged, SHA256
5ae61e04e36b0afd56a5b580951741b3f4b16813c0a1acb521dfa0691c3e8107.
External judge preflight passed in2.82s. Live H100 kernel and32-token bridge
validation are recorded separately in live_contract_k32.json and worker logs.

Launch verified:8 workers ready with H100 CUDA witnesses. Fresh independent
agent ran VERIFY.md verbatim:32-token bridge,2 consecutive synthetic takeovers,
student alignment error0,teacher0.0011590095,exit0. Original8 EngineCore PIDs
map to distinct physical devices. Overlapping diagnostic steps exposed only7
GPUs through NVML; the omitted teacher PID121416 holds /dev/nvidia6, mapped
by /proc/driver/nvidia/gpus to UUID GPU-cb6499d9-f9bc-1da5-b258-02359d752545.
The7 other engine UUIDs were confirmed with nvidia-smi. This visibility anomaly
in diagnostic steps did not prevent original rank5 from serving requests.
Final launch snapshot:13/1000 completed,137 traces,zero question failures.
