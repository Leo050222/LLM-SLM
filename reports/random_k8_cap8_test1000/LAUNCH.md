# Random teacher k8, max8 handoffs: DeepMath test1000

GPU102474 on exclusive nv-h100-007,8 H100,4 student/teacher replicas,
128 concurrent questions. CPU score array102475(3x64cores),aggregate102476.
Frozen code opd_online_code/20260911_random_k8_cap8. Run root:
/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_deepmath/random_k8_cap8_test1000_h100_seed20260911

Per-question positions are sampled uniformly from all legal non-overlapping
8token intervals in that question's existing independent baseline response
length. All1000 questions have8 planned positions. This uses baseline length
as an offline scheduling aid, not answer correctness or future online tokens.
Teacher ends early at EOS; positions beyond early termination are not reached.
The response/context limits remain40264/40960; seed20260911,temperature/top_p1,
64token student drafts,student1.7B,teacher14B. No external judge, no gap scores,
and no banned first teacher token. Compared with the targeted k8 experiment,
location selection,8-handoff cap,and first-token ban differ. This is therefore
not a pure single-variable ablation of anchor selection.

Four unit tests pass: non-overlap/schedule determinism,exactly8cap/full64teacher
tokens/checkpoint resume equality,early studentEOS,early teacherEOS. Live GPU
witness invocation in VERIFY.md. All questions and scoring match previous runs.

Live fresh-agent witness passed verbatim: RANDOM_LIVE_OK,8 takeovers,64teacher
tokens,all8segments exactly8tokens at positions0,8,...56,zero judge calls.
All8 worker CUDA H100 witnesses passed. Launch snapshot182completed at3m21s.

Completed:GPU102474 exited0:0 in6m57s; scoring102475 shards0:0 in44/38/37s;
aggregate102476 exited0:0 in5s. EVALUATION_COMPLETE present,1000scores.
Accuracy42.6%(426/1000),7031takeovers,56162teacher tokens,zero judge calls.
Full-cohort trace audit passed planned position identity,<=8 handoffs,
<=8tokens/bridge,complete status and source-length consistency.
Takeover count histogram:0:1,1:7,2:15,3:22,4:49,5:62,6:109,7:112,8:623.
24bridges ended below8tokens. All final resources released.
