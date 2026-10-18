# All main math/reasoning x8 scoring, measured concurrency up to2000

Job107039 replaces107011 and resumes every saved score after SHA256 and exact existing r1 scoring-signature checks. All175 additional groups394450rows are included; first-round56350scores remain in their completed original scoring root. Scientific extraction, prompt, GPT-4o-mini request parameters and 900-second timeout count-wrong policy remain unchanged. Runtime source hashes are separately recorded.

Uses32CPU/96GiB,noGPU,UNLIMITED. Async HTTP through an authenticated CONNECT relay pinned to api.openlux.ai:443, end-to-end verified TLS, no redirects. File-descriptor soft limits raised to16384(client)/32768(relay). I/O32threads,2000async consumers,8192bounded queue,continuous scheduling across configuration boundaries.

Real pending experiment rows are scored once and retained during the256/512/1000/2000 sweep,90seconds per stage. Records actual peak in-flight,success/sec,429,transport/schemaerrors,p50/p95. On >1%429 or >5%transport/schemaerrors, stop escalating. Select smallest tested stable limit within10% of best measured throughput. Immediate429backoff halves live limit. Retries follow original8attemptpolicy; unresolved errors remain null, resumable, never counted as a negative.

Telemetry: progress.json and concurrency_test.json under opd_rescore/math_grid_score_all_2000_20260919. Final summary includes every group/dataset; only after all394450new scores are final and original56350completion is confirmed, write global450800ALL_SCORING_COMPLETE. Source generation integrity checked for each row. Tests: bounded gate, cooldown release, real asyncproxy/APIequivalence probe; full startup three-case API check.
