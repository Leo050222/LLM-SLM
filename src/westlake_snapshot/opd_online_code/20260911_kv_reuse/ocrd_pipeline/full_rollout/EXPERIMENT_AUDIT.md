# Teacher full-rollout integrity audit

- Audit date: 2026-08-27
- Scope: `teacher_full_qwen3_14b_n8_seed20260826`
- Verdict: PASS for repairing the missing root GPU completion marker through the guarded repair job; this is not evidence that the original GPU wrapper reached its final marker write.
- Population: 13,244 selected-anchor traces, 8 rollouts each, 105,952 rollouts total.
- Strict audit job: 94329 (`COMPLETED 0:0`).
- Reviewed verifier SHA-256: `04a84adf7a136c3d4481f88cbe8c2124dd9b7822fb7c18aab0e7dff56cc48d93`.

The audit checks exact bundle metadata and artifact set, nonnegative token IDs, deterministic seeds, constrained first-token inequality, Top-16 records, termination semantics, token reconstruction, and tokenizer decode equality for every stored text field. It reported zero missing and zero extra artifacts, 121,247,514 generated tokens, 105,951 stop completions and one length completion.

Before marker repair, job 94329 also wrote a sorted SHA-256 manifest for all 13,244 bundle files. The repair job must confirm the audit state and verifier SHA, rerun strict validation, recompute all bundle hashes, byte-compare the manifests, reject known active writers, and only then write repair provenance and the GPU marker. The normal verifier must subsequently create the dataset manifest, summary, and `FULL_ROLLOUT_BATCH_COMPLETE` marker.

Non-blocking limits: model identity is recorded as a filesystem path rather than a model-weight digest; output-tree quiescence detects known rollout writers rather than arbitrary processes; downstream correctness claims require a separately bound ground-truth dataset digest.
