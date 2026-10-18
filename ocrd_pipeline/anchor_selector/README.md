# OCRD Anchor Selector

This module selects the first model-approved anchor candidate from one student
trace. It implements the strict candidate rule

\[
|g_t| = |\log \pi_S(y_t\mid s_t)-\log \pi_T(y_t\mid s_t)| > 2.
\]

For every candidate, it sends the prefix `s_t`, the sampled student token
`y_t`, and the teacher's top-16 next tokens with probabilities to
`claude-opus-5`. The judge is instructed to accept only a substantive logical
branch or local correction, not a formatting or wording difference. Candidates
are judged in chronological order and the first accepted position is written
to `selected_anchor.json`.

## Why the module has separate commands

The H800 cluster does not have working outbound access to the judge endpoint.
Therefore GPU scoring/prefix decoding and the remote judge must not be run in
one process:

1. `score-student` on a GPU node;
2. `score-teacher` on a GPU node;
3. `candidates` on either node;
4. `prepare-judge` on a machine that has the local tokenizer;
5. transfer the small judge bundle and run `judge-prepared` on a networked
   machine.

Student and teacher models are never loaded at the same time.

## Input trace

The input can be one JSON object, a JSON list, or JSONL. It must contain:

```json
{
  "problem": "optional but strongly recommended",
  "problem_id": "optional",
  "dataset_index": 0,
  "prompt_token_ids": [1, 2, 3],
  "response_token_ids": [4, 5, 6],
  "response": "optional decoded student response"
}
```

This is directly compatible with `rollouts.jsonl` produced by the existing
`scripts/run_opd_logit_diagnostic.py` pipeline.

Select a row using one of `--row-index`, `--dataset-index`, or `--problem-id`.
Use the same selector in every phase.

## Example

```bash
MODULE=/storage/yukaichengLab/wangxin/test_eiloeo/OPD/ocrd_pipeline/anchor_selector
TRACE=/path/to/rollouts.jsonl
RUN=/path/to/anchor_runs/example_000

python "$MODULE/run_anchor_selector.py" score-student \
  --trace "$TRACE" --dataset-index 0 --work-dir "$RUN"

python "$MODULE/run_anchor_selector.py" score-teacher \
  --trace "$TRACE" --dataset-index 0 --work-dir "$RUN"

python "$MODULE/run_anchor_selector.py" candidates \
  --trace "$TRACE" --dataset-index 0 --work-dir "$RUN"

python "$MODULE/run_anchor_selector.py" prepare-judge \
  --trace "$TRACE" --dataset-index 0 --work-dir "$RUN" \
  --max-candidates 10

# Copy judge_trace.json, judge_inputs.jsonl, candidate_manifest.json and
# candidates.jsonl to the same work-dir layout on a networked machine, then:
python "$MODULE/run_anchor_selector.py" judge-prepared --work-dir "$RUN"
```

The model paths default to the shared Qwen3-1.7B student and Qwen3-14B teacher.
Override them with `--student`, `--teacher`, and `--tokenizer` if needed.

## Artifacts

```text
student_scores.npz
student_manifest.json
teacher_scores.npz
teacher_manifest.json
candidates.jsonl
candidate_manifest.json
judge_trace.json
judge_inputs.jsonl
judge_input_manifest.json
judgments.jsonl
selected_anchor.json
```

`anchor_prefix_token_ids` ends immediately before the candidate student token.
It can therefore be passed directly to the next teacher-bridge module.

Judging is resumable. `judgments.jsonl` is rewritten atomically after every API
reply. A rerun skips positions already judged and stops immediately if a cached
accepted anchor exists.

## Configuration and secret handling

Non-secret defaults are in `config.json`. The requested API key is already in
`.secrets.env`, which is ignored by Git. The program never prints the key. An
environment variable named `OPENLUX_API_KEY` takes precedence when present.

The default API settings use OpenLux's OpenAI-compatible interface:

- `POST https://api.openlux.ai/v1/chat/completions`
- `Authorization: Bearer <API key>`
- OpenAI-compatible `messages`
- non-streaming JSON response

To verify connectivity and the requested model name without trace artifacts:

```bash
python "$MODULE/run_anchor_selector.py" probe-api --work-dir /tmp/anchor_api_probe
```

## Current scientific limitation

The judge receives next-token top-16 evidence exactly as requested. A single
token distribution may be insufficient to infer a future reasoning branch, so
the prompt deliberately treats insufficient evidence as a rejection. A later
version can add short teacher lookahead probes without changing the scoring or
artifact interface.

## Tests

```bash
python -m unittest discover -s "$MODULE/tests" -v
```
