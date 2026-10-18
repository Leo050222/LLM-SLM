from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from anchor_selector.core import build_candidates, decoded_prefix, trace_identity  # noqa: E402
from anchor_selector.openlux_client import (  # noqa: E402
    OpenLuxClient,
    extract_json_object,
    validate_judgment,
)
from anchor_selector.run_anchor_judge_batch import (  # noqa: E402
    is_billing_error,
    process_record,
)


class FakeTokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"piece-{token_id}"

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "|".join(str(value) for value in token_ids)


class CandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = {
            "prompt_token_ids": [1, 2],
            "response_token_ids": [10, 11, 12],
        }
        self.student = np.asarray([-1.0, -5.0, -1.0], dtype=np.float32)
        self.teacher = np.asarray([-3.1, -2.0, -3.0], dtype=np.float32)
        self.top_ids = np.asarray(
            [[20, 21], [22, 23], [24, 25]], dtype=np.int32
        )
        self.top_lp = np.asarray(
            [[-0.1, -1.0], [-0.2, -1.1], [-0.3, -1.2]], dtype=np.float32
        )

    def test_strict_absolute_threshold(self) -> None:
        candidates = build_candidates(
            self.trace,
            FakeTokenizer(),
            self.student,
            self.teacher,
            self.top_ids,
            self.top_lp,
            threshold=2.0,
        )
        # Positions 0 and 1 have gaps 2.1 and 3.0. Position 2 is exactly
        # 2.0 and must not pass the user's strict > threshold rule.
        self.assertEqual([row["response_position"] for row in candidates], [0, 1])
        self.assertEqual(candidates[0]["gap_direction"], "student_higher")
        self.assertEqual(candidates[1]["gap_direction"], "teacher_higher")

    def test_prefix_excludes_candidate_token(self) -> None:
        text, truncated, original_chars = decoded_prefix(
            self.trace, FakeTokenizer(), response_position=1, max_chars=100
        )
        self.assertEqual(text, "1|2|10")
        self.assertFalse(truncated)
        self.assertEqual(original_chars, len(text))


class JudgeParsingTests(unittest.TestCase):
    def test_fenced_json(self) -> None:
        value = extract_json_object(
            "```json\n"
            + json.dumps(
                {
                    "select_anchor": True,
                    "label": "reasoning_branch",
                    "confidence": 0.9,
                    "reason": "different subgoal",
                    "student_direction": "A",
                    "teacher_possible_direction": "B",
                }
            )
            + "\n```"
        )
        judgment = validate_judgment(value)
        self.assertTrue(judgment["select_anchor"])

    def test_model_boolean_controls_selection(self) -> None:
        judgment = validate_judgment(
            {
                "select_anchor": True,
                "label": "local_correction",
                "confidence": 0.4,
                "reason": "weak",
            }
        )
        self.assertTrue(judgment["select_anchor"])
        self.assertTrue(judgment["label_consistent"])

    def test_client_retries_malformed_judge_json(self) -> None:
        valid = json.dumps(
            {
                "select_anchor": False,
                "label": "surface_only",
                "confidence": 0.9,
                "reason": "formatting only",
                "student_direction": "same",
                "teacher_possible_direction": "same",
            }
        )

        class FakeClient(OpenLuxClient):
            def __init__(self) -> None:
                super().__init__(
                    api_url="https://example.invalid",
                    api_key="test-key",
                    model="test-model",
                    max_retries=1,
                )
                self.responses = ["{invalid", valid]

            def _request(self, user_prompt: str) -> dict[str, object]:
                del user_prompt
                content = self.responses.pop(0)
                return {
                    "id": "test-id",
                    "model": "test-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": content},
                        }
                    ],
                    "usage": {},
                }

        client = FakeClient()
        with patch("anchor_selector.openlux_client.time.sleep"):
            judgment, _ = client.judge("test")
        self.assertFalse(judgment["select_anchor"])
        self.assertEqual(client.responses, [])


class BatchedJudgeTests(unittest.TestCase):
    def test_billing_error_detection_is_specific(self) -> None:
        self.assertTrue(is_billing_error(RuntimeError("HTTP 402 payment required")))
        self.assertTrue(is_billing_error(RuntimeError("账户余额不足，请充值")))
        self.assertTrue(is_billing_error(RuntimeError("insufficient_quota")))
        self.assertFalse(is_billing_error(RuntimeError("HTTP 429 rate limit")))
        self.assertFalse(is_billing_error(RuntimeError("temporary upstream timeout")))

    def test_four_at_a_time_selects_earliest_yes_and_stops(self) -> None:
        trace = {
            "dataset_index": 7,
            "problem_id": "p7",
            "problem": "test problem",
            "prompt_token_ids": [1, 2],
            "response_token_ids": list(range(10, 22)),
        }
        candidates = []
        for position, token_id in enumerate(trace["response_token_ids"]):
            candidates.append(
                {
                    "candidate_index": position,
                    "response_position": position,
                    "student_token_id": token_id,
                    "student_token_piece": str(token_id),
                    "student_token_text": str(token_id),
                    "student_logprob": -1.0,
                    "teacher_logprob_on_student_token": -4.0,
                    "student_probability": 0.3,
                    "teacher_probability_on_student_token": 0.02,
                    "signed_logprob_gap_student_minus_teacher": 3.0,
                    "absolute_logprob_gap": 3.0,
                    "gap_direction": "student_higher",
                    "teacher_top_tokens": [],
                }
            )
        record = {
            "status": "ANCHOR_CANDIDATES_OK",
            **trace_identity(trace),
            "candidate_count": len(candidates),
            "trace": trace,
            "candidates": candidates,
        }

        class FakeRuntime:
            def __init__(self) -> None:
                self.calls = 0
                self.lock = threading.Lock()

            def prefix(self, trace_value: dict, position: int) -> tuple[str, bool, int]:
                del trace_value, position
                return "prefix", False, 6

            def client(self) -> "FakeRuntime":
                return self

            def judge(self, prompt: str) -> tuple[dict, dict]:
                position = next(
                    value
                    for value in range(12)
                    if f'"response_position": {value}' in prompt
                )
                with self.lock:
                    self.calls += 1
                # Candidate 7 returns before candidate 5, but position 5 must
                # still win because anchor selection is chronological.
                if position == 5:
                    time.sleep(0.03)
                selected = position in {5, 7}
                return (
                    {
                        "select_anchor": selected,
                        "label": "reasoning_branch" if selected else "same_reasoning",
                        "confidence": 0.9,
                        "reason": "test",
                        "student_direction": "s",
                        "teacher_possible_direction": "t",
                        "label_consistent": True,
                    },
                    {"usage": {"total_tokens": 1}},
                )

        runtime = FakeRuntime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_path = root / "record.json"
            record_path.write_text(json.dumps(record), encoding="utf-8")
            with ThreadPoolExecutor(max_workers=4) as api_executor:
                result = process_record(
                    record_path,
                    root / "output",
                    runtime,  # type: ignore[arg-type]
                    max_candidates=0,
                    candidate_batch_size=4,
                    api_executor=api_executor,
                )
            final = json.loads(
                (root / "output" / "traces" / "trace_00007" / "selected_anchor.json").read_text()
            )
        self.assertEqual(result["status"], "ANCHOR_SELECTED")
        self.assertEqual(final["anchor_response_position"], 5)
        self.assertEqual(runtime.calls, 8)


if __name__ == "__main__":
    unittest.main()
