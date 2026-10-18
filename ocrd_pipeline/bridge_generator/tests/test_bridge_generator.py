from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR.parent))

from bridge_generator.core import (  # noqa: E402
    assemble_bridge_artifact,
    load_selected_anchor,
    validate_bridge_tokens,
)
from bridge_generator.run_bridge_generator import bad_word_token_sequences  # noqa: E402
from bridge_generator.run_bridge_batch import (  # noqa: E402
    discover_selected_anchors,
    output_path,
    shard_records,
    valid_cached_bundle,
)
from bridge_generator.verify_bridge_batch import validate_artifact  # noqa: E402


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {"token": [7], " token": [8]}.get(text, [99])


class BridgeValidationTests(unittest.TestCase):
    def test_bad_word_preflight_covers_leading_space_variant(self) -> None:
        self.assertEqual(
            bad_word_token_sequences(FakeTokenizer(), " token"), [[7], [8]]
        )

    def test_first_token_must_differ(self) -> None:
        with self.assertRaisesRegex(ValueError, "equals the student"):
            validate_bridge_tokens([7, 8], student_token_id=7, requested_length=2)

    def test_bridge_length_is_exact(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_bridge_tokens([8, 9], student_token_id=7, requested_length=3)

    def test_load_selected_anchor_rejects_no_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.json"
            path.write_text(json.dumps({"status": "NO_ANCHOR_SELECTED"}))
            with self.assertRaisesRegex(ValueError, "not selected"):
                load_selected_anchor(path)

    def test_artifact_contains_resumable_prefix(self) -> None:
        anchor = {
            "status": "ANCHOR_SELECTED",
            "dataset_index": 3,
            "problem_id": "p3",
            "trace_sha256": "abc",
            "anchor_response_position": 4,
            "anchor_absolute_sequence_position": 6,
            "anchor_prefix_token_ids": [1, 2, 3],
            "student_candidate_token_id": 7,
            "student_candidate_token_text": "student",
        }
        artifact = assemble_bridge_artifact(
            anchor=anchor,
            anchor_path=Path("anchor.json"),
            teacher_model="teacher",
            tokenizer_path="tokenizer",
            requested_length=2,
            bridge_token_ids=[8, 9],
            bridge_token_texts=["a", "b"],
            bridge_text="ab",
            chosen_logprobs=[-0.1, -0.2],
            sampling={"seed": 1},
            first_step={"chosen_token_id": 8},
            continuation_finish_reason="length",
        )
        self.assertEqual(artifact["bridge_prefix_token_ids"], [1, 2, 3, 8, 9])
        self.assertTrue(artifact["first_token_differs_from_student"])

    def test_sharding_is_deterministic_by_dataset_index(self) -> None:
        rows = [
            (Path(f"{index}.json"), {"dataset_index": index}) for index in range(7)
        ]
        self.assertEqual(
            [row[1]["dataset_index"] for row in shard_records(rows, 1, 3, None)],
            [1, 4],
        )

    def test_discovery_accepts_nested_traces_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces" / "trace_00003" / "selected_anchor.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "status": "ANCHOR_SELECTED",
                        "dataset_index": 3,
                        "anchor_prefix_token_ids": [1],
                        "student_candidate_token_id": 2,
                    }
                )
            )
            rows = discover_selected_anchors(Path(directory))
            self.assertEqual([row[1]["dataset_index"] for row in rows], [3])

    def test_cached_and_strict_validation_accept_valid_artifact(self) -> None:
        anchor = {
            "status": "ANCHOR_SELECTED",
            "dataset_index": 3,
            "problem_id": "p3",
            "trace_sha256": "abc",
            "anchor_response_position": 4,
            "anchor_absolute_sequence_position": 6,
            "anchor_prefix_token_ids": [1, 2, 3],
            "student_candidate_token_id": 7,
            "student_candidate_token_text": "student",
        }
        artifact = assemble_bridge_artifact(
            anchor=anchor,
            anchor_path=Path("anchor.json"),
            teacher_model="teacher",
            tokenizer_path="teacher",
            requested_length=2,
            bridge_token_ids=[8, 9],
            bridge_token_texts=["a", "b"],
            bridge_text="ab",
            chosen_logprobs=[-0.1, -0.2],
            sampling={"seed": 1},
            first_step={"chosen_token_id": 8},
            continuation_finish_reason="length",
        )
        validate_artifact(artifact, anchor, 2, "teacher")
        with tempfile.TemporaryDirectory() as directory:
            artifact["rollout_index"] = 0
            bundle = {
                "status": "BRIDGE_ROLLOUTS_GENERATED",
                "trace_sha256": "abc",
                "teacher_model": "teacher",
                "requested_bridge_length": 2,
                "requested_rollouts": 1,
                "actual_rollouts": 1,
                "rollouts": [artifact],
            }
            path = output_path(Path(directory), 3, 2, 1)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(bundle))
            self.assertTrue(valid_cached_bundle(path, anchor, 2, 1, "teacher"))

    def test_strict_validation_rejects_corrupt_resumable_prefix(self) -> None:
        anchor = {
            "dataset_index": 1,
            "trace_sha256": "x",
            "anchor_prefix_token_ids": [1, 2],
            "student_candidate_token_id": 3,
        }
        value = {
            "status": "BRIDGE_GENERATED",
            "dataset_index": 1,
            "trace_sha256": "x",
            "teacher_model": "t",
            "requested_bridge_length": 1,
            "actual_bridge_length": 1,
            "bridge_token_ids": [4],
            "bridge_token_texts": ["a"],
            "bridge_chosen_logprobs": [-0.1],
            "bridge_first_token_id": 4,
            "first_token_differs_from_student": True,
            "anchor_prefix_token_count": 2,
            "bridge_prefix_token_ids": [1, 2, 99],
            "first_step": {"chosen_token_id": 4},
        }
        with self.assertRaisesRegex(ValueError, "resumable bridge prefix"):
            validate_artifact(value, anchor, 1, "t")


if __name__ == "__main__":
    unittest.main()
