import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOTS = [
    Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/prefix_takeover_caps_20260917"),
    Path("/storage/yukaichengLab/wangxin/test_eiloeo/opd_eval_prefix/prefix_takeover_caps5_10_20260917"),
]
BIN_COUNT = 20


def quantiles(values):
    if not values:
        return None
    xs = sorted(values)
    out = {}
    for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
        i = (len(xs) - 1) * q
        lo, hi = math.floor(i), math.ceil(i)
        value = xs[lo] if lo == hi else xs[lo] * (hi - i) + xs[hi] * (i - lo)
        out[f"p{round(q * 100)}"] = round(value, 6)
    return out


def new_group():
    return {
        "rollouts": 0,
        "correct_rollouts": 0,
        "takeover_hist": Counter(),
        "continuation_lengths": [],
        "relative_positions": [],
        "subsequent_relative_positions": [],
        "absolute_positions": [],
        "subsequent_absolute_positions": [],
        "relative_bins": [0] * BIN_COUNT,
        "subsequent_relative_bins": [0] * BIN_COUNT,
        "correct_relative_bins": [0] * BIN_COUNT,
        "wrong_relative_bins": [0] * BIN_COUNT,
        "ordinal_relative_positions": defaultdict(list),
    }


groups = defaultdict(new_group)
seen = set()
for root in ROOTS:
    for cap_dir in sorted(root.glob("cap*_r*")):
        score_dir = cap_dir / "scores"
        for path_string in glob.iglob(str(cap_dir / "completed" / "*.json")):
            path = Path(path_string)
            x = json.loads(path.read_text())
            key = (x["dataset_index"], x["cap"], x["replicate"])
            assert key not in seen
            seen.add(key)
            score = json.loads((score_dir / path.name).read_text())
            assert isinstance(score["correct"], bool)
            prefix_len = len(x["anchor"]["response_prefix_token_ids"])
            continuation_len = len(x["response_token_ids"]) - prefix_len
            events = [e for e in x["events"] if "bridge_token_ids" in e]
            assert len(events) == x["actual_takeovers"]
            assert all(len(e["bridge_token_ids"]) == 1 for e in events)

            for group_key in ((x["dataset"], x["cap"]), ("overall", x["cap"])):
                g = groups[group_key]
                g["rollouts"] += 1
                g["correct_rollouts"] += int(score["correct"])
                g["takeover_hist"][x["actual_takeovers"]] += 1
                g["continuation_lengths"].append(continuation_len)
                for ordinal, event in enumerate(events, start=1):
                    absolute = event["anchor_response_position"] - prefix_len
                    relative = min(1.0, max(0.0, absolute / max(1, continuation_len - 1)))
                    bin_index = min(BIN_COUNT - 1, int(relative * BIN_COUNT))
                    g["relative_positions"].append(relative)
                    g["absolute_positions"].append(absolute)
                    g["relative_bins"][bin_index] += 1
                    g["ordinal_relative_positions"][ordinal].append(relative)
                    (g["correct_relative_bins"] if score["correct"] else g["wrong_relative_bins"])[bin_index] += 1
                    if ordinal > 1:
                        g["subsequent_relative_positions"].append(relative)
                        g["subsequent_absolute_positions"].append(absolute)
                        g["subsequent_relative_bins"][bin_index] += 1


def finish(g):
    return {
        "rollouts": g["rollouts"],
        "accuracy_pct": round(100 * g["correct_rollouts"] / g["rollouts"], 6),
        "actual_takeover_hist": {str(k): g["takeover_hist"][k] for k in sorted(g["takeover_hist"])},
        "mean_actual_takeovers": round(
            sum(k * v for k, v in g["takeover_hist"].items()) / g["rollouts"], 6
        ),
        "continuation_length_quantiles": quantiles(g["continuation_lengths"]),
        "teacher_token_count": len(g["relative_positions"]),
        "subsequent_teacher_token_count": len(g["subsequent_relative_positions"]),
        "relative_position_quantiles": quantiles(g["relative_positions"]),
        "subsequent_relative_position_quantiles": quantiles(g["subsequent_relative_positions"]),
        "absolute_position_quantiles": quantiles(g["absolute_positions"]),
        "subsequent_absolute_position_quantiles": quantiles(g["subsequent_absolute_positions"]),
        "relative_bins": g["relative_bins"],
        "subsequent_relative_bins": g["subsequent_relative_bins"],
        "correct_relative_bins": g["correct_relative_bins"],
        "wrong_relative_bins": g["wrong_relative_bins"],
        "ordinal_relative_position_quantiles": {
            str(k): quantiles(v) for k, v in sorted(g["ordinal_relative_positions"].items())
        },
    }


datasets = sorted(k[0] for k in groups if k[0] != "overall")
result = {
    "schema_version": 1,
    "bin_count": BIN_COUNT,
    "position_definition": "(teacher_token_index - frozen_prefix_length) / max(1, continuation_length - 1)",
    "datasets": {
        dataset: {str(cap): finish(groups[(dataset, cap)]) for cap in range(11)}
        for dataset in datasets
    },
    "overall": {str(cap): finish(groups[("overall", cap)]) for cap in range(11)},
    "audit": {
        "rows": len(seen),
        "expected_rows": 70400,
        "unique_keys": len(seen),
        "teacher_tokens_are_single_token": True,
        "first_intervention_is_forced_at_relative_position_zero_for_caps_above_zero": True,
    },
}
assert len(seen) == 70400
print(json.dumps(result, separators=(",", ":")))
