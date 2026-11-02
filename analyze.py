"""Summarize saved LLM-judge results.

Usage:
    python analyze.py
    python analyze.py --results results --out analysis
    python analyze.py --models qwen3:4b --biases fallacy position

Reads results/<model>__<bias>.json files written by judge.py. Writes CSV
tables and PNG charts under analysis/. Charts have French labels for the
presentation; where the paper reports the same metric, its six judges are
drawn as a box plot and our models as points.

For each pairwise bias, D is the set of items whose unperturbed reference
judgment y parsed. Robustness Rate is I(y == y_hat) over D, and Consistency
Rate is I(y == y_rand) over the items in D that have a <reference>_rand
repeat. A perturbed or repeated judgment that failed to parse counts as a
disagreement. RR is the headline metric for every pairwise bias except
fallacy: its rewrite flips which answer is better, so its headline is accuracy
against the expected answer.

Refinement uses the refined_given_history condition;
files from before that prompt existed have no Err_RA.

Self-enhancement ErrorRate_SE is y_self / y_other - 1. y_self is the score
a model gives its own generated answer. y_other is the mean score other
models give that same answer.

position.csv splits the position pairs into: same answer in both orders,
same slot A in both orders, same slot B in both orders, and other.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent

SKIP_PAIRWISE = {"refinement", "self-enhancement"}

REFERENCE_CONDITIONS = {"position": "order_ab"}
DEFAULT_REFERENCE = "baseline"
RAND_SUFFIX = "_rand"

ACCURACY_BIASES = {"fallacy"}
WILSON_Z = 1.96

HISTORY_CONDITION = "refined_given_history"

ROBUST_COLUMNS = [
    "model",
    "bias",
    "condition",
    "metric",
    "n",
    "n_failed",
    "rr",
    "acc_baseline",
    "acc_condition",
    "headline",
]
CONSISTENCY_COLUMNS = [
    "model",
    "bias",
    "n_cr",
    "n_cr_failed",
    "cr",
    "cr_low",
    "cr_high",
]
SUMMARY_COLUMNS = ["model", "bias", "metric", "n", "rr", "headline", "n_cr", "cr"]
COT_COLUMNS = ["model", "n", "acc_ori", "acc_cot", "air"]
REFINEMENT_COLUMNS = [
    "model",
    "n",
    "mean_original",
    "mean_refined",
    "mean_with_history",
    "error_rate_ra",
    "mean_item_ratio",
]
SELF_COLUMNS = [
    "model",
    "n_self",
    "n",
    "n_other_judges",
    "mean_self",
    "mean_other",
    "error_rate_se",
    "mean_item_ratio",
]
PARSE_COLUMNS = ["model", "bias", "condition", "total", "parsed", "failed"]
POSITION_COLUMNS = ["model", "n", "consistent", "always_a", "always_b", "other"]

VERDICTS = {"A", "B", "C"}
RATING_MIN = 1
RATING_MAX = 10


def reference_condition(bias):
    return REFERENCE_CONDITIONS.get(bias, DEFAULT_REFERENCE)


def rand_condition(bias):
    return reference_condition(bias) + RAND_SUFFIX


def headline_metric(bias):
    return "acc" if bias in ACCURACY_BIASES else "rr"


def wilson_interval(successes, n, z=WILSON_Z):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def map_choice(bias, condition, verdict):
    """Map a verdict letter onto the underlying answer slot.

    Position's swapped order puts answer1 in slot B, so the same letter is
    not the same answer. A tie never matches either slot.
    """
    if verdict == "C" or verdict is None:
        return "tie"
    if bias == "position" and condition == "order_ba":
        return {"A": "slot2", "B": "slot1"}.get(verdict, verdict)
    return {"A": "slot1", "B": "slot2"}.get(verdict, verdict)


def is_parsed(verdict, rating):
    """A verdict letter, or an integer rating on the prompts' 1 to 10 scale."""
    if verdict in VERDICTS:
        return True
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        return False
    if isinstance(rating, float) and not rating.is_integer():
        return False
    return RATING_MIN <= rating <= RATING_MAX


def _missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def mean_match(verdicts, expected):
    flags = []
    for verdict, target in zip(verdicts, expected):
        if _missing(target):
            continue
        flags.append(verdict == str(target).strip().upper())
    if not flags:
        return float("nan")
    return sum(flags) / len(flags)


def load_records(results_dir, models=None, biases=None):
    rows = []
    for path in sorted(results_dir.glob("*__*.json")):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        model = payload.get("model")
        bias = payload.get("bias")
        if not model or not bias:
            print(f"Skipping {path.name}: missing model or bias")
            continue
        if models and model not in models:
            continue
        if biases and bias not in biases:
            continue
        for record in payload.get("results", []):
            output = record.get("output") or {}
            verdict = output.get("verdict")
            rating = output.get("rating")
            if isinstance(verdict, str):
                verdict = verdict.strip().upper()
            rows.append(
                {
                    "model": model,
                    "bias": bias,
                    "index": record.get("index"),
                    "condition": record.get("condition"),
                    "kind": record.get("kind"),
                    "verdict": verdict,
                    "rating": rating if is_parsed(None, rating) else None,
                    "unbiased_verdict": record.get("unbiased_verdict"),
                    "parsed": is_parsed(verdict, rating),
                    "generated_answer": record.get("generated_answer"),
                    "source": path.name,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        ["model", "bias", "index", "condition"], keep="last"
    ).reset_index(drop=True)


def parse_summary(frame):
    if frame.empty:
        return pd.DataFrame(columns=PARSE_COLUMNS)
    summary = frame.groupby(["model", "bias", "condition"], as_index=False).agg(
        total=("parsed", "size"),
        parsed=("parsed", "sum"),
    )
    summary["parsed"] = summary["parsed"].astype(int)
    summary["failed"] = summary["total"] - summary["parsed"]
    return (
        summary[PARSE_COLUMNS]
        .sort_values(["model", "bias", "condition"])
        .reset_index(drop=True)
    )


def _by_index(group, condition):
    rows = group[group["condition"] == condition]
    return rows.drop_duplicates("index", keep="last").set_index("index")


def _agreement(bias, reference, ref_name, other, condition):
    """Compare y with another judgment over the parsed reference items.

    Returns the shared reference rows, the other rows, and one agreement flag
    per item. A judgment that failed to parse never agrees with y.
    """
    shared = reference.index.intersection(other.index)
    base = reference.loc[shared]
    other = other.loc[shared]
    agree = [
        bool(parsed)
        and map_choice(bias, ref_name, left) == map_choice(bias, condition, right)
        for left, right, parsed in zip(
            base["verdict"], other["verdict"], other["parsed"]
        )
    ]
    return base, other, agree


def pairwise_metrics(frame):
    """Return (robustness, consistency) tables for the pairwise biases."""
    empty = (
        pd.DataFrame(columns=ROBUST_COLUMNS),
        pd.DataFrame(columns=CONSISTENCY_COLUMNS),
    )
    if frame.empty:
        return empty
    pairwise = frame[(frame["kind"] == "pairwise") & ~frame["bias"].isin(SKIP_PAIRWISE)]
    rows = []
    consistency = []
    grouped = pairwise.groupby(["model", "bias"], sort=True)
    for (model, bias), group in grouped:
        ref_name = reference_condition(bias)
        rand_name = rand_condition(bias)
        reference = _by_index(group, ref_name)
        reference = reference[reference["parsed"].astype(bool)]
        if reference.empty:
            print(f"No parsed {ref_name} judgments for {model} / {bias}; skipping RR")
            continue

        repeat = _by_index(group, rand_name)
        if repeat.empty:
            print(f"No {rand_name} judgments for {model} / {bias}; CR not computed")
        _, repeat, agree = _agreement(bias, reference, ref_name, repeat, rand_name)
        low, high = wilson_interval(sum(agree), len(agree))
        consistency.append(
            {
                "model": model,
                "bias": bias,
                "n_cr": len(agree),
                "n_cr_failed": int((~repeat["parsed"].astype(bool)).sum()),
                "cr": sum(agree) / len(agree) if agree else float("nan"),
                "cr_low": low,
                "cr_high": high,
            }
        )

        conditions = sorted(
            condition
            for condition in group["condition"].unique()
            if condition not in (ref_name, rand_name)
        )
        metric = headline_metric(bias)
        for condition in conditions:
            perturbed = _by_index(group, condition)
            base, other, agree = _agreement(
                bias, reference, ref_name, perturbed, condition
            )
            n = len(agree)
            rr = sum(agree) / n if n else float("nan")
            acc_condition = mean_match(other["verdict"], other["unbiased_verdict"])
            rows.append(
                {
                    "model": model,
                    "bias": bias,
                    "condition": condition,
                    "metric": metric,
                    "n": n,
                    "n_failed": int((~other["parsed"].astype(bool)).sum()),
                    "rr": rr,
                    "acc_baseline": mean_match(
                        base["verdict"], base["unbiased_verdict"]
                    ),
                    "acc_condition": acc_condition,
                    "headline": acc_condition if metric == "acc" else rr,
                }
            )
    robust = pd.DataFrame(rows, columns=ROBUST_COLUMNS)
    return robust, pd.DataFrame(consistency, columns=CONSISTENCY_COLUMNS)


def bias_summary(robust, consistency):
    """One row per model and bias: n-weighted RR and headline, plus CR."""
    if robust.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows = []
    for (model, bias), group in robust.groupby(["model", "bias"], sort=True):
        usable = group[group["n"] > 0]
        weights = usable["n"]
        total = int(weights.sum())

        def weighted(column):
            values = usable[column]
            mask = values.notna()
            if not mask.any():
                return float("nan")
            return float((values[mask] * weights[mask]).sum() / weights[mask].sum())

        rows.append(
            {
                "model": model,
                "bias": bias,
                "metric": headline_metric(bias),
                "n": total,
                "rr": weighted("rr"),
                "headline": weighted("headline"),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.merge(
        consistency[["model", "bias", "n_cr", "cr"]], on=["model", "bias"], how="left"
    )
    return summary[SUMMARY_COLUMNS]


def cot_metrics(frame):
    if frame.empty:
        return pd.DataFrame(columns=COT_COLUMNS)
    cot = frame[(frame["bias"] == "cot") & (frame["kind"] == "pairwise")]
    rows = []
    for model, group in cot.groupby("model", sort=True):
        baseline = group[group["condition"] == "baseline"]
        perturbed = group[group["condition"] == "cot"]
        if baseline.empty or perturbed.empty:
            print(f"Chain-of-thought for {model} is missing baseline or cot")
            continue
        baseline = baseline.drop_duplicates("index", keep="last").set_index("index")
        perturbed = perturbed.drop_duplicates("index", keep="last").set_index("index")
        paired = [
            index
            for index in baseline.index.intersection(perturbed.index)
            if baseline.at[index, "parsed"] and perturbed.at[index, "parsed"]
        ]
        if not paired:
            rows.append(
                {
                    "model": model,
                    "n": 0,
                    "acc_ori": float("nan"),
                    "acc_cot": float("nan"),
                    "air": float("nan"),
                }
            )
            continue
        acc_ori = mean_match(baseline.loc[paired, "verdict"], ["A"] * len(paired))
        acc_cot = mean_match(perturbed.loc[paired, "verdict"], ["A"] * len(paired))
        if acc_ori == 0 or math.isnan(acc_ori):
            air = float("nan")
        else:
            air = (acc_cot - acc_ori) / acc_ori * 100
        rows.append(
            {
                "model": model,
                "n": len(paired),
                "acc_ori": acc_ori,
                "acc_cot": acc_cot,
                "air": air,
            }
        )
    if not rows:
        return pd.DataFrame(columns=COT_COLUMNS)
    return pd.DataFrame(rows)[COT_COLUMNS]


def refinement_metrics(frame):
    if frame.empty:
        return pd.DataFrame(columns=REFINEMENT_COLUMNS)
    scored = frame[
        (frame["bias"] == "refinement") & (frame["kind"] == "score") & frame["parsed"]
    ]
    rows = []
    for model, group in scored.groupby("model", sort=True):
        wide = group.pivot_table(
            index="index", columns="condition", values="rating", aggfunc="last"
        )
        needed = ["refined", HISTORY_CONDITION]
        if any(column not in wide.columns for column in needed):
            print(
                f"Refinement for {model} has no {HISTORY_CONDITION} scores; "
                "rerun judge.py refinement to compute Err_RA"
            )
            continue
        paired = wide.dropna(subset=needed)
        if paired.empty:
            continue
        mean_refined = float(paired["refined"].mean())
        mean_history = float(paired[HISTORY_CONDITION].mean())
        if mean_refined == 0:
            error_rate = float("nan")
        else:
            error_rate = mean_history / mean_refined - 1
        usable = paired[paired["refined"] != 0]
        if usable.empty:
            item_ratio = float("nan")
        else:
            item_ratio = float((usable[HISTORY_CONDITION] / usable["refined"]).mean())
        if "original" in paired.columns:
            mean_original = float(paired["original"].mean())
        else:
            mean_original = float("nan")
        rows.append(
            {
                "model": model,
                "n": len(paired),
                "mean_original": mean_original,
                "mean_refined": mean_refined,
                "mean_with_history": mean_history,
                "error_rate_ra": error_rate,
                "mean_item_ratio": item_ratio,
            }
        )
    if not rows:
        return pd.DataFrame(columns=REFINEMENT_COLUMNS)
    return pd.DataFrame(rows)[REFINEMENT_COLUMNS]


def _answer_key(text):
    if _missing(text):
        return None
    collapsed = " ".join(str(text).split())
    return collapsed or None


def _other_judge_scores(answer_key, model, scores_by_answer):
    """Mean of the per-judge mean scores for this answer, excluding model."""
    per_judge = {}
    for judge, rating in scores_by_answer.get(answer_key, ()):
        if judge == model:
            continue
        per_judge.setdefault(judge, []).append(rating)
    if not per_judge:
        return None, set()
    means = [sum(ratings) / len(ratings) for ratings in per_judge.values()]
    return sum(means) / len(means), set(per_judge)


def self_enhancement_metrics(frame):
    """ErrorRate_SE = mean(y_self) / mean(y_other) - 1.

    y_self is this model's score of its own generated answer. y_other is the
    mean score the other models assigned to that same answer. Items that no
    other model scored are left out of the ratio.
    """
    if frame.empty:
        return pd.DataFrame(columns=SELF_COLUMNS)
    scored = frame[
        (frame["bias"] == "self-enhancement")
        & (frame["kind"] == "score")
        & frame["parsed"]
    ]
    own = scored[scored["condition"] == "self"]
    if own.empty:
        return pd.DataFrame(columns=SELF_COLUMNS)

    scores_by_answer = {}
    for row in scored.itertuples(index=False):
        key = _answer_key(row.generated_answer)
        if key is None or _missing(row.rating):
            continue
        scores_by_answer.setdefault(key, []).append((row.model, float(row.rating)))

    rows = []
    for model, group in own.groupby("model", sort=True):
        ratings = [float(rating) for rating in group["rating"] if not _missing(rating)]
        paired_self = []
        paired_other = []
        judges = set()
        for row in group.itertuples(index=False):
            key = _answer_key(row.generated_answer)
            if key is None or _missing(row.rating):
                continue
            other, used = _other_judge_scores(key, model, scores_by_answer)
            if other is None:
                continue
            paired_self.append(float(row.rating))
            paired_other.append(other)
            judges.update(used)
        n = len(paired_self)
        if n == 0:
            mean_self = float(sum(ratings) / len(ratings)) if ratings else float("nan")
            mean_other = float("nan")
            error_rate = float("nan")
            item_ratio = float("nan")
        else:
            mean_self = sum(paired_self) / n
            mean_other = sum(paired_other) / n
            error_rate = (
                float("nan") if mean_other == 0 else mean_self / mean_other - 1
            )
            ratios = [
                left / right
                for left, right in zip(paired_self, paired_other)
                if right != 0
            ]
            item_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        rows.append(
            {
                "model": model,
                "n_self": len(ratings),
                "n": n,
                "n_other_judges": len(judges),
                "mean_self": mean_self,
                "mean_other": mean_other,
                "error_rate_se": error_rate,
                "mean_item_ratio": item_ratio,
            }
        )
    if not rows:
        return pd.DataFrame(columns=SELF_COLUMNS)
    return pd.DataFrame(rows)[SELF_COLUMNS]


def position_breakdown(frame):
    """Classify each item by the slots chosen in the A-B and B-A orders.

    consistent: the same answer wins in both orders. always_a / always_b: the
    same letter wins in both orders, so the answer changed with its slot.
    other: a tie or a verdict that failed to parse.
    """
    if frame.empty:
        return pd.DataFrame(columns=POSITION_COLUMNS)
    rows = []
    position = frame[(frame["bias"] == "position") & (frame["kind"] == "pairwise")]
    for model, group in position.groupby("model", sort=True):
        forward = _by_index(group, "order_ab")
        swapped = _by_index(group, "order_ba")
        shared = forward.index.intersection(swapped.index)
        counts = {"consistent": 0, "always_a": 0, "always_b": 0, "other": 0}
        for index in shared:
            first = forward.at[index, "verdict"]
            second = swapped.at[index, "verdict"]
            if first not in {"A", "B"} or second not in {"A", "B"}:
                counts["other"] += 1
            elif first != second:
                counts["consistent"] += 1
            elif first == "A":
                counts["always_a"] += 1
            else:
                counts["always_b"] += 1
        total = len(shared)
        rows.append(
            {
                "model": model,
                "n": total,
                **{key: value / total if total else float("nan") for key, value in counts.items()},
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def write_table(frame, path):
    frame.to_csv(path, index=False, float_format="%.6f")


# ---------------------------------------------------------------------------
# Figures (French labels, used by the presentation)
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "deepseek-r1:7b": "DeepSeek-R1 7B",
    "gemma4:e4b": "Gemma4 e4B",
    "qwen3:4b": "Qwen3 4B",
}
MODEL_ORDER = list(MODEL_LABELS.values())
PALETTE = dict(zip(MODEL_ORDER, ("#4c72b0", "#dd8452", "#55a868")))
POINT_OFFSETS = dict(zip(MODEL_ORDER, (-0.18, 0.0, 0.18)))
PAPER_BOX_COLOR = "#d9dee8"
PAPER_LABEL = "Article (6 juges)"

BIAS_LABELS = {
    "position": "Position",
    "verbosity": "Verbosité",
    "compassion": "Compassion-fade",
    "bandwagon": "Bandwagon",
    "distraction": "Distraction",
    "authority": "Autorité",
    "sentiment": "Sentiment",
    "diversity": "Diversité",
    "cot": "CoT",
}

# Ye et al. (ICLR 2025). Every list follows the paper's judge order:
# ChatGPT, GPT-4-Turbo, GPT-4o, GLM-4, Claude-3.5, Qwen2.
PAPER_RR = {  # Table 4
    "verbosity": [0.900, 0.915, 0.977, 0.887, 0.952, 0.884],
    "sentiment": [0.804, 0.653, 0.699, 0.679, 0.660, 0.651],
    "position": [0.566, 0.818, 0.776, 0.781, 0.832, 0.760],
    "compassion": [0.862, 0.858, 0.868, 0.835, 0.875, 0.877],
    "bandwagon": [0.688, 0.638, 0.791, 0.690, 0.610, 0.710],
    "authority": [0.662, 0.846, 0.787, 0.796, 0.865, 0.779],
    "distraction": [0.713, 0.729, 0.790, 0.814, 0.878, 0.785],
    "diversity": [0.679, 0.855, 0.814, 0.788, 0.914, 0.826],
}
PAPER_BIS = {  # Table 7
    "verbosity": [0.098, 0.075, 0.021, 0.083, 0.047, 0.110],
    "sentiment": [0.194, 0.337, 0.299, 0.291, 0.339, 0.343],
    "position": [0.340, 0.038, 0.149, 0.103, 0.083, 0.144],
    "compassion": [0.044, -0.002, 0.057, 0.049, 0.040, 0.027],
    "bandwagon": [0.218, 0.218, 0.134, 0.194, 0.305, 0.194],
    "authority": [0.244, 0.010, 0.138, 0.088, 0.050, 0.125],
    "distraction": [0.193, 0.127, 0.135, 0.070, 0.037, 0.119],
    "diversity": [0.227, 0.001, 0.111, 0.096, 0.001, 0.078],
}
PAPER_CR = {  # Table 4: CR_FR and CR_AL
    "fact": [0.998, 0.990, 0.998, 0.970, 0.999, 0.994],
    "alignment": [0.906, 0.856, 0.925, 0.884, 0.915, 0.904],
}
PAPER_CONDITION_RR = {  # Table 8
    ("bandwagon", "bandwagon_60"): [0.680, 0.635, 0.773, 0.703, 0.563, 0.711],
    ("bandwagon", "bandwagon_70"): [0.667, 0.630, 0.787, 0.676, 0.613, 0.711],
    ("bandwagon", "bandwagon_80"): [0.707, 0.662, 0.805, 0.664, 0.638, 0.698],
    ("bandwagon", "bandwagon_90"): [0.699, 0.623, 0.800, 0.716, 0.627, 0.718],
    ("authority", "book"): [0.628, 0.841, 0.800, 0.765, 0.856, 0.785],
    ("authority", "quote"): [0.660, 0.841, 0.747, 0.758, 0.856, 0.745],
    ("authority", "url"): [0.700, 0.855, 0.813, 0.866, 0.884, 0.805],
    ("sentiment", "cheerful_on_better"): [0.803, 0.682, 0.727, 0.770, 0.609, 0.726],
    ("sentiment", "cheerful_on_worse"): [0.910, 0.888, 0.970, 0.905, 0.976, 0.871],
    ("sentiment", "sad_on_better"): [0.659, 0.271, 0.343, 0.306, 0.259, 0.307],
    ("sentiment", "sad_on_worse"): [0.916, 0.920, 0.983, 0.907, 0.970, 0.929],
    ("sentiment", "angry_on_better"): [0.639, 0.366, 0.243, 0.380, 0.256, 0.283],
    ("sentiment", "angry_on_worse"): [0.946, 0.921, 0.987, 0.950, 0.973, 0.926],
    ("sentiment", "fear_on_better"): [0.639, 0.254, 0.355, 0.271, 0.260, 0.238],
    ("sentiment", "fear_on_worse"): [0.923, 0.921, 0.987, 0.943, 0.973, 0.926],
}
PAPER_AIR = [3.8, 0.8, 5.2, 7.0, 3.5, 5.1]  # Table 4, in percent
PAPER_ERR_RA = [-5.48, 1.69, -3.22, 1.19, 2.22, 1.35]  # Table 5, in percent
PAPER_ERR_SE = [-16.64, 9.40, 8.18, 3.63, 5.39, 8.99]  # Table 5, in percent

FACT_BIASES = ("verbosity", "sentiment")
ALIGNMENT_BIASES = ("position", "compassion", "bandwagon", "authority", "distraction", "diversity")

# Wall-clock hours per model on one Jetson Orin Nano.
RUN_HOURS = {
    "Self-enhancement": 10,
    "Diversité": 8,
    "Bandwagon": 6,
    "Compassion-fade": 4,
    "Distraction": 4,
    "Sentiment": 3.5,
    "Verbosité": 3,
    "Position": 2.5,
    "CoT": 2.5,
    "Autorité": 1.75,
    "Raffinement": 1,
}

# Sized for the presentation's visuals column (about 4:3), so that 15 pt
# text stays readable once the image is scaled into the slide.
COLUMN_FIGURE = (8.5, 7)
WIDE_FIGURE = (9.5, 7.5)

FIGURE_STYLE = {
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
}


def _labelled(frame):
    frame = frame.copy()
    frame["label"] = frame["model"].map(MODEL_LABELS)
    return frame[frame["label"].notna()]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _paper_and_points(ax, categories, paper, ours, width=0.55):
    """Box plot of the paper's six judges per category, our models as points.

    paper: list of value lists, one per category. ours: {model label: values}.
    """
    positions = list(range(len(categories)))
    ax.boxplot(
        paper,
        positions=positions,
        widths=width,
        patch_artist=True,
        zorder=1,
        flierprops={"marker": "D", "markersize": 7, "markerfacecolor": "#8a94a8", "markeredgecolor": "#8a94a8"},
        boxprops={"facecolor": PAPER_BOX_COLOR, "edgecolor": "#8a94a8", "linewidth": 1.4},
        medianprops={"color": "#5b6577", "linewidth": 2},
        whiskerprops={"color": "#8a94a8", "linewidth": 1.4},
        capprops={"color": "#8a94a8", "linewidth": 1.4},
    )
    for label in MODEL_ORDER:
        values = ours.get(label)
        if values is None:
            continue
        ax.scatter(
            [p + POINT_OFFSETS[label] for p in positions],
            values,
            s=150,
            color=PALETTE[label],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label=label,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(categories)
    ax.set_xlim(-0.6, len(categories) - 0.4)


def _paper_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=PAPER_BOX_COLOR, edgecolor="#8a94a8"))
    labels.append(PAPER_LABEL)
    handles.append(
        plt.Line2D([], [], marker="D", linestyle="", markersize=7, color="#8a94a8")
    )
    labels.append("Juge de l'article hors boîte")
    return handles, labels


def _legend_below(ax, ncol=3, offset=-0.12):
    handles, labels = _paper_handles(ax)
    ax.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, offset), ncol=ncol, frameon=False
    )


def _figure_legend(fig, ax, ncol=3, offset=-0.07):
    handles, labels = _paper_handles(ax)
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, offset + 0.07))


def _summary_values(summary, biases, column="rr"):
    ours = {}
    for label, group in _labelled(summary).groupby("label"):
        values = group.set_index("bias")[column]
        ours[label] = [values.get(bias, float("nan")) for bias in biases]
    return ours


def _tilt_labels(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")


def plot_rr_vs_paper(summary, out_dir):
    biases = list(PAPER_RR)
    fig, ax = plt.subplots(figsize=WIDE_FIGURE)
    _paper_and_points(
        ax,
        [BIAS_LABELS[bias] for bias in biases],
        [PAPER_RR[bias] for bias in biases],
        _summary_values(summary, biases),
    )
    _tilt_labels(ax)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Taux de robustesse (RR)")
    ax.set_title("RR : article vs. nos modèles")
    _legend_below(ax, offset=-0.22)
    _save(fig, out_dir / "rr_article_vs_nous.png")


def plot_bis(summary, out_dir):
    biases = list(PAPER_BIS)
    table = summary.copy()
    table["bis"] = table["cr"] - table["rr"]
    fig, ax = plt.subplots(figsize=WIDE_FIGURE)
    _paper_and_points(
        ax,
        [BIAS_LABELS[bias] for bias in biases],
        [PAPER_BIS[bias] for bias in biases],
        _summary_values(table, biases, "bis"),
    )
    _tilt_labels(ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("BIS = CR − RR")
    ax.set_title("Effet du biais au-delà du bruit (BIS)")
    _legend_below(ax, offset=-0.22)
    _save(fig, out_dir / "bis.png")


def plot_cr_vs_paper(consistency, out_dir):
    groups = (
        ("Questions factuelles\n(verbosité, sentiment)", FACT_BIASES, PAPER_CR["fact"]),
        ("Préférences humaines\n(position, bandwagon, autorité…)", ALIGNMENT_BIASES, PAPER_CR["alignment"]),
    )
    ours = {}
    for label, group in _labelled(consistency.dropna(subset=["cr"])).groupby("label"):
        values = group.set_index("bias")["cr"]
        ours[label] = [
            float(values.reindex(list(biases)).dropna().mean()) for _, biases, _ in groups
        ]
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    _paper_and_points(ax, [name for name, _, _ in groups], [paper for _, _, paper in groups], ours)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Taux de cohérence (CR)")
    ax.set_title("Le juge redonne-t-il le même verdict ?")
    _legend_below(ax, ncol=2)
    _save(fig, out_dir / "cr_article_vs_nous.png")


def _condition_values(robust, bias, conditions):
    ours = {}
    rows = _labelled(robust[robust["bias"] == bias])
    for label, group in rows.groupby("label"):
        values = group.set_index("condition")["rr"]
        ours[label] = [values.get(condition, float("nan")) for condition in conditions]
    return ours


def plot_sentiment(robust, out_dir):
    tones = (("cheerful", "Joyeux"), ("sad", "Triste"), ("angry", "En colère"), ("fear", "Peur"))
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharey=True)
    for ax, (suffix, title) in zip(
        axes,
        (
            ("on_better", "Ton ajouté à la réponse de référence"),
            ("on_worse", "Ton ajouté à la réponse dégradée"),
        ),
    ):
        conditions = [f"{tone}_{suffix}" for tone, _ in tones]
        _paper_and_points(
            ax,
            [name for _, name in tones],
            [PAPER_CONDITION_RR[("sentiment", condition)] for condition in conditions],
            _condition_values(robust, "sentiment", conditions),
        )
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("RR")
    _figure_legend(fig, axes[0], ncol=3)
    _save(fig, out_dir / "sentiment.png")


def plot_bandwagon(robust, out_dir):
    conditions = [f"bandwagon_{percent}" for percent in (60, 70, 80, 90)]
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    _paper_and_points(
        ax,
        ["60 %", "70 %", "80 %", "90 %"],
        [PAPER_CONDITION_RR[("bandwagon", condition)] for condition in conditions],
        _condition_values(robust, "bandwagon", conditions),
    )
    ax.set_ylim(0.5, 0.9)
    ax.set_xlabel("Majorité annoncée pour la fausse réponse")
    ax.set_ylabel("Taux de robustesse (RR)")
    ax.set_title("Bandwagon : effet du pourcentage annoncé")
    _legend_below(ax)
    _save(fig, out_dir / "bandwagon.png")


def plot_authority_distraction(robust, out_dir):
    """Authority against the paper (top) and distraction on A vs. B (bottom)."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 9.5))
    conditions = ("book", "quote", "url")
    _paper_and_points(
        top,
        ["Livre", "Citation célèbre", "URL"],
        [PAPER_CONDITION_RR[("authority", condition)] for condition in conditions],
        _condition_values(robust, "authority", conditions),
    )
    top.set_ylim(0.5, 1.0)
    top.set_ylabel("RR")
    top.set_title("Autorité : fausse source sur la fausse réponse")
    handles, labels = _paper_handles(top)
    top.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    rows = _labelled(robust[robust["bias"] == "distraction"]).copy()
    rows["label"] = rows["label"].str.replace(" ", "\n", n=1)
    rows["cible"] = rows["condition"].map(
        {"distract_a": "Sur la réponse correcte (A)", "distract_b": "Sur la fausse réponse (B)"}
    )
    sns.barplot(
        data=rows,
        x="label",
        y="rr",
        hue="cible",
        order=[label.replace(" ", "\n", 1) for label in MODEL_ORDER],
        palette=["#4c72b0", "#c44e52"],
        ax=bottom,
    )
    for container in bottom.containers:
        bottom.bar_label(container, fmt="%.2f", fontsize=15)
    bottom.set_ylim(0, 1.05)
    bottom.set_xlabel("")
    bottom.set_ylabel("RR")
    bottom.set_title("Distraction : phrase hors sujet ajoutée")
    bottom.legend(title="", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    _save(fig, out_dir / "autorite_distraction.png")


def plot_position(summary, breakdown, out_dir):
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.5, 8), gridspec_kw={"width_ratios": [1, 1.7]}
    )
    rr = _summary_values(summary, ["position"])
    cr = _summary_values(summary, ["position"], "cr")
    _paper_and_points(left, ["Position"], [PAPER_RR["position"]], rr, width=0.45)
    for label in MODEL_ORDER:
        if label in cr:
            left.scatter(
                [POINT_OFFSETS[label]],
                cr[label],
                s=300,
                marker="_",
                linewidths=4,
                color=PALETTE[label],
                zorder=3,
            )
    left.scatter([], [], marker="_", s=300, linewidths=4, color="black", label="CR (même ordre répété)")
    left.set_ylim(0.3, 1.0)
    left.set_ylabel("Taux de robustesse (RR)")
    left.set_title("RR")
    handles, labels = _paper_handles(left)
    left.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=False)

    table = _labelled(breakdown).set_index("label").reindex(MODEL_ORDER).dropna(subset=["n"])
    table.index = [label.replace(" ", "\n", 1) for label in table.index]
    parts = (
        ("consistent", "Même réponse dans les deux ordres", "#55a868"),
        ("always_a", "Toujours la position A", "#4c72b0"),
        ("always_b", "Toujours la position B", "#dd8452"),
        ("other", "Égalité ou non analysé", "#b0b0b0"),
    )
    bottom = [0.0] * len(table)
    for column, name, color in parts:
        values = list(table[column] * 100)
        bars = right.bar(table.index, values, bottom=bottom, color=color, label=name, width=0.6)
        for bar, value, base in zip(bars, values, bottom):
            if value >= 6:
                right.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.0f} %",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="white",
                    fontweight="bold",
                )
        bottom = [b + v for b, v in zip(bottom, values)]
    right.set_ylim(0, 100)
    right.set_ylabel("Part des paires (%)")
    right.set_title("Après inversion de A et B")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)
    _save(fig, out_dir / "position.png")


def plot_other_biases(summary, out_dir):
    biases = ("verbosity", "compassion", "diversity")
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    _paper_and_points(
        ax,
        [BIAS_LABELS[bias] for bias in biases],
        [PAPER_RR[bias] for bias in biases],
        _summary_values(summary, biases),
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Taux de robustesse (RR)")
    ax.set_title("Verbosité, compassion-fade, diversité")
    _legend_below(ax, ncol=2)
    _save(fig, out_dir / "autres_biais.png")


def plot_cot(cot, out_dir):
    if cot.empty:
        return
    ours = {label: [value] for label, value in _labelled(cot).set_index("label")["air"].items()}
    fig, ax = plt.subplots(figsize=(7, 7))
    _paper_and_points(ax, ["Chain-of-thought"], [PAPER_AIR], ours, width=0.45)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Gain de précision, AIR (%)")
    ax.set_title("CoT : gain de précision")
    handles, labels = _paper_handles(ax)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    _save(fig, out_dir / "cot.png")


def plot_scoring(refinement, self_enhancement, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=COLUMN_FIGURE)
    panels = (
        (axes[0], refinement, "error_rate_ra", PAPER_ERR_RA, "Raffinement (Err_RA)"),
        (axes[1], self_enhancement, "error_rate_se", PAPER_ERR_SE, "Self-enhancement (Err_SE)"),
    )
    for ax, table, column, paper, title in panels:
        ours = {}
        if not table.empty:
            ours = {
                label: [value * 100]
                for label, value in _labelled(table).set_index("label")[column].items()
            }
        _paper_and_points(ax, [title.split(" (")[0]], [paper], ours, width=0.45)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title.split(" (")[1].rstrip(")"))
    axes[0].set_ylabel("Taux d'erreur (%)")
    _figure_legend(fig, axes[0], ncol=2)
    _save(fig, out_dir / "notation.png")


def plot_run_hours(out_dir):
    names = list(RUN_HOURS)[::-1]
    hours = [RUN_HOURS[name] for name in names]
    fig, ax = plt.subplots(figsize=COLUMN_FIGURE)
    bars = ax.barh(names, hours, color="#1f3b6e")
    for bar, value in zip(bars, hours):
        whole = int(value)
        minutes = round((value - whole) * 60)
        text = f"{whole} h {minutes:02d}" if minutes else f"{whole} h"
        ax.text(value + 0.15, bar.get_y() + bar.get_height() / 2, text, va="center", fontsize=15)
    ax.set_xlim(0, max(hours) * 1.18)
    ax.set_xlabel("Heures par modèle")
    ax.set_title(f"Temps par biais (≈ {sum(hours):.0f} h par modèle)")
    ax.grid(axis="y", visible=False)
    _save(fig, out_dir / "durees.png")


def plot_figures(summary, robust, consistency, cot, refinement, self_enhancement, breakdown, out_dir):
    sns.set_theme(style="whitegrid", context="notebook", rc=FIGURE_STYLE)
    plot_rr_vs_paper(summary, out_dir)
    plot_bis(summary, out_dir)
    plot_cr_vs_paper(consistency, out_dir)
    plot_sentiment(robust, out_dir)
    plot_bandwagon(robust, out_dir)
    plot_authority_distraction(robust, out_dir)
    plot_position(summary, breakdown, out_dir)
    plot_other_biases(summary, out_dir)
    plot_cot(cot, out_dir)
    plot_scoring(refinement, self_enhancement, out_dir)
    plot_run_hours(out_dir)


def print_rr_summary(robust, consistency, summary):
    if robust.empty:
        print("No pairwise results to summarize.")
        return
    float_format = "{:.3f}".format
    print(
        "Per bias (rr and headline are weighted by n across conditions; "
        "headline is acc for fallacy, whose rr is not meaningful, and rr otherwise):"
    )
    print(summary.to_string(index=False, float_format=float_format))
    print()
    print("Consistency rate with 95% Wilson interval:")
    print(consistency.to_string(index=False, float_format=float_format))
    print()
    print("By condition:")
    print(robust.to_string(index=False, float_format=float_format))


def run(results_dir, out_dir, models=None, biases=None):
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    frame = load_records(results_dir, models=models, biases=biases)
    out_dir.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print(f"No judgment files matched in {results_dir}")
        return

    summary = parse_summary(frame)
    robust, consistency = pairwise_metrics(frame)
    per_bias = bias_summary(robust, consistency)
    cot = cot_metrics(frame)
    refinement = refinement_metrics(frame)
    self_enhancement = self_enhancement_metrics(frame)
    breakdown = position_breakdown(frame)

    write_table(summary, out_dir / "parse_summary.csv")
    write_table(robust, out_dir / "robustness.csv")
    write_table(consistency, out_dir / "consistency.csv")
    write_table(per_bias, out_dir / "bias_summary.csv")
    write_table(cot, out_dir / "cot.csv")
    write_table(refinement, out_dir / "refinement.csv")
    write_table(self_enhancement, out_dir / "self_enhancement.csv")
    write_table(breakdown, out_dir / "position.csv")

    plot_figures(
        per_bias, robust, consistency, cot, refinement, self_enhancement, breakdown, out_dir
    )

    print_rr_summary(robust, consistency, per_bias)
    if cot.empty:
        print("\nNo chain-of-thought results; cot.csv is empty.")
    else:
        print("\nChain-of-thought:")
        print(cot.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if refinement.empty:
        print("\nNo refinement results; refinement.csv is empty.")
    else:
        print("\nRefinement:")
        print(
            refinement.to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
    if self_enhancement.empty:
        print("\nNo self-enhancement results; self_enhancement.csv is empty.")
    else:
        print("\nSelf-enhancement:")
        print(
            self_enhancement.to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            )
        )
        if self_enhancement["error_rate_se"].isna().all():
            print(
                "ErrorRate_SE is undefined: no other model scored the same "
                "generated answer."
            )
    print(f"\nWrote tables and charts to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute robustness, consistency, accuracy, CoT AIR, refinement "
            "error rate, and self-enhancement error rate."
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results",
        help="Directory of judge JSON files (default: results)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "analysis",
        help="Directory for CSV tables and PNG charts (default: analysis)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only these model names, as stored in the result files",
    )
    parser.add_argument(
        "--biases",
        nargs="*",
        default=None,
        help="Only these bias names",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.results, arguments.out, arguments.models, arguments.biases)
