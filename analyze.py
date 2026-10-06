"""Summarize saved LLM-judge results.

Usage:
    python analyze.py
    python analyze.py --results results --out analysis
    python analyze.py --models qwen3:4b --biases fallacy position

Reads results/<model>__<bias>.json files written by judge.py. Writes CSV
tables and PNG charts under analysis/. Self-enhancement is not computed.

For each pairwise bias, D is the set of items whose unperturbed reference
judgment y parsed. Robustness Rate is I(y == y_hat) over D, and Consistency
Rate is I(y == y_rand) over the items in D that have a <reference>_rand
repeat. A perturbed or repeated judgment that failed to parse counts as a
disagreement. RR is the headline metric for every pairwise bias except
fallacy: its rewrite flips which answer is better, so its headline is accuracy
against the expected answer.

Refinement uses the refined_given_history condition;
files from before that prompt existed have no Err_RA.
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
IGNORED_BIASES = {"self-enhancement"}

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
PARSE_COLUMNS = ["model", "bias", "condition", "total", "parsed", "failed"]

VERDICTS = {"A", "B", "C"}


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
    if verdict in VERDICTS:
        return True
    return isinstance(rating, (int, float)) and not isinstance(rating, bool)


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
        if bias in IGNORED_BIASES:
            print(f"Skipping {path.name}: self-enhancement is not analyzed")
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
        for left, right, parsed in zip(base["verdict"], other["verdict"], other["parsed"])
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
            item_ratio = float(
                (usable[HISTORY_CONDITION] / usable["refined"]).mean()
            )
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


def write_table(frame, path):
    frame.to_csv(path, index=False, float_format="%.6f")


def _column_order(labels):
    def sort_key(label):
        bias, _, condition = label.partition("/")
        reference_first = 0 if condition == reference_condition(bias) else 1
        return (bias, reference_first, condition)

    return sorted(set(labels), key=sort_key)


def _heatmap(table, value, path, title, vmin=0, vmax=1):
    if table.empty:
        return
    pivot = table.pivot(index="model", columns="column", values=value)
    pivot = pivot.reindex(columns=_column_order(pivot.columns))
    labels = pivot.map(lambda cell: "" if pd.isna(cell) else f"{cell:.2f}")
    width = max(8, 0.85 * len(pivot.columns) + 3)
    height = max(3.5, 0.7 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        pivot,
        annot=labels,
        fmt="",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _interpretable_rr(robust):
    """Fallacy's expected answer flips under the perturbation, so its RR is noise."""
    return robust[robust["bias"] != "fallacy"]


def plot_rr_heatmap(robust, out_dir):
    table = _interpretable_rr(robust).dropna(subset=["rr"]).copy()
    table["column"] = table["bias"] + "/" + table["condition"]
    _heatmap(
        table,
        "rr",
        out_dir / "rr_heatmap.png",
        "Robustness rate",
    )


def plot_cr_heatmap(consistency, out_dir):
    table = consistency.dropna(subset=["cr"]).copy()
    table["column"] = table["bias"]
    _heatmap(
        table,
        "cr",
        out_dir / "cr_heatmap.png",
        "Consistency rate",
    )


def plot_accuracy_heatmap(robust, out_dir):
    if robust.empty:
        return
    cells = []
    for (model, bias), group in robust.groupby(["model", "bias"], sort=True):
        best = group.sort_values(["n", "condition"], ascending=[False, True]).iloc[0]
        cells.append(
            {
                "model": model,
                "column": f"{bias}/{reference_condition(bias)}",
                "accuracy": best["acc_baseline"],
            }
        )
        for _, row in group.iterrows():
            cells.append(
                {
                    "model": model,
                    "column": f"{bias}/{row['condition']}",
                    "accuracy": row["acc_condition"],
                }
            )
    table = pd.DataFrame(cells).dropna(subset=["accuracy"])
    _heatmap(
        table,
        "accuracy",
        out_dir / "accuracy_heatmap.png",
        "Accuracy against the expected answer",
    )


def plot_rr_bars(robust, out_dir):
    robust = _interpretable_rr(robust)
    if robust.empty:
        return
    for bias, group in robust.groupby("bias", sort=True):
        fig, ax = plt.subplots(
            figsize=(max(6, 1.4 * group["condition"].nunique() + 2), 4.5)
        )
        sns.barplot(data=group, x="condition", y="rr", hue="model", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Robustness rate")
        ax.set_xlabel("")
        ax.set_title(f"Robustness rate: {bias}")
        fig.tight_layout()
        fig.savefig(out_dir / f"rr_{bias}.png", dpi=150)
        plt.close(fig)


def plot_cot(cot, out_dir):
    if cot.empty or cot["air"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(cot) + 2), 4.5))
    sns.barplot(data=cot, x="model", y="air", color="#3b6ea5", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Accuracy improvement rate (%)")
    ax.set_xlabel("")
    ax.set_title("Chain-of-thought accuracy improvement")
    fig.tight_layout()
    fig.savefig(out_dir / "cot_air.png", dpi=150)
    plt.close(fig)


def plot_refinement(refinement, out_dir):
    if refinement.empty:
        return
    long = refinement.melt(
        id_vars=["model"],
        value_vars=["mean_original", "mean_refined", "mean_with_history"],
        var_name="version",
        value_name="score",
    )
    long["version"] = long["version"].map(
        {
            "mean_original": "Original",
            "mean_refined": "Refined",
            "mean_with_history": "With history",
        }
    )
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(refinement) + 2), 4.5))
    sns.barplot(data=long, x="model", y="score", hue="version", ax=ax)
    ax.set_ylabel("Mean score")
    ax.set_xlabel("")
    ax.set_title("Refinement scores")
    fig.tight_layout()
    fig.savefig(out_dir / "refinement_scores.png", dpi=150)
    plt.close(fig)


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

    write_table(summary, out_dir / "parse_summary.csv")
    write_table(robust, out_dir / "robustness.csv")
    write_table(consistency, out_dir / "consistency.csv")
    write_table(per_bias, out_dir / "bias_summary.csv")
    write_table(cot, out_dir / "cot.csv")
    write_table(refinement, out_dir / "refinement.csv")

    sns.set_theme(style="whitegrid", context="notebook")
    plot_rr_heatmap(robust, out_dir)
    plot_cr_heatmap(consistency, out_dir)
    plot_accuracy_heatmap(robust, out_dir)
    plot_rr_bars(robust, out_dir)
    plot_cot(cot, out_dir)
    plot_refinement(refinement, out_dir)

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
    print(f"\nWrote tables and charts to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute robustness, consistency, accuracy, CoT AIR, and refinement "
            "error rate."
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
