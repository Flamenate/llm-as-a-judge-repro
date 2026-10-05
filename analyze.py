"""Summarize saved LLM-judge results.

Usage:
    python analyze.py
    python analyze.py --results results --out analysis
    python analyze.py --models qwen3:4b --biases fallacy position

Reads results/<model>__<bias>.json files written by judge.py. Writes CSV
tables and PNG charts under analysis/. Consistency Rate and self-enhancement
are not computed.
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

ROBUST_COLUMNS = [
    "model",
    "bias",
    "condition",
    "n",
    "rr",
    "acc_baseline",
    "acc_condition",
]
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


def pairwise_metrics(frame):
    if frame.empty:
        return pd.DataFrame(columns=ROBUST_COLUMNS)
    pairwise = frame[(frame["kind"] == "pairwise") & ~frame["bias"].isin(SKIP_PAIRWISE)]
    rows = []
    grouped = pairwise.groupby(["model", "bias"], sort=True)
    for (model, bias), group in grouped:
        ref_name = reference_condition(bias)
        reference = group[group["condition"] == ref_name]
        if reference.empty:
            print(f"No {ref_name} judgments for {model} / {bias}; skipping RR")
            continue
        reference = reference.drop_duplicates("index", keep="last").set_index("index")
        conditions = sorted(
            condition
            for condition in group["condition"].unique()
            if condition != ref_name
        )
        for condition in conditions:
            perturbed = group[group["condition"] == condition]
            perturbed = perturbed.drop_duplicates("index", keep="last").set_index(
                "index"
            )
            shared = reference.index.intersection(perturbed.index)
            paired = [
                index
                for index in shared
                if reference.at[index, "parsed"] and perturbed.at[index, "parsed"]
            ]
            if not paired:
                rows.append(
                    {
                        "model": model,
                        "bias": bias,
                        "condition": condition,
                        "n": 0,
                        "rr": float("nan"),
                        "acc_baseline": float("nan"),
                        "acc_condition": float("nan"),
                    }
                )
                continue
            base = reference.loc[paired]
            other = perturbed.loc[paired]
            same_answer = [
                map_choice(bias, ref_name, left) == map_choice(bias, condition, right)
                for left, right in zip(base["verdict"], other["verdict"])
            ]
            rows.append(
                {
                    "model": model,
                    "bias": bias,
                    "condition": condition,
                    "n": len(paired),
                    "rr": sum(same_answer) / len(paired),
                    "acc_baseline": mean_match(
                        base["verdict"], base["unbiased_verdict"]
                    ),
                    "acc_condition": mean_match(
                        other["verdict"], other["unbiased_verdict"]
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(columns=ROBUST_COLUMNS)
    return pd.DataFrame(rows)[ROBUST_COLUMNS]


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
        needed = ["refined", "refined_with_history"]
        if any(column not in wide.columns for column in needed):
            print(f"Refinement for {model} is missing refined scores")
            continue
        paired = wide.dropna(subset=needed)
        if paired.empty:
            continue
        mean_refined = float(paired["refined"].mean())
        mean_history = float(paired["refined_with_history"].mean())
        if mean_refined == 0:
            error_rate = float("nan")
        else:
            error_rate = mean_history / mean_refined - 1
        usable = paired[paired["refined"] != 0]
        if usable.empty:
            item_ratio = float("nan")
        else:
            item_ratio = float(
                (usable["refined_with_history"] / usable["refined"]).mean()
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


def plot_rr_heatmap(robust, out_dir):
    table = robust.dropna(subset=["rr"]).copy()
    table["column"] = table["bias"] + "/" + table["condition"]
    _heatmap(
        table,
        "rr",
        out_dir / "rr_heatmap.png",
        "Robustness rate",
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


def print_rr_summary(robust):
    if robust.empty:
        print("No pairwise results to summarize.")
        return
    print("Robustness rate by model and bias:")
    rolled = robust.groupby(["model", "bias"], as_index=False).agg(
        rr=("rr", "mean"),
        conditions=("condition", "nunique"),
        n=("n", "sum"),
    )
    print(rolled.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()
    print("Robustness rate by condition:")
    print(robust.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def run(results_dir, out_dir, models=None, biases=None):
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory not found: {results_dir}")
    frame = load_records(results_dir, models=models, biases=biases)
    out_dir.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        print(f"No judgment files matched in {results_dir}")
        return

    summary = parse_summary(frame)
    robust = pairwise_metrics(frame)
    cot = cot_metrics(frame)
    refinement = refinement_metrics(frame)

    write_table(summary, out_dir / "parse_summary.csv")
    write_table(robust, out_dir / "robustness.csv")
    write_table(cot, out_dir / "cot.csv")
    write_table(refinement, out_dir / "refinement.csv")

    sns.set_theme(style="whitegrid", context="notebook")
    plot_rr_heatmap(robust, out_dir)
    plot_accuracy_heatmap(robust, out_dir)
    plot_rr_bars(robust, out_dir)
    plot_cot(cot, out_dir)
    plot_refinement(refinement, out_dir)

    print_rr_summary(robust)
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
        description="Compute robustness, accuracy, CoT AIR, and refinement error rate.",
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
