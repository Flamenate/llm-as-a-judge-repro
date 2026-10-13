"""French figures for the presentation.

Usage:
    python analysis/figures_fr.py

Reads the CSV tables written by analyze.py and writes PNG charts to
analysis/fr/. Paper values are copied from Ye et al. (ICLR 2025), Tables 4, 5
and 7.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ANALYSIS = Path(__file__).resolve().parent
OUT = ANALYSIS / "fr"

MODELS = {
    "deepseek-r1:7b": "DeepSeek-R1 7B",
    "gemma4:e4b": "Gemma4 e4B",
    "qwen3:4b": "Qwen3 4B",
}
MODEL_ORDER = list(MODELS.values())
PALETTE = dict(zip(MODEL_ORDER, sns.color_palette("deep", 3)))

BIASES = {
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

# Table 4: ChatGPT, GPT-4-Turbo, GPT-4o, GLM-4, Claude-3.5, Qwen2.
PAPER_RR = {
    "verbosity": [0.900, 0.915, 0.977, 0.887, 0.952, 0.884],
    "sentiment": [0.804, 0.653, 0.699, 0.679, 0.660, 0.651],
    "position": [0.566, 0.818, 0.776, 0.781, 0.832, 0.760],
    "compassion": [0.862, 0.858, 0.868, 0.835, 0.875, 0.877],
    "bandwagon": [0.688, 0.638, 0.791, 0.690, 0.610, 0.710],
    "authority": [0.662, 0.846, 0.787, 0.796, 0.865, 0.779],
    "distraction": [0.713, 0.729, 0.790, 0.814, 0.878, 0.785],
    "diversity": [0.679, 0.855, 0.814, 0.788, 0.914, 0.826],
}

# Table 7: BIS = CR - RR, same model order as above.
PAPER_BIS = {
    "verbosity": [0.098, 0.075, 0.021, 0.083, 0.047, 0.110],
    "sentiment": [0.194, 0.337, 0.299, 0.291, 0.339, 0.343],
    "position": [0.340, 0.038, 0.149, 0.103, 0.083, 0.144],
    "compassion": [0.044, -0.002, 0.057, 0.049, 0.040, 0.027],
    "bandwagon": [0.218, 0.218, 0.134, 0.194, 0.305, 0.194],
    "authority": [0.244, 0.010, 0.138, 0.088, 0.050, 0.125],
    "distraction": [0.193, 0.127, 0.135, 0.070, 0.037, 0.119],
    "diversity": [0.227, 0.001, 0.111, 0.096, 0.001, 0.078],
}

# Table 5 and Table 8: error rates in percent.
PAPER_ERR_SE = [-16.64, 9.40, 8.18, 3.63, 5.39, 8.99]
PAPER_ERR_RA = [-5.48, 1.69, -3.22, 1.19, 2.22, 1.35]


def load(name):
    frame = pd.read_csv(ANALYSIS / name)
    frame["modèle"] = frame["model"].map(MODELS)
    return frame


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)


def plot_rr_vs_paper(summary):
    biases = list(PAPER_RR)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(biases))
    for i, bias in enumerate(biases):
        low, high = min(PAPER_RR[bias]), max(PAPER_RR[bias])
        ax.add_patch(
            plt.Rectangle(
                (i - 0.35, low), 0.7, high - low, color="lightgray", zorder=1
            )
        )
    offsets = dict(zip(MODEL_ORDER, (-0.2, 0.0, 0.2)))
    for label in MODEL_ORDER:
        rows = summary[summary["modèle"] == label].set_index("bias")
        values = [rows["rr"].get(bias, np.nan) for bias in biases]
        ax.scatter(
            x + offsets[label],
            values,
            s=70,
            color=PALETTE[label],
            label=label,
            zorder=3,
            edgecolor="black",
            linewidth=0.5,
        )
    ax.scatter([], [], marker="s", s=120, color="lightgray", label="Article (min–max, 6 juges)")
    ax.set_xticks(x)
    ax.set_xticklabels([BIASES[bias] for bias in biases])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Taux de robustesse (RR)")
    ax.set_title("RR : article vs. nos modèles")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "rr_article_vs_nous.png")


def plot_bis(summary):
    table = summary.dropna(subset=["cr"]).copy()
    table = table[table["bias"].isin(BIASES)]
    table["bis"] = table["cr"] - table["rr"]
    table["biais"] = table["bias"].map(BIASES)
    order = [BIASES[bias] for bias in BIASES if bias in set(table["bias"])]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=table,
        x="biais",
        y="bis",
        hue="modèle",
        hue_order=MODEL_ORDER,
        order=order,
        palette=PALETTE,
        ax=ax,
    )
    paper_mean = {BIASES[bias]: np.mean(values) for bias, values in PAPER_BIS.items()}
    for i, label in enumerate(order):
        if label in paper_mean:
            ax.hlines(
                paper_mean[label], i - 0.4, i + 0.4, color="black", linestyle="--", linewidth=1.2
            )
    ax.plot([], [], color="black", linestyle="--", label="Moyenne article")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("BIS = CR − RR")
    ax.set_title("Effet du biais au-delà du bruit (BIS)")
    ax.legend(fontsize=9)
    save(fig, "bis.png")


def plot_position(robust):
    rows = robust[(robust["bias"] == "position") & (robust["condition"] == "order_ba")]
    long = pd.concat(
        [
            pd.DataFrame(
                {"modèle": rows["modèle"], "ordre": "Bonne réponse en A", "précision": rows["acc_baseline"]}
            ),
            pd.DataFrame(
                {"modèle": rows["modèle"], "ordre": "Bonne réponse en B", "précision": rows["acc_condition"]}
            ),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(
        data=long,
        x="modèle",
        y="précision",
        hue="ordre",
        order=MODEL_ORDER,
        palette=["#4c72b0", "#dd8452"],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Précision")
    ax.set_title("Biais de position : précision selon l'emplacement de la bonne réponse")
    ax.legend(fontsize=9)
    save(fig, "position_precision.png")


def plot_rr_heatmap(robust):
    table = robust[robust["bias"] != "fallacy"].copy()
    table["condition_fr"] = table["bias"].map(BIASES) + " / " + table["condition"]
    pivot = table.pivot(index="condition_fr", columns="modèle", values="rr")
    pivot = pivot.reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8.5, 14))
    ax.grid(False)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="viridis",
        linewidths=0.4,
        cbar_kws={"label": "RR"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Taux de robustesse par condition", pad=24)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=9)
    save(fig, "rr_heatmap_fr.png")


def plot_cr_heatmap(consistency):
    table = consistency.copy()
    table["biais"] = table["bias"].map(BIASES)
    table = table.dropna(subset=["biais"])
    pivot = table.pivot(index="modèle", columns="biais", values="cr")
    pivot = pivot.reindex(index=MODEL_ORDER, columns=[b for b in BIASES.values() if b in pivot.columns])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.grid(False)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=0.5,
        vmax=1,
        cmap="viridis",
        linewidths=0.4,
        cbar_kws={"label": "CR"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Taux de cohérence (CR) — case vide : pas de répétition")
    ax.tick_params(axis="x", rotation=30)
    save(fig, "cr_heatmap_fr.png")


def plot_sentiment(robust):
    rows = robust[robust["bias"] == "sentiment"].copy()
    tones = {"cheerful": "Joyeux", "sad": "Triste", "angry": "En colère", "fear": "Peur"}
    rows["ton"] = rows["condition"].str.split("_").str[0].map(tones)
    rows["cible"] = np.where(
        rows["condition"].str.endswith("better"), "Ton sur la meilleure réponse", "Ton sur la moins bonne réponse"
    )
    grid = sns.catplot(
        data=rows,
        x="ton",
        y="rr",
        hue="modèle",
        col="cible",
        kind="bar",
        order=list(tones.values()),
        hue_order=MODEL_ORDER,
        palette=PALETTE,
        height=4.2,
        aspect=1.2,
    )
    grid.set_titles("{col_name}")
    grid.set_axis_labels("", "RR")
    grid.set(ylim=(0, 1))
    grid.figure.suptitle("Biais de sentiment", y=1.03)
    grid.figure.savefig(OUT / "sentiment_fr.png", dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def plot_scoring(refinement, self_enhancement):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ra = refinement.set_index("modèle")["error_rate_ra"] * 100
    se = self_enhancement.set_index("modèle")["error_rate_se"] * 100
    for ax, ours, paper, title in (
        (axes[0], ra, PAPER_ERR_RA, "Raffinement (Err_RA)"),
        (axes[1], se, PAPER_ERR_SE, "Self-enhancement (Err_SE)"),
    ):
        ours = ours.reindex(MODEL_ORDER)
        ax.grid(False, axis="x")
        ax.axhspan(min(paper), max(paper), color="lightgray", zorder=0, label="Article (min–max)")
        bars = ax.bar(
            ours.index, ours.values, color=[PALETTE[m] for m in ours.index], zorder=2
        )
        ax.bar_label(bars, fmt="%+.1f %%", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Taux d'erreur (%)")
        ax.tick_params(axis="x", labelsize=9)
        ax.legend(fontsize=8, loc="upper right")
    save(fig, "notation_fr.png")


def main():
    OUT.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    summary = load("bias_summary.csv")
    robust = load("robustness.csv")
    consistency = load("consistency.csv")
    refinement = load("refinement.csv")
    self_enhancement = load("self_enhancement.csv")

    plot_rr_vs_paper(summary)
    plot_bis(summary)
    plot_position(robust)
    plot_rr_heatmap(robust)
    plot_cr_heatmap(consistency)
    plot_sentiment(robust)
    plot_scoring(refinement, self_enhancement)
    print(f"Wrote French figures to {OUT}")


if __name__ == "__main__":
    main()
