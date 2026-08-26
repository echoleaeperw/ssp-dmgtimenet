"""Plot paper-facing summaries for the three-layer stability protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "evaluation_v3" / "reports"
EXTENSIONS = ROOT / "artifacts" / "evaluation_v3" / "extensions"
OUTPUT_DIR = ROOT / "artifacts" / "evaluation_v3" / "figures"

LEARNED_MODELS = [
    ("SSP", "ssp_dmgtimenet_v6", "#d62728"),
    ("Int-LSTM", "interaction_lstm", "#1f77b4"),
    ("Transformer", "platoon_transformer", "#2ca02c"),
    ("Full-graph", "full_graph_attention", "#9467bd"),
    ("LSTM", "platoon_lstm", "#8c564b"),
    ("CNN hybrid", "cnn_int_lstm_idm", "#ff7f0e"),
]
N_MODELS = [
    ("SSP", "ssp_dmgtimenet_v6", "#d62728", "o"),
    ("Int-LSTM", "interaction_lstm", "#1f77b4", "s"),
    ("Transformer", "platoon_transformer", "#2ca02c", "^"),
]


def load(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    return {key: float(value) for key, value in metrics.items()}


def plot_highd() -> None:
    data = {
        directory: load(REPORTS / directory / "test_report.json")
        for _, directory, _ in LEARNED_MODELS
    }
    labels = [label for label, _, _ in LEARNED_MODELS]
    colors = [color for _, _, color in LEARNED_MODELS]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    coverage = [
        100.0 * data[directory]["stab/detection_coverage"]
        for _, directory, _ in LEARNED_MODELS
    ]
    fpr = [
        100.0 * data[directory]["stab/detection_fpr"]
        for _, directory, _ in LEARNED_MODELS
    ]
    width = 0.38
    axes[0].bar(x - width / 2, coverage, width, label="Coverage", color="#4c78a8")
    axes[0].bar(x + width / 2, fpr, width, label="FPR", color="#f58518")
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("(a) Leader-disturbance detection")
    axes[0].legend(frameon=False)

    external_p95 = [
        data[directory]["stab/gt_ref_p95_gain"]
        for _, directory, _ in LEARNED_MODELS
    ]
    axes[1].bar(x, external_p95, color=colors)
    axes[1].axhline(1.05, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("GT-referenced p95 gain")
    axes[1].set_title("(b) Unified external response")

    conditional = [
        100.0 * data[directory]["stab/conditional_internal_unstable_window_ratio"]
        for _, directory, _ in LEARNED_MODELS
    ]
    support = [
        round(data[directory]["stab/conditional_internal_n_windows"])
        for _, directory, _ in LEARNED_MODELS
    ]
    axes[2].bar(x, conditional, color=colors)
    axes[2].set_ylabel("Conditional instability (%)")
    axes[2].set_title("(c) Internal stability given detection")
    for idx, (rate, count) in enumerate(zip(conditional, support, strict=True)):
        axes[2].text(idx, rate + 1.0, f"n={count}", ha="center", va="bottom", fontsize=8)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "highd_protocol_v3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_n_extension() -> None:
    n_values = [3, 5, 6, 7]
    data: dict[str, dict[int, dict[str, float]]] = {}
    gt_support: dict[int, int] = {}
    for _, directory, _, _ in N_MODELS:
        data[directory] = {}
        for n_value in n_values:
            metrics = load(
                EXTENSIONS / f"n_ext_N{n_value}" / directory / "test_report.json"
            )
            data[directory][n_value] = metrics
            gt_support[n_value] = round(metrics["stab/detection_n_gt_excited"])

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    keys = [
        ("stab/detection_coverage", "Coverage (%)", 100.0),
        ("stab/gt_ref_p95_gain", "GT-referenced p95 gain", 1.0),
        (
            "stab/conditional_internal_unstable_window_ratio",
            "Conditional instability (%)",
            100.0,
        ),
    ]
    for axis, (key, ylabel, scale) in zip(axes, keys, strict=True):
        for label, directory, color, marker in N_MODELS:
            values = [scale * data[directory][n_value][key] for n_value in n_values]
            values = [np.nan if not math.isfinite(value) else value for value in values]
            axis.plot(
                n_values,
                values,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=7,
                label=label,
            )
        axis.set_xlabel("Platoon length N")
        axis.set_ylabel(ylabel)
        axis.set_xticks(n_values)
        axis.grid(alpha=0.25)
    axes[1].axhline(1.05, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("(a) Disturbance coverage")
    axes[1].set_title("(b) Unified external response")
    axes[2].set_title("(c) Conditional internal stability")
    axes[0].legend(frameon=False)
    for n_value in n_values:
        axes[1].text(
            n_value,
            axes[1].get_ylim()[1] * 0.97,
            f"GT n={gt_support[n_value]}",
            ha="center",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "n_extension_v3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_highd()
    plot_n_extension()
    print(f"written: {OUTPUT_DIR / 'highd_protocol_v3.png'}")
    print(f"written: {OUTPUT_DIR / 'n_extension_v3.png'}")


if __name__ == "__main__":
    main()
