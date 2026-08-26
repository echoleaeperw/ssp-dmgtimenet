"""Generate paper-result figures from tabulated HighD / NGSIM / ablation numbers.

These charts fill gaps where the manuscript currently only has tables
(N-extension, ablation, NGSIM, horizon, safety, latency).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils.plot_style import apply_paper_style, color_for, save_figure, style_axes

# ---------------------------------------------------------------------------
# Tabulated numbers (artifacts/paper_tables.md)
# ---------------------------------------------------------------------------

LEARNING = [
    "Int-LSTM",
    "Transformer",
    "Full-graph Attn",
    "LSTM",
    "SSP-DMGTimeNet",
    "CNN-Int-LSTM-IDM",
]

HIGHD = {
    "Int-LSTM": dict(v=0.102, tail=0.114, uns=91.30, gt_uns=88.89, gt_amp=2.865, retain=23, gt_retain=18, ttc=11.679, jerk=0.125),
    "Transformer": dict(v=0.158, tail=0.170, uns=85.71, gt_uns=83.33, gt_amp=2.735, retain=28, gt_retain=24, ttc=12.454, jerk=0.107),
    "Full-graph Attn": dict(v=0.164, tail=0.169, uns=84.62, gt_uns=86.36, gt_amp=3.529, retain=26, gt_retain=22, ttc=12.845, jerk=0.106),
    "LSTM": dict(v=0.213, tail=0.209, uns=95.12, gt_uns=100.0, gt_amp=3.240, retain=41, gt_retain=11, ttc=14.185, jerk=0.121),
    "SSP-DMGTimeNet": dict(v=0.347, tail=0.222, uns=0.65, gt_uns=0.0, gt_amp=0.898, retain=1847, gt_retain=62, ttc=12.175, jerk=0.116),
    "CNN-Int-LSTM-IDM": dict(v=0.564, tail=0.543, uns=88.89, gt_uns=93.33, gt_amp=7.253, retain=18, gt_retain=15, ttc=14.020, jerk=2.727),
}

NGSIM_US101 = {
    "Int-LSTM": dict(v=3.343, uns=96.10, gt_uns=96.08, gt_amp=16.581),
    "Transformer": dict(v=1.340, uns=95.61, gt_uns=95.49, gt_amp=9.235),
    "Full-graph Attn": dict(v=1.321, uns=90.69, gt_uns=90.87, gt_amp=10.022),
    "LSTM": dict(v=2.333, uns=88.72, gt_uns=88.55, gt_amp=6.359),
    "SSP-DMGTimeNet": dict(v=1.316, uns=3.90, gt_uns=4.12, gt_amp=1.791),
    "CNN-Int-LSTM-IDM": dict(v=1.286, uns=92.74, gt_uns=92.77, gt_amp=None),  # explosion → omit amp
}

NGSIM_I80 = {
    "Int-LSTM": dict(v=3.721, uns=96.27, gt_uns=96.21, gt_amp=20.399),
    "Transformer": dict(v=1.277, uns=97.33, gt_uns=97.29, gt_amp=8.499),
    "Full-graph Attn": dict(v=1.256, uns=88.71, gt_uns=88.54, gt_amp=9.407),
    "LSTM": dict(v=2.148, uns=87.52, gt_uns=86.83, gt_amp=5.004),
    "SSP-DMGTimeNet": dict(v=1.252, uns=4.10, gt_uns=4.22, gt_amp=1.536),
    "CNN-Int-LSTM-IDM": dict(v=1.108, uns=93.95, gt_uns=93.99, gt_amp=None),
}

N_EXT = {
    "SSP-DMGTimeNet": {
        3: dict(v=0.268, uns=0.00, gt_amp=0.726),
        5: dict(v=0.347, uns=0.65, gt_amp=0.898),
        6: dict(v=0.401, uns=0.96, gt_amp=1.033),
        7: dict(v=0.397, uns=24.73, gt_amp=1.217),
    },
    "Int-LSTM": {
        3: dict(v=0.091, uns=36.14, gt_amp=2.296),
        5: dict(v=0.102, uns=91.30, gt_amp=2.865),
        6: dict(v=0.113, uns=100.0, gt_amp=2.383),
        7: dict(v=0.131, uns=None, gt_amp=None),
    },
    "Transformer": {
        3: dict(v=0.128, uns=30.02, gt_amp=2.391),
        5: dict(v=0.158, uns=85.71, gt_amp=2.735),
        6: dict(v=0.207, uns=100.0, gt_amp=1.084),
        7: dict(v=0.223, uns=100.0, gt_amp=1.479),
    },
}

ABLATION = [
    ("SSP (full)", 0.347, 0.00, 0.898, 2.165),
    ("w/o delay bias", 0.382, 0.00, 0.896, 2.111),
    ("w/o adj", 0.336, 0.00, 0.973, 2.170),
    ("w/o CFE", 0.385, 0.00, 0.936, 1.832),
    ("full graph", 0.372, 0.00, 0.926, 2.593),
    ("w/o sub", 0.366, 0.00, 0.864, 2.237),
    ("w/o HGF", 0.371, 1.61, 1.036, 3.746),
    ("fixed τ", 0.375, 1.61, 1.101, 2.037),
    ("w/o FFT", 0.303, 8.06, 1.103, 10.208),
]

HORIZON = {
    "Int-LSTM": dict(v=(0.071, 0.078, 0.102), s=(0.221, 0.219, 0.262), a=(0.018, 0.043, 0.068)),
    "Transformer": dict(v=(0.126, 0.134, 0.158), s=(0.358, 0.363, 0.399), a=(0.029, 0.056, 0.079)),
    "LSTM": dict(v=(0.209, 0.196, 0.213), s=(0.555, 0.483, 0.511), a=(0.027, 0.050, 0.074)),
    "SSP-DMGTimeNet": dict(v=(0.464, 0.368, 0.347), s=(0.270, 0.257, 0.290), a=(0.027, 0.054, 0.077)),
}

SENS = {
    "Int-LSTM": dict(v0=3.918, v1=2.279, u0=96.98, u1=96.90),
    "Transformer": dict(v0=1.378, v1=0.898, u0=96.94, u1=95.50),
    "Full-graph Attn": dict(v0=1.369, v1=0.880, u0=91.01, u1=86.77),
    "LSTM": dict(v0=2.157, v1=1.914, u0=86.96, u1=91.22),
    "SSP-DMGTimeNet": dict(v0=1.319, v1=1.074, u0=3.75, u1=3.23),
    "CNN-Int-LSTM-IDM": dict(v0=1.129, v1=1.080, u0=93.33, u1=91.23),
}

LATENCY = {
    "Int-LSTM": 0.790,
    "Transformer": 1.151,
    "Full-graph Attn": 2.202,
    "LSTM": 0.593,
    "SSP-DMGTimeNet": 8.742,
    "CNN-Int-LSTM-IDM": 30.232,
}


def _short(name: str) -> str:
    return {
        "SSP-DMGTimeNet": "SSP (ours)",
        "Full-graph Attn": "Full-graph",
        "CNN-Int-LSTM-IDM": "CNN-IDM",
        "SSP (full)": "SSP",
    }.get(name, name)


def _bar_colors(names: list[str]) -> list[str]:
    return [color_for(n if n != "SSP (ours)" else "SSP-DMGTimeNet", i) for i, n in enumerate(names)]


def fig_highd_main(out: Path) -> None:
    names = LEARNING
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    # v-MAE
    ax = axes[0]
    style_axes(ax, grid=False)
    vals = [HIGHD[n]["v"] for n in names]
    bars = ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    bars[names.index("SSP-DMGTimeNet")].set_edgecolor("#111111")
    bars[names.index("SSP-DMGTimeNet")].set_linewidth(1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel(r"$v$-MAE (m/s)")
    ax.set_title("(a) Accuracy")
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    # unstable %
    ax = axes[1]
    style_axes(ax, grid=False)
    vals = [HIGHD[n]["uns"] for n in names]
    bars = ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    bars[names.index("SSP-DMGTimeNet")].set_edgecolor("#111111")
    bars[names.index("SSP-DMGTimeNet")].set_linewidth(1.4)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("Unstable window (%)")
    ax.set_title("(b) Stability (per-model floor)")
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    # GT max amp
    ax = axes[2]
    style_axes(ax, grid=False)
    vals = [HIGHD[n]["gt_amp"] for n in names]
    bars = ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    bars[names.index("SSP-DMGTimeNet")].set_edgecolor("#111111")
    bars[names.index("SSP-DMGTimeNet")].set_linewidth(1.4)
    ax.axhline(1.0, color="#C0392B", ls="--", lw=1.6, label=r"$A=1$")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("GT-subset max amplification")
    ax.set_title("(c) GT-excited response")
    ax.legend(loc="upper right", frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("HighD main results (learning models)", y=1.02)
    save_figure(fig, out / "highd_main_bars.png")
    plt.close(fig)


def fig_coverage(out: Path) -> None:
    names = LEARNING
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    style_axes(ax, grid=False)
    r1 = [HIGHD[n]["retain"] for n in names]
    r2 = [HIGHD[n]["gt_retain"] for n in names]
    ax.bar(x - w / 2, r1, width=w, color="#2980B9", label="per-model floor / 1847", edgecolor="white")
    ax.bar(x + w / 2, r2, width=w, color="#C0392B", label="GT-subset retain / 62", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("Retained windows")
    ax.set_title("Excitation coverage (HighD test)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, out / "highd_coverage.png")
    plt.close(fig)


def fig_ngsim(out: Path) -> None:
    names = LEARNING
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    style_axes(ax, grid=False)
    ax.bar(x - w / 2, [NGSIM_US101[n]["v"] for n in names], width=w, color="#2980B9", label="US-101", edgecolor="white")
    ax.bar(x + w / 2, [NGSIM_I80[n]["v"] for n in names], width=w, color="#C0392B", label="I-80", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel(r"$v$-MAE (m/s)")
    ax.set_title("(a) Zero-shot accuracy")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    style_axes(ax, grid=False)
    ax.bar(x - w / 2, [NGSIM_US101[n]["uns"] for n in names], width=w, color="#2980B9", label="US-101", edgecolor="white")
    ax.bar(x + w / 2, [NGSIM_I80[n]["uns"] for n in names], width=w, color="#C0392B", label="I-80", edgecolor="white")
    ax.set_yscale("symlog", linthresh=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("Unstable window (%)")
    ax.set_title("(b) Zero-shot stability")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("NGSIM zero-shot (HighD → NGSIM)", y=1.02)
    save_figure(fig, out / "ngsim_zeroshot.png")
    plt.close(fig)


def fig_n_extension(out: Path) -> None:
    """Platoon-length robustness: stability / accuracy / chain gain vs N."""
    ns = [3, 5, 6, 7]
    # GT-excited support sizes (same protocol as paper tables).
    gt_n = {3: 495, 5: 62, 6: 22, 7: 2}
    models = ("SSP-DMGTimeNet", "Int-LSTM", "Transformer")
    markers = {"SSP-DMGTimeNet": "o", "Int-LSTM": "s", "Transformer": "^"}
    labels = {
        "SSP-DMGTimeNet": "SSP-DMGTimeNet (ours)",
        "Int-LSTM": "Int-LSTM",
        "Transformer": "Transformer",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0))
    plotted = {}

    # (a) unstable-window rate (log)
    ax = axes[0]
    style_axes(ax)
    for name in models:
        ys = []
        for n in ns:
            val = N_EXT[name][n]["uns"]
            if val is None:
                ys.append(np.nan)
            elif float(val) <= 0.0:
                ys.append(5e-2)  # visual floor for log axis
            else:
                ys.append(float(val))
        (line,) = ax.plot(
            ns,
            ys,
            marker=markers[name],
            ms=9 if "SSP" in name else 8,
            color=color_for(name),
            lw=2.4,
            label=labels[name],
        )
        plotted[labels[name]] = line
    ax.set_yscale("log")
    ax.set_xlabel(r"Platoon length $N$")
    ax.set_ylabel("Unstable window ratio (%)")
    ax.set_title("(a) Temporal instability vs platoon length")
    ax.set_xticks(ns)
    ax.set_ylim(3e-2, 2e2)

    # (b) v-MAE
    ax = axes[1]
    style_axes(ax)
    for name in models:
        ys = [N_EXT[name][n]["v"] for n in ns]
        ax.plot(
            ns,
            ys,
            marker=markers[name],
            ms=9 if "SSP" in name else 8,
            color=color_for(name),
            lw=2.4,
            label=labels[name],
        )
    ax.set_xlabel(r"Platoon length $N$")
    ax.set_ylabel(r"$v$-MAE (m/s)")
    ax.set_title("(b) Accuracy vs platoon length")
    ax.set_xticks(ns)
    ax.set_ylim(0.05, 0.48)

    # (c) GT-subset max amplification
    ax = axes[2]
    style_axes(ax)
    for name in models:
        ys = [
            N_EXT[name][n]["gt_amp"] if N_EXT[name][n]["gt_amp"] is not None else np.nan
            for n in ns
        ]
        ax.plot(
            ns,
            ys,
            marker=markers[name],
            ms=9 if "SSP" in name else 8,
            color=color_for(name),
            lw=2.4,
            label=labels[name],
        )
    from matplotlib.lines import Line2D

    ax.axhline(1.0, color="#555555", ls="--", lw=1.5, zorder=1)
    ax.set_xlabel(r"Platoon length $N$")
    ax.set_ylabel("GT-subset max amplification")
    ax.set_title("(c) Chain amplification on GT-excited subset")
    ax.set_xticks(ns)
    # Encode GT support in tick labels to avoid floating annotations.
    ax.set_xticklabels([f"{n}\n($n={gt_n[n]}$)" for n in ns])
    ax.set_ylim(0.55, 3.20)

    # One shared legend below all panels — avoids in-axes collisions.
    bound_proxy = Line2D([0], [0], color="#555555", ls="--", lw=1.5, label="Stability boundary")
    fig.legend(
        list(plotted.values()) + [bound_proxy],
        list(plotted.keys()) + ["Stability boundary"],
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.90, bottom=0.20, wspace=0.30)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("n_extension.png", "n_ext_trend.png"):
        fig.savefig(out / name, dpi=220, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def fig_ablation(out: Path) -> None:
    names = [a[0] for a in ABLATION]
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    v = [a[1] for a in ABLATION]
    gt_uns = [a[2] for a in ABLATION]
    gt_amp = [a[3] for a in ABLATION]
    fft = [a[4] for a in ABLATION]
    colors = ["#C0392B" if i == 0 else "#5D6D7E" for i in range(len(names))]
    colors[-1] = "#E67E22"  # highlight w/o FFT

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    specs = [
        (axes[0, 0], v, r"$v$-MAE (m/s)", "(a) Accuracy"),
        (axes[0, 1], gt_uns, "GT-subset unstable (%)", "(b) GT unstable"),
        (axes[1, 0], gt_amp, "GT-subset max amp", "(c) Chain gain"),
        (axes[1, 1], fft, r"$\mathrm{fft\_gain\_max}$", "(d) Spectral gain"),
    ]
    for ax, vals, ylab, title in specs:
        style_axes(ax, grid=False)
        ax.bar(x, vals, color=colors, edgecolor="white", width=0.75)
        if "amp" in ylab.lower() or "gain" in ylab.lower():
            ax.axhline(1.0, color="#C0392B", ls="--", lw=1.3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=35, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("Ablation on HighD (GT-excited subset)", y=1.01)
    save_figure(fig, out / "ablation_bars.png")
    plt.close(fig)


def fig_horizon(out: Path) -> None:
    hs = np.array([1.0, 2.0, 3.0])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, key, title in zip(
        axes,
        ("v", "s", "a"),
        (r"(a) $v$-MAE", r"(b) $s$-MAE", r"(c) $a$-MAE"),
        strict=True,
    ):
        style_axes(ax)
        for name, series in HORIZON.items():
            ax.plot(
                hs,
                series[key],
                marker="o",
                ms=8 if "SSP" in name else 6,
                lw=2.5 if "SSP" in name else 2.0,
                color=color_for(name),
                label=_short(name),
            )
        ax.set_xlabel("Horizon (s)")
        ax.set_ylabel("MAE")
        ax.set_title(title)
        ax.set_xticks(hs)
        ax.legend(frameon=False, fontsize=12)
    fig.suptitle("Prefix-horizon accuracy (HighD)", y=1.02)
    save_figure(fig, out / "horizon_mae.png")
    plt.close(fig)


def fig_safety(out: Path) -> None:
    names = [n for n in LEARNING if n != "CNN-Int-LSTM-IDM"] + ["CNN-Int-LSTM-IDM"]
    # CNN jerk is outlier; keep but use twin or clip note
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    style_axes(ax, grid=False)
    vals = [HIGHD[n]["ttc"] for n in names]
    ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("TTC $p_{05}$ (s)")
    ax.set_title("(a) Safety margin")
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    style_axes(ax, grid=False)
    vals = [HIGHD[n]["jerk"] for n in names]
    ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("RMS jerk")
    ax.set_title("(b) Comfort (log scale)")
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("Safety / comfort (HighD test)", y=1.02)
    save_figure(fig, out / "safety_comfort.png")
    plt.close(fig)


def fig_latency(out: Path) -> None:
    names = list(LATENCY.keys())
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    style_axes(ax, grid=False)
    vals = [LATENCY[n] for n in names]
    ax.bar(x, vals, color=_bar_colors(names), edgecolor="white", width=0.72)
    ax.axhline(100.0, color="#7F8C8D", ls="--", lw=1.5, label="10 Hz budget (100 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("GPU median latency (ms)")
    ax.set_title("Single-sample inference latency (batch=1)")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)
    save_figure(fig, out / "latency.png")
    plt.close(fig)


def fig_sensitivity(out: Path) -> None:
    names = list(SENS.keys())
    short = [_short(n) for n in names]
    x = np.arange(len(names))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    style_axes(ax, grid=False)
    ax.bar(x - w / 2, [SENS[n]["v0"] for n in names], width=w, color="#7F8C8D", label="raw", edgecolor="white")
    ax.bar(x + w / 2, [SENS[n]["v1"] for n in names], width=w, color="#2980B9", label="MP reconstr.", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel(r"$v$-MAE (m/s)")
    ax.set_title("(a) Accuracy sensitivity")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    style_axes(ax, grid=False)
    ax.bar(x - w / 2, [SENS[n]["u0"] for n in names], width=w, color="#7F8C8D", label="raw", edgecolor="white")
    ax.bar(x + w / 2, [SENS[n]["u1"] for n in names], width=w, color="#C0392B", label="MP reconstr.", edgecolor="white")
    ax.set_yscale("symlog", linthresh=5.0)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=25, ha="right")
    ax.set_ylabel("Unstable window (%)")
    ax.set_title("(b) Stability sensitivity")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#D0D0D0", alpha=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("I-80 0400–0415: raw vs Montanino–Punzo", y=1.02)
    save_figure(fig, out / "ngsim_sensitivity.png")
    plt.close(fig)


def fig_tail_vs_mean(out: Path) -> None:
    names = LEARNING
    short = [_short(n) for n in names]
    mean_v = np.array([HIGHD[n]["v"] for n in names])
    tail_v = np.array([HIGHD[n]["tail"] for n in names])
    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    style_axes(ax)
    for i, n in enumerate(names):
        ax.scatter(mean_v[i], tail_v[i], s=220 if "SSP" in n else 120,
                   marker="*" if "SSP" in n else "o",
                   color=color_for(n, i), edgecolors="white", linewidths=0.8, label=_short(n), zorder=3)
    lim0, lim1 = 0.05, 0.60
    ax.plot([lim0, lim1], [lim0, lim1], color="#7F8C8D", ls="--", lw=1.5, label="tail = mean")
    ax.set_xlim(lim0, lim1)
    ax.set_ylim(lim0, lim1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Mean $v$-MAE (m/s)")
    ax.set_ylabel(r"Tail $v$-MAE (m/s)")
    ax.set_title("Error accumulation along the platoon")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    save_figure(fig, out / "tail_vs_mean_mae.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("../artifacts/figures/paper_results"))
    parser.add_argument("--font-size", type=float, default=15.0)
    args = parser.parse_args()
    apply_paper_style(args.font_size)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    fig_highd_main(out)
    fig_coverage(out)
    fig_tail_vs_mean(out)
    fig_ngsim(out)
    fig_n_extension(out)
    fig_ablation(out)
    fig_horizon(out)
    fig_safety(out)
    fig_latency(out)
    fig_sensitivity(out)
    print(f"Wrote paper-result figures to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
