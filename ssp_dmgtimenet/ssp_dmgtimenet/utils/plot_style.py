"""Paper figure style: Times-compatible roman + clean academic defaults."""

from __future__ import annotations

import logging
import matplotlib as mpl
from matplotlib import font_manager

_LOG = logging.getLogger("ssp.plot_style")

_TIMES_CANDIDATES = (
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "DejaVu Serif",
)

# Colorblind-friendly, muted palette. SSP is always first/red-accent.
MODEL_COLORS = {
    "ground truth": "#222222",
    "GT": "#222222",
    "SSP-DMGTimeNet": "#C0392B",
    "SSP-DMGTimeNet (ours)": "#C0392B",
    "Int-LSTM": "#2980B9",
    "Transformer": "#27AE60",
    "LSTM": "#8E44AD",
    "Full-graph Attn": "#D35400",
    "Full-graph Attention": "#D35400",
    "CNN-Int-LSTM-IDM": "#7F8C8D",
}

_FALLBACK_CYCLE = ("#2980B9", "#27AE60", "#8E44AD", "#D35400", "#16A085", "#7F8C8D")


def resolve_times_family() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _TIMES_CANDIDATES:
        if name in available:
            return name
    return "DejaVu Serif"


def color_for(name: str, index: int = 0) -> str:
    if name in MODEL_COLORS:
        return MODEL_COLORS[name]
    return _FALLBACK_CYCLE[index % len(_FALLBACK_CYCLE)]


def apply_paper_style(font_size_pt: float = 15.0) -> str:
    """Apply Times-like roman text and cleaner matplotlib defaults."""
    family = resolve_times_family()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "DejaVu Serif", "serif"],
            "font.size": font_size_pt,
            "axes.titlesize": font_size_pt,
            "axes.labelsize": font_size_pt,
            "xtick.labelsize": font_size_pt,
            "ytick.labelsize": font_size_pt,
            "legend.fontsize": font_size_pt - 1,
            "figure.titlesize": font_size_pt,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.0,
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.edgecolor": "#B0B0B0",
            "legend.framealpha": 0.92,
            "legend.borderpad": 0.4,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    _LOG.info("Matplotlib paper style: family=%s size=%.1fpt", family, font_size_pt)
    return family


# Back-compat alias used by older call sites.
def apply_times_roman_10pt(font_size_pt: float = 15.0) -> str:
    return apply_paper_style(font_size_pt=font_size_pt)


def style_axes(ax, *, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, which="major", color="#D0D0D0", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)


def save_figure(fig, out_path, *, dpi: int = 220) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
