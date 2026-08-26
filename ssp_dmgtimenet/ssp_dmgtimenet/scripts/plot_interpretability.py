"""Generate interpretability and stability figures for scheme C.

The script accepts one or more (config, checkpoint) pairs (the first acts
as the reference for the τ analysis) and a dataset split. It emits the
seven figure categories from scheme-C §6.5:

1. Learned ``τ_i`` along the platoon position.
2. True vs predicted disturbance propagation delay (cross-correlation peaks).
3. Spatio-temporal velocity heatmaps: ground truth / each model.
4. Adjacent amplification ``A_i`` boxplot.
5. Sub-platoon ``A_{j→i}`` heatmap.
6. Frequency-domain transfer gain ``G(f)`` curves.
7. Stability/accuracy Pareto scatter (MAE vs unstable-window-ratio).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.dataset import build_platoon_loaders
from ..losses.total import resolve_target_indices
from ..metrics.stability import (
    adjacent_amplification,
    fft_gain,
    leader_excitation_amplitude,
    phase_delay,
    subplatoon_amplification,
)
from ..training.evaluator import EvaluatorConfig
from ..training.factory import build_model, initialise_normalisation_for_model
from ..utils.config import load_config
from ..utils.plot_style import apply_paper_style, color_for, save_figure, style_axes


@dataclass(slots=True)
class ModelArtifact:
    name: str
    config_path: Path
    checkpoint_path: Path

    def load(self, device: torch.device) -> tuple[torch.nn.Module, dict, list[int]]:
        cfg = load_config(self.config_path)
        model = build_model(cfg["model"])
        state = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict)
        model = model.to(device).eval()
        vars_ = cfg.get("loss", {}).get("prediction", {}).get("variables") or []
        return model, cfg.to_dict(), list(vars_)


def _parse_models(arg: list[str]) -> list[ModelArtifact]:
    out: list[ModelArtifact] = []
    for item in arg:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"--model expects 'name:config:checkpoint', got {item!r}")
        name, config_path, ckpt_path = parts
        out.append(ModelArtifact(name=name, config_path=Path(config_path), checkpoint_path=Path(ckpt_path)))
    return out


def _gather_predictions(
    model: torch.nn.Module,
    loader,
    output_channels: Sequence[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    history_v: list[np.ndarray] = []
    hist_v_idx = resolve_target_indices(("v",))[0]
    with torch.no_grad():
        for batch in loader:
            history_raw = batch["history_raw"].to(device, non_blocking=True)
            history_mask = batch["history_mask"].to(device, non_blocking=True)
            future_raw = batch["future_raw"].to(device, non_blocking=True)
            future_mask = batch["future_mask"].to(device, non_blocking=True)
            output = model(history_raw, history_mask)
            pred = output["predictions"].detach().cpu().numpy()
            idx = list(resolve_target_indices(tuple(output_channels)))
            target = future_raw[..., idx].detach().cpu().numpy()
            mask = future_mask[..., idx].detach().cpu().numpy()
            preds.append(pred)
            targets.append(target)
            masks.append(mask)
            history_v.append(history_raw[..., hist_v_idx].detach().cpu().numpy())
    if not preds:
        raise RuntimeError("No samples available for plotting")
    return {
        "pred": np.concatenate(preds, axis=0),
        "target": np.concatenate(targets, axis=0),
        "mask": np.concatenate(masks, axis=0),
        "history_v": np.concatenate(history_v, axis=0),
    }


def _plot_tau(model: torch.nn.Module, out_path: Path) -> None:
    if not hasattr(model, "sp_daca_blocks"):
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    style_axes(ax)
    tau_min: float | None = None
    tau_max: float | None = None
    n_pairs = 0
    all_vals: list[np.ndarray] = []
    cmap = plt.get_cmap("tab20")
    line_id = 0
    for layer_idx, block in enumerate(model.sp_daca_blocks):
        cfg = block.attn.config
        tau_min, tau_max = float(cfg.tau_min), float(cfg.tau_max)
        tau = block.attn.tau.detach().cpu().numpy()  # (H, N-1)
        n_pairs = tau.shape[1]
        for h in range(tau.shape[0]):
            ax.plot(
                np.arange(n_pairs) + 1,
                tau[h],
                marker="o",
                markersize=5,
                color=cmap(line_id % 20),
                label=f"L{layer_idx}-H{h}",
                alpha=0.9,
            )
            all_vals.append(tau[h])
            line_id += 1
    vals = np.concatenate(all_vals) if all_vals else np.zeros(1)
    if tau_min is not None and tau_max is not None:
        collapse = 0.5 * (tau_min + tau_max)
        ax.axhline(
            y=collapse,
            color="#7F8C8D",
            linestyle="--",
            linewidth=1.6,
            label=rf"prior mid ($\tau$={collapse:.2f}s)",
        )
        ax.axhspan(0.8, 1.2, color="#27AE60", alpha=0.10, label="plausible 0.8–1.2 s")
    if n_pairs:
        ax.set_xticks(np.arange(1, n_pairs + 1))
    ax.set_xlabel(r"Pair index $i \to i+1$")
    ax.set_ylabel(r"Learned $\tau_i$ (s)")
    ax.set_title(rf"SP-DACA $\tau$: {vals.min():.2f}–{vals.max():.2f} s")
    ax.legend(loc="upper right", ncol=2, fontsize=11)
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_phase_delay_scatter(
    truth_long: np.ndarray,
    predictions_long: dict[str, np.ndarray],
    truth_gate: np.ndarray,
    target_hz: float,
    out_path: Path,
    max_lag_seconds: float = 2.5,
    excitation_floor: float = 0.05,
) -> None:
    # Two design choices keep this estimator physically meaningful:
    # (1) Gate on the prediction-window GT leader excitation (detrended RMS >=
    #     floor): on HighD only ~3.4% of windows carry an upstream disturbance,
    #     and this is exactly the table-6.1b GT-excited subset, so every model
    #     is compared on the same externally-defined disturbance events.
    # (2) Estimate the lag on the 8s history+prediction window WITHOUT the
    #     short running-mean detrend (it high-passes away the slow stop-and-go
    #     oscillation that actually propagates, leaving noise); a bare
    #     mean-removal preserves it. ``max_lag`` is capped at tau_max=2.5s.
    keep = leader_excitation_amplitude(truth_gate, detrend_window_steps=8) >= excitation_floor
    if not np.any(keep):
        keep = np.ones(truth_gate.shape[0], dtype=bool)
    n_kept = int(keep.sum())
    truth_delays = phase_delay(
        truth_long[keep], target_hz=target_hz, detrend_window_steps=0,
        max_lag_seconds=max_lag_seconds, subsample=True,
    ).flatten()
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    style_axes(ax)
    for i, (name, pred) in enumerate(predictions_long.items()):
        pred_delays = phase_delay(
            pred[keep], target_hz=target_hz, detrend_window_steps=0,
            max_lag_seconds=max_lag_seconds, subsample=True,
        ).flatten()
        if truth_delays.std() > 0 and pred_delays.std() > 0:
            r = float(np.corrcoef(truth_delays, pred_delays)[0, 1])
            label = f"{name} ($r$={r:+.2f})"
        else:
            label = name
        ax.scatter(
            truth_delays,
            pred_delays,
            alpha=0.65,
            s=36 if "SSP" in name else 28,
            c=color_for(name, i),
            edgecolors="white",
            linewidths=0.4,
            label=label,
            zorder=3 if "SSP" in name else 2,
        )
    lim = float(np.nanmax(np.abs(truth_delays))) * 1.1 + 0.1
    ax.plot([-lim, lim], [-lim, lim], color="#7F8C8D", linestyle="--", linewidth=1.5, label="ideal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True propagation delay (s)")
    ax.set_ylabel("Predicted propagation delay (s)")
    ax.set_title(f"Propagation delay (GT-excited, $n$={n_kept})")
    ax.legend(loc="lower right")
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_velocity_heatmap(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_hz: float,
    out_path: Path,
    sample_idx: int = 0,
) -> None:
    panels = ["GT", *list(predictions.keys())]
    data = [truth[sample_idx]] + [predictions[n][sample_idx] for n in predictions]
    vmin = min(float(d.min()) for d in data)
    vmax = max(float(d.max()) for d in data)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.5, 1.55 * len(panels) + 0.6), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, name, arr in zip(axes, panels, data, strict=True):
        extent = [0, arr.shape[0] / target_hz, arr.shape[1] - 0.5, -0.5]
        im = ax.imshow(arr.T, aspect="auto", extent=extent, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_ylabel("vehicle")
        ax.set_yticks(range(arr.shape[1]))
        ax.set_yticklabels([rf"$C_{{{i + 1}}}$" for i in range(arr.shape[1])])
        ax.set_title(name, loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
        cbar.set_label(r"$v$ (m/s)")
    axes[-1].set_xlabel("Future time (s)")
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_adjacent_amp_box(
    predictions: dict[str, np.ndarray],
    target_hz: float,
    out_path: Path,
) -> None:
    # Log scale: the DMG cascade blows up to A~5e3 which on a linear axis
    # flattens every other model's box onto the x-axis.
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    style_axes(ax, grid=False)
    data = []
    labels = []
    colors = []
    for i, (name, pred) in enumerate(predictions.items()):
        amp = adjacent_amplification(pred, detrend_window_steps=8)
        if amp.ndim == 1:
            amp = amp[None, :]
        data.append(amp.flatten())
        labels.append(name)
        colors.append(color_for(name, i))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.3},
        boxprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch, c in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(y=1.0, color="#C0392B", linestyle="--", linewidth=1.6, label=r"$A=1$")
    ax.set_yscale("log")
    ax.set_ylabel(r"Adjacent amplification $A_i$")
    ax.set_title("Adjacent amplification distribution")
    ax.tick_params(axis="x", rotation=15)
    ax.yaxis.grid(True, which="major", color="#D0D0D0", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_subplatoon_heatmap(pred: np.ndarray, name: str, out_path: Path) -> None:
    sub_amp = subplatoon_amplification(pred, detrend_window_steps=8)
    if sub_amp.ndim == 2:
        avg = sub_amp
    else:
        avg = sub_amp.mean(axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(avg, cmap="RdBu_r", vmin=0.5, vmax=2.0)
    n = avg.shape[0]
    for i in range(n):
        for j in range(n):
            if j >= i:
                continue
            ax.text(i, j, f"{avg[j, i]:.2f}", ha="center", va="center", color="#111111", fontsize=12)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([rf"$C_{{{k + 1}}}$" for k in range(n)])
    ax.set_yticklabels([rf"$C_{{{k + 1}}}$" for k in range(n)])
    ax.set_title(rf"Sub-platoon $A_{{j\to i}}$ ({name})")
    ax.set_xlabel("Downstream $i$")
    ax.set_ylabel("Upstream $j$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$A_{j\to i}$")
    save_figure(fig, out_path)
    plt.close(fig)


def _mean_pair_gain(gain: np.ndarray) -> np.ndarray:
    """Average G_{j->i}(f) over batch and the strictly-upper (j<i) pairs."""

    if gain.ndim == 3:
        gain = gain[None, ...]
    B, N, _, F = gain.shape
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    return gain[:, upper, :].mean(axis=(0, 1))


def _excited_subset(v: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Keep windows whose detrended leader RMS amplitude is >= floor (m/s).

    Mirrors the evaluator's excitation gate: flat-leader windows turn the
    transfer-gain ratio into a division artefact and would dominate the mean
    curves. Falls back to the full set if nothing survives.
    """

    keep = leader_excitation_amplitude(v, detrend_window_steps=8) >= floor
    if not np.any(keep):
        return v
    return v[keep]


def _plot_fft_gain(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    target_hz: float,
    band: tuple[float, float],
    out_path: Path,
    n_fft: int,
) -> None:
    # 8s history+prediction window with zero-padded FFT (the bare 3s window
    # holds a single in-band bin) and a log gain axis (DMG blows up to ~1e4).
    # Curves average only excitation-gated windows (leader RMS >= 0.05 m/s).
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    style_axes(ax)
    truth_gain, truth_freqs = fft_gain(
        _excited_subset(truth), target_hz=target_hz, band=band, detrend_window_steps=8, n_fft=n_fft
    )
    ax.plot(
        truth_freqs,
        _mean_pair_gain(truth_gain),
        color=color_for("ground truth"),
        linewidth=2.6,
        label="ground truth",
    )
    for i, (name, pred) in enumerate(predictions.items()):
        gain, freqs = fft_gain(
            _excited_subset(pred), target_hz=target_hz, band=band, detrend_window_steps=8, n_fft=n_fft
        )
        ax.plot(
            freqs,
            _mean_pair_gain(gain),
            color=color_for(name, i),
            linewidth=2.8 if "SSP" in name else 2.1,
            label=name,
            alpha=0.95,
        )
    ax.axhline(y=1.0, color="#C0392B", linestyle="--", linewidth=1.6, label=r"$G=1$")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Mean transfer gain $G(f)$")
    ax.set_title("Frequency-domain transfer gain")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


# Learning models with a defined string-stability statistic. Constant-leader
# physics/DMG baselines have an undefined unstable-window ratio (0 retained
# windows) and are intentionally excluded from the Pareto (carried by a paper
# footnote) instead of plotted at a spurious 0.
_PARETO_MODELS = [
    ("SSP-DMGTimeNet (ours)", "ssp_dmgtimenet_v6"),
    ("Int-LSTM", "interaction_lstm"),
    ("Transformer", "platoon_transformer"),
    ("Full-graph Attn", "full_graph_attention"),
    ("LSTM", "platoon_lstm"),
    ("CNN-Int-LSTM-IDM", "cnn_int_lstm_idm"),
]


def _parse_report_kv(path: Path) -> dict[str, dict[str, float]]:
    """Parse a ``test_report.md`` into ``{section: {metric: value}}``."""

    section: str | None = None
    out: dict[str, dict[str, float]] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip()
            if line.startswith("#"):
                section = line.lstrip("#").strip()
                out.setdefault(section, {})
            elif line.startswith("|") and section:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) == 2:
                    try:
                        out[section][cells[0]] = float(cells[1])
                    except ValueError:
                        pass
    return out


def _plot_pareto(reports_dir: Path, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    style_axes(ax)
    plotted = 0
    for i, (name, dirn) in enumerate(_PARETO_MODELS):
        report = reports_dir / dirn / "test_report.md"
        if not report.exists():
            continue
        kv = _parse_report_kv(report)
        mae_v = kv.get("Accuracy (per variable)", {}).get("v")
        unstable = kv.get("Stability", {}).get("unstable_window_ratio")
        retained = kv.get("Stability", {}).get("excitation_n_retained", 0.0)
        if mae_v is None or unstable is None or retained <= 0:
            continue
        marker = "*" if "SSP" in name else "o"
        size = 420 if "SSP" in name else 130
        ax.scatter(
            mae_v,
            unstable * 100.0,
            s=size,
            marker=marker,
            color=color_for(name, i),
            edgecolors="white",
            linewidths=0.8,
            label=name,
            zorder=3,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel(r"Test $v$-MAE (m/s) $\rightarrow$ worse accuracy")
    ax.set_ylabel(r"Unstable window ratio (%) $\rightarrow$ worse stability")
    # symlog separates SSP's sub-1% value from the 85-95% baseline cluster that
    # a linear axis would flatten together.
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Stability–accuracy Pareto")
    ax.legend(loc="center right")
    save_figure(fig, out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scheme-C interpretability figures")
    parser.add_argument("--model", type=str, action="append", required=True,
                        help="Format: 'name:config_path:checkpoint_path'. Repeat for multiple models.")
    parser.add_argument("--reference-config", type=Path, required=True,
                        help="Config used to load the dataset and feature normalisation.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.plot")
    apply_paper_style(15.0)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _parse_models(args.model)
    if not artifacts:
        raise ValueError("Provide at least one --model entry")

    cfg_ref = load_config(args.reference_config)
    paths_section = cfg_ref.get("paths", {})
    train_path = Path(paths_section["train"])
    val_path = Path(paths_section["val"])
    test_path = Path(paths_section["test"]) if paths_section.get("test") else None
    loaders, normalisation, _ = build_platoon_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=int(cfg_ref.get("trainer", {}).get("batch_size", 64)),
        num_workers=args.num_workers,
        return_raw=True,
    )
    if args.split not in loaders:
        raise KeyError(f"Split {args.split!r} not in loaders {list(loaders)}")
    loader = loaders[args.split]
    output_channels = cfg_ref.get("loss", {}).get("prediction", {}).get("variables") or [
        "v", "s", "a", "x_rel_leader",
    ]

    device = torch.device(args.device)
    target_hz = float(cfg_ref.get("data", {}).get("target_hz", 10.0))
    fft_band = tuple(
        float(x)
        for x in cfg_ref.get("loss", {}).get("stability", {}).get("fft_band_hz", (0.05, 0.5))
    )

    evaluator_cfg = EvaluatorConfig(
        target_hz=target_hz,
        output_channels=tuple(output_channels),
        fft_band_hz=fft_band,
    )
    predictions_v: dict[str, np.ndarray] = {}
    predictions_v_long: dict[str, np.ndarray] = {}
    targets_v: np.ndarray | None = None
    targets_v_long: np.ndarray | None = None

    for art in artifacts:
        log.info("Processing %s", art.name)
        model, _, _ = art.load(device)
        output_var_indices = [normalisation.feature_names.index(name) for name in output_channels]
        output_mean = torch.as_tensor(normalisation.mean[output_var_indices], dtype=torch.float32)
        output_std = torch.as_tensor(normalisation.std[output_var_indices], dtype=torch.float32)
        input_mean = torch.as_tensor(normalisation.mean, dtype=torch.float32)
        input_std = torch.as_tensor(normalisation.std, dtype=torch.float32)
        initialise_normalisation_for_model(model, input_mean, input_std, output_mean, output_std)
        gathered = _gather_predictions(model, loader, output_channels, device)
        v_index = output_channels.index("v")
        pred_v = gathered["pred"][..., v_index]
        target_v = gathered["target"][..., v_index]
        history_v = gathered["history_v"]
        if args.max_samples is not None:
            pred_v = pred_v[: args.max_samples]
            target_v = target_v[: args.max_samples]
            history_v = history_v[: args.max_samples]
        predictions_v[art.name] = pred_v
        # 8s window (history + prediction) for spectral/lag figures.
        predictions_v_long[art.name] = np.concatenate([history_v, pred_v], axis=1)
        if targets_v is None:
            targets_v = target_v
            targets_v_long = np.concatenate([history_v, target_v], axis=1)
        if hasattr(model, "sp_daca_blocks"):
            _plot_tau(model, out_dir / f"tau_{art.name}.png")
        _plot_subplatoon_heatmap(pred_v, art.name, out_dir / f"sub_amp_{art.name}.png")

    if targets_v is None or targets_v_long is None:
        raise RuntimeError("No targets gathered")

    _plot_phase_delay_scatter(
        targets_v_long,
        predictions_v_long,
        targets_v,
        target_hz,
        out_dir / "phase_delay_scatter.png",
    )
    _plot_velocity_heatmap(targets_v, predictions_v, target_hz, out_dir / "velocity_heatmap.png")
    _plot_adjacent_amp_box(predictions_v, target_hz, out_dir / "adjacent_amp_box.png")
    _plot_fft_gain(
        targets_v_long,
        predictions_v_long,
        target_hz,
        fft_band,
        out_dir / "fft_gain.png",
        n_fft=evaluator_cfg.fft_n_fft,
    )
    _plot_pareto(out_dir.parent / "reports", out_dir / "pareto.png")
    log.info("Wrote figures to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
