"""Qualitative platoon trajectory case studies.

Default figure is a *velocity-only* hist|pred panel (cleaner for papers).
Windows are chosen from a strong GT-leader excitation pool with filters that
prefer: visible leader Δv, GT/baseline amplification, low ours amplification,
low ours v-MAE, and small hist→pred boundary jump.

Example
-------
python -m ssp_dmgtimenet.scripts.plot_trajectory_cases \\
  --reference-config configs/ssp_dmgtimenet_v6.yaml \\
  --model "SSP-DMGTimeNet:configs/ssp_dmgtimenet_v6.yaml:../artifacts/checkpoints/ssp_dmgtimenet_v6/best.pt" \\
  --model "Int-LSTM:configs/baseline_int_lstm.yaml:../artifacts/checkpoints/interaction_lstm/best.pt" \\
  --model "Transformer:configs/baseline_transformer.yaml:../artifacts/checkpoints/platoon_transformer/best.pt" \\
  --ours-name SSP-DMGTimeNet --contrast-name Int-LSTM \\
  --out-dir ../artifacts/figures/trajectory_cases
"""

from __future__ import annotations

import argparse
import json
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
from ..data.windowing import FEATURE_NAMES
from ..losses.total import resolve_target_indices
from ..metrics.stability import leader_excitation_amplitude, subplatoon_amplification
from ..training.factory import build_model, initialise_normalisation_for_model
from ..utils.config import load_config
from ..utils.plot_style import apply_paper_style

CHANNEL_LABELS = {
    "v": r"$v$ (m/s)",
    "a": r"$a$ (m/s$^2$)",
    "s": r"$s$ (m)",
}
# Reference-paper layout labels (Fig. 11 style).
REF_CHANNEL_LABELS = {
    "v": r"$v_{ego}$ (m/s)",
    "a": r"$a_{ego}$ (m/s$^2$)",
    "s": r"DHW (m)",
}
PALETTE = {
    "SSP-DMGTimeNet": "#d62728",
    "SSP-DMGTimeNet (ours)": "#d62728",
    "Int-LSTM": "#1f77b4",
    "Transformer": "#2ca02c",
    "LSTM": "#9467bd",
    "Full-graph Attention": "#8c564b",
    "CNN-Int-LSTM-IDM": "#17becf",
}
MARKERS = {
    "GT": ("o", 4.0),
    "SSP-DMGTimeNet": ("s", 4.0),
    "SSP-DMGTimeNet (ours)": ("s", 4.0),
    "Int-LSTM": ("^", 4.0),
    "Transformer": ("v", 4.0),
    "LSTM": ("D", 3.5),
    "Full-graph Attention": ("*", 5.0),
    "CNN-Int-LSTM-IDM": ("P", 4.0),
}
_FALLBACK_MARKERS = ("o", "^", "v", "D", "s", "*", "P", "X")


@dataclass(slots=True)
class ModelArtifact:
    name: str
    config_path: Path
    checkpoint_path: Path

    def load(self, device: torch.device) -> torch.nn.Module:
        cfg = load_config(self.config_path)
        model = build_model(cfg["model"])
        state = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = state["state_dict"] if "state_dict" in state else state
        model.load_state_dict(state_dict)
        return model.to(device).eval()


def _parse_models(arg: list[str]) -> list[ModelArtifact]:
    out: list[ModelArtifact] = []
    for item in arg:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"--model expects 'name:config:checkpoint', got {item!r}")
        name, config_path, ckpt_path = parts
        out.append(ModelArtifact(name=name, config_path=Path(config_path), checkpoint_path=Path(ckpt_path)))
    return out


def _gather(
    model: torch.nn.Module,
    loader,
    output_channels: Sequence[str],
    device: torch.device,
) -> dict[str, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    histories: list[np.ndarray] = []
    out_idx = list(resolve_target_indices(tuple(output_channels)))
    with torch.no_grad():
        for batch in loader:
            history_raw = batch["history_raw"].to(device, non_blocking=True)
            history_mask = batch["history_mask"].to(device, non_blocking=True)
            future_raw = batch["future_raw"].to(device, non_blocking=True)
            output = model(history_raw, history_mask)
            preds.append(output["predictions"].detach().cpu().numpy())
            targets.append(future_raw[..., out_idx].detach().cpu().numpy())
            histories.append(history_raw.detach().cpu().numpy())
    if not preds:
        raise RuntimeError("No samples available for plotting")
    return {
        "pred": np.concatenate(preds, axis=0),
        "target": np.concatenate(targets, axis=0),
        "history": np.concatenate(histories, axis=0),
    }


def _channel_index(output_channels: Sequence[str], name: str) -> int:
    if name not in output_channels:
        raise KeyError(f"Channel {name!r} not in output variables {list(output_channels)}")
    return list(output_channels).index(name)


def _feat_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


def _max_subplatoon_amp(v_btn: np.ndarray, detrend_steps: int) -> np.ndarray:
    ratios = np.asarray(subplatoon_amplification(v_btn, detrend_steps))
    if ratios.ndim == 2:
        ratios = ratios[None, ...]
    B, N, _ = ratios.shape
    max_amp = np.ones(B, dtype=np.float64)
    for b in range(B):
        vals = [float(ratios[b, j, i]) for j in range(N) for i in range(j + 1, N)]
        max_amp[b] = max(vals) if vals else 1.0
    return max_amp


def _mae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-window MAE over time and vehicles. shapes (B,T,N)."""
    return np.mean(np.abs(pred - target), axis=(1, 2))


def _boundary_jump(history_v: np.ndarray, future_v: np.ndarray) -> np.ndarray:
    """Mean |v_pred[0]-v_hist[-1]| across vehicles. shapes (B,H,N)/(B,T,N)."""
    return np.mean(np.abs(future_v[:, 0, :] - history_v[:, -1, :]), axis=1)


def _select_cases(
    *,
    history_v: np.ndarray,
    target_v: np.ndarray,
    pred_v_by_model: dict[str, np.ndarray],
    ours_name: str,
    contrast_name: str | None,
    detrend_steps: int,
    excitation_floor: float,
    n_cases: int,
    manual_indices: list[int] | None,
    max_ours_mae: float,
    max_boundary_jump: float,
    min_gt_amp: float,
) -> list[dict]:
    B = target_v.shape[0]
    if manual_indices:
        return [{"index": int(i), "mode": "manual"} for i in manual_indices if 0 <= i < B]

    gt_exc = leader_excitation_amplitude(target_v, detrend_steps)
    gt_amp = _max_subplatoon_amp(target_v, detrend_steps)
    ours_v = pred_v_by_model[ours_name]
    ours_amp = _max_subplatoon_amp(ours_v, detrend_steps)
    ours_mae = _mae(ours_v, target_v)
    ours_jump = _boundary_jump(history_v, ours_v)
    contrast_amp = (
        _max_subplatoon_amp(pred_v_by_model[contrast_name], detrend_steps)
        if contrast_name and contrast_name in pred_v_by_model
        else None
    )
    leader_dv = target_v[:, -1, 0] - target_v[:, 0, 0]

    # Soft filters: HighD future-horizon leader excitation rarely exceeds ~0.1 m/s,
    # and SSP often has O(1) m/s hist→pred level jumps, so jump is a penalty not a hard cut.
    pool = np.where(
        (gt_exc >= excitation_floor)
        & (gt_amp >= min_gt_amp)
        & (ours_mae <= max_ours_mae)
        & (ours_amp <= 1.25)
    )[0]
    if pool.size == 0:
        pool = np.where((gt_exc >= excitation_floor) & (ours_amp <= 1.5))[0]
    if pool.size == 0:
        pool = np.where(gt_exc >= excitation_floor)[0]
    if pool.size == 0:
        raise RuntimeError(f"No GT excitation windows with floor={excitation_floor}")

    def _score(idx: int) -> float:
        # Prefer visible |Δv|, strong GT amp, damped ours, low MAE; jump is soft.
        s = 8.0 * float(gt_exc[idx])
        s += 1.5 * float(abs(leader_dv[idx]))
        s += 0.6 * float(gt_amp[idx])
        s += 4.0 * max(0.0, 1.0 - float(ours_amp[idx]))
        s -= 3.0 * float(ours_mae[idx])
        s -= 0.4 * float(ours_jump[idx])
        if max_boundary_jump < float("inf") and ours_jump[idx] > max_boundary_jump:
            s -= 2.0 * float(ours_jump[idx] - max_boundary_jump)
        if contrast_amp is not None:
            s += 1.2 * max(0.0, float(contrast_amp[idx]) - 1.0)
        return s

    accel_pool = [int(i) for i in pool if leader_dv[i] >= 1.0]
    decel_pool = [int(i) for i in pool if leader_dv[i] <= -1.0]
    rest_pool = [int(i) for i in pool]

    selected: list[dict] = []
    used: set[int] = set()

    def _take(candidates: list[int], mode: str) -> None:
        if len(selected) >= n_cases or not candidates:
            return
        for idx in sorted(candidates, key=_score, reverse=True):
            if idx in used:
                continue
            selected.append(
                {
                    "index": idx,
                    "mode": mode,
                    "gt_exc": float(gt_exc[idx]),
                    "gt_max_amp": float(gt_amp[idx]),
                    "ours_max_amp": float(ours_amp[idx]),
                    "ours_v_mae": float(ours_mae[idx]),
                    "ours_boundary_jump": float(ours_jump[idx]),
                    "contrast_max_amp": (
                        float(contrast_amp[idx]) if contrast_amp is not None else None
                    ),
                    "leader_dv": float(leader_dv[idx]),
                    "score": float(_score(idx)),
                }
            )
            used.add(idx)
            break

    _take(accel_pool, "accel")
    _take(decel_pool, "decel")
    while len(selected) < n_cases:
        before = len(selected)
        _take(rest_pool, "mixed")
        if len(selected) == before:
            break
    if not selected:
        raise RuntimeError("Case selection failed")
    return selected


def _series_style(name: str, index: int = 0) -> tuple[str, str, float]:
    """Return (color, marker, markersize) for a series name."""
    color = "#111111" if name == "GT" else PALETTE.get(name, "#1f77b4")
    if name in MARKERS:
        marker, ms = MARKERS[name]
    else:
        marker = _FALLBACK_MARKERS[index % len(_FALLBACK_MARKERS)]
        ms = 4.0
    return color, marker, ms


def _plot_case_curves(
    *,
    case: dict,
    history: np.ndarray,
    target: np.ndarray,
    preds: dict[str, np.ndarray],
    model_order: list[str],
    output_channels: Sequence[str],
    channels: Sequence[str],
    target_hz: float,
    out_path: Path,
    vehicles: Sequence[int] | None,
    relative_to_hist_end: bool,
    layout: str = "default",
) -> None:
    idx = int(case["index"])
    hist = history[idx]
    tgt = target[idx]
    H, N, _ = hist.shape
    T = tgt.shape[0]
    cols = list(vehicles) if vehicles is not None else list(range(N))
    use_ref = layout.strip().lower() in {"ref", "reference", "paper"}

    if use_ref:
        # Match reference Fig. 11: discrete Time Steps, absolute levels.
        t_hist = np.arange(1, H + 1, dtype=np.float64)
        t_fut = np.arange(H + 1, H + T + 1, dtype=np.float64)
        t_split = float(H) + 0.5
        relative_to_hist_end = False
        channel_labels = REF_CHANNEL_LABELS
        markevery_hist = max(1, H // 10)
        markevery_fut = max(1, T // 8)
    else:
        t_hist = np.arange(H) / float(target_hz)
        t_fut = (H + np.arange(T)) / float(target_hz)
        t_split = H / float(target_hz)
        channel_labels = CHANNEL_LABELS
        markevery_hist = max(1, H // 12)
        markevery_fut = max(1, T // 10)

    n_r, n_c = len(channels), len(cols)
    fig_w = (2.45 * n_c + 1.0) if use_ref else (2.35 * n_c + 0.8)
    fig_h = (2.25 * n_r + 1.4) if use_ref else (2.15 * n_r + 0.6)
    fig, axes = plt.subplots(
        n_r,
        n_c,
        figsize=(fig_w, fig_h),
        sharex=True,
        squeeze=False,
    )

    legend_handles: dict[str, object] = {}

    for r, ch in enumerate(channels):
        f_idx = _feat_index(ch)
        c_idx = _channel_index(output_channels, ch)
        for j, veh in enumerate(cols):
            ax = axes[r, j]
            h = hist[:, veh, f_idx]
            g = tgt[:, veh, c_idx]
            if relative_to_hist_end:
                # Align all future curves to the last observed state so the
                # figure shows response shape / propagation, not absolute bias.
                anchor = float(h[-1])
                h_plot = h - anchor
                g_plot = g - anchor
                ylab = r"$\Delta$" + CHANNEL_LABELS.get(ch, ch)
            else:
                h_plot, g_plot = h, g
                ylab = channel_labels.get(ch, CHANNEL_LABELS.get(ch, ch))

            gt_color, gt_marker, gt_ms = _series_style("GT")
            gt_y = np.concatenate([h_plot, g_plot], axis=0)
            gt_x = np.concatenate([t_hist, t_fut], axis=0)
            (h_gt,) = ax.plot(
                gt_x,
                gt_y,
                color=gt_color,
                lw=1.8,
                marker=gt_marker,
                ms=gt_ms,
                markevery=markevery_hist,
                label="Ground Truth",
            )
            legend_handles.setdefault("Ground Truth", h_gt)

            for m_i, name in enumerate(model_order):
                p = preds[name][idx, :, veh, c_idx]
                p_plot = p - float(h[-1]) if relative_to_hist_end else p
                color, marker, ms = _series_style(name, m_i)
                (h_line,) = ax.plot(
                    t_fut,
                    p_plot,
                    color=color,
                    lw=1.45,
                    marker=marker,
                    ms=ms,
                    markevery=markevery_fut,
                    label=name,
                )
                legend_handles.setdefault(name, h_line)

            ax.axvline(t_split, color="0.45", ls="--", lw=1.0)
            if relative_to_hist_end:
                ax.axhline(0.0, color="0.7", lw=0.8)

            if use_ref:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.9)
                ax.grid(True, ls="--", color="#BDBDBD", lw=0.7, alpha=0.85)
                ax.set_axisbelow(True)
                if j == 0:
                    ax.set_ylabel(ylab, fontsize=12)
                else:
                    ax.set_ylabel("")
                if r == n_r - 1:
                    ax.set_xlabel("Time Steps", fontsize=11)
                    # Column tag under each bottom panel (reference layout).
                    ax.text(
                        0.5,
                        -0.34,
                        f"Following Car {j + 1}",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=12,
                    )
                ax.tick_params(labelsize=10)
            else:
                if r == 0:
                    ax.set_title("Leader ($C_1$)" if veh == 0 else f"$C_{{{veh + 1}}}$")
                if j == 0:
                    ax.set_ylabel(ylab)
                if r == n_r - 1:
                    ax.set_xlabel("Time (s)")
                ax.grid(True, alpha=0.25, lw=0.6)

    if use_ref:
        n_items = len(legend_handles)
        ncol = 4 if n_items >= 5 else min(4, n_items)
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="lower center",
            ncol=ncol,
            frameon=False,
            fontsize=11,
            markerscale=1.15,
            handlelength=2.2,
            columnspacing=1.4,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.subplots_adjust(left=0.07, right=0.995, top=0.98, bottom=0.16, wspace=0.22, hspace=0.22)
    else:
        fig.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )
        fig.suptitle(
            f"Case {idx} [{case.get('mode', 'case')}]  "
            f"GT exc={case.get('gt_exc', float('nan')):.3f}  "
            f"GT amp={case.get('gt_max_amp', float('nan')):.2f}  "
            f"ours amp={case.get('ours_max_amp', float('nan')):.2f}  "
            f"ours v-MAE={case.get('ours_v_mae', float('nan')):.3f}",
            y=1.02,
        )
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_case_heatmap(
    *,
    case: dict,
    target_v: np.ndarray,
    pred_v: dict[str, np.ndarray],
    model_order: list[str],
    target_hz: float,
    out_path: Path,
) -> None:
    """Side-by-side future velocity heatmaps: GT | models (paper-style)."""
    idx = int(case["index"])
    panels = ["GT", *model_order]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.2, 1.35 * len(panels) + 0.4), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    data = [target_v[idx]] + [pred_v[name][idx] for name in model_order]
    vmin = min(float(d.min()) for d in data)
    vmax = max(float(d.max()) for d in data)
    for ax, name, arr in zip(axes, panels, data, strict=True):
        extent = [0.0, arr.shape[0] / target_hz, arr.shape[1] - 0.5, -0.5]
        im = ax.imshow(arr.T, aspect="auto", extent=extent, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        ax.set_ylabel("vehicle")
        ax.set_yticks(range(arr.shape[1]))
        ax.set_yticklabels([f"$C_{{{i + 1}}}$" for i in range(arr.shape[1])])
        ax.set_title(name, loc="left")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1].set_xlabel("Future time (s)")
    fig.suptitle(
        f"Case {idx} [{case.get('mode', 'case')}] velocity field  "
        f"(GT amp={case.get('gt_max_amp', float('nan')):.2f}, "
        f"ours amp={case.get('ours_max_amp', float('nan')):.2f})",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--reference-config", type=Path, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-cases", type=int, default=2)
    p.add_argument("--case-index", type=int, action="append", default=None)
    p.add_argument("--ours-name", type=str, default="SSP-DMGTimeNet")
    p.add_argument("--contrast-name", type=str, default="Int-LSTM")
    p.add_argument("--excitation-floor", type=float, default=0.05,
                   help="Min GT leader detrended RMS (same floor as metric tables)")
    p.add_argument("--min-gt-amp", type=float, default=1.5)
    p.add_argument("--max-ours-mae", type=float, default=0.60)
    p.add_argument("--max-boundary-jump", type=float, default=1e9,
                   help="Soft penalty threshold; SSP often jumps ~1.5 m/s so default is off")
    p.add_argument("--detrend-window-steps", type=int, default=8)
    p.add_argument("--channels", type=str, default="v",
                   help="Comma-separated subset of v,a,s (default: v only)")
    p.add_argument("--vehicles", type=str, default="all",
                   help="'all' or comma-separated 0-based indices, e.g. 0,1,2,3,4")
    p.add_argument("--relative-to-hist-end", action=argparse.BooleanOptionalAction, default=True,
                   help="Plot Δ from last history sample (recommended; hides absolute level bias)")
    p.add_argument(
        "--layout",
        type=str,
        default="default",
        choices=("default", "ref", "reference", "paper"),
        help="default: current paper panels; ref/reference/paper: Fig.11-style boxed grid + bottom legend",
    )
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.plot_cases")
    apply_paper_style(15.0)

    artifacts = _parse_models(args.model)
    if args.ours_name not in {a.name for a in artifacts}:
        raise ValueError(f"--ours-name {args.ours_name!r} not among models")

    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    for ch in channels:
        if ch not in CHANNEL_LABELS:
            raise ValueError(f"Unsupported channel {ch!r}")

    cfg_ref = load_config(args.reference_config)
    paths_section = cfg_ref.get("paths", {})
    loaders, normalisation, _ = build_platoon_loaders(
        train_path=Path(paths_section["train"]),
        val_path=Path(paths_section["val"]),
        test_path=Path(paths_section["test"]) if paths_section.get("test") else None,
        batch_size=int(cfg_ref.get("trainer", {}).get("batch_size", 64)),
        num_workers=args.num_workers,
        return_raw=True,
    )
    loader = loaders[args.split]
    output_channels = list(
        cfg_ref.get("loss", {}).get("prediction", {}).get("variables")
        or ["v", "s", "a", "x_rel_leader"]
    )
    for ch in channels:
        if ch not in output_channels:
            raise KeyError(f"Need output channel {ch!r}")

    device = torch.device(args.device)
    target_hz = float(cfg_ref.get("data", {}).get("target_hz", 10.0))

    preds: dict[str, np.ndarray] = {}
    history: np.ndarray | None = None
    target: np.ndarray | None = None
    model_order = [a.name for a in artifacts]

    for art in artifacts:
        log.info("Inferring %s", art.name)
        model = art.load(device)
        out_idx = [normalisation.feature_names.index(name) for name in output_channels]
        initialise_normalisation_for_model(
            model,
            torch.as_tensor(normalisation.mean, dtype=torch.float32),
            torch.as_tensor(normalisation.std, dtype=torch.float32),
            torch.as_tensor(normalisation.mean[out_idx], dtype=torch.float32),
            torch.as_tensor(normalisation.std[out_idx], dtype=torch.float32),
        )
        gathered = _gather(model, loader, output_channels, device)
        pred, tgt, hist = gathered["pred"], gathered["target"], gathered["history"]
        if args.max_samples is not None:
            pred, tgt, hist = pred[: args.max_samples], tgt[: args.max_samples], hist[: args.max_samples]
        preds[art.name] = pred
        if target is None:
            target, history = tgt, hist

    assert target is not None and history is not None
    v_idx = _channel_index(output_channels, "v")
    target_v = target[..., v_idx]
    history_v = history[..., _feat_index("v")]
    pred_v = {name: arr[..., v_idx] for name, arr in preds.items()}

    cases = _select_cases(
        history_v=history_v,
        target_v=target_v,
        pred_v_by_model=pred_v,
        ours_name=args.ours_name,
        contrast_name=args.contrast_name,
        detrend_steps=args.detrend_window_steps,
        excitation_floor=args.excitation_floor,
        n_cases=args.n_cases,
        manual_indices=args.case_index,
        max_ours_mae=args.max_ours_mae,
        max_boundary_jump=args.max_boundary_jump,
        min_gt_amp=args.min_gt_amp,
    )
    if args.case_index:
        gt_exc = leader_excitation_amplitude(target_v, args.detrend_window_steps)
        gt_amp = _max_subplatoon_amp(target_v, args.detrend_window_steps)
        ours_amp = _max_subplatoon_amp(pred_v[args.ours_name], args.detrend_window_steps)
        ours_mae = _mae(pred_v[args.ours_name], target_v)
        ours_jump = _boundary_jump(history_v, pred_v[args.ours_name])
        for case in cases:
            i = case["index"]
            case.update(
                {
                    "gt_exc": float(gt_exc[i]),
                    "gt_max_amp": float(gt_amp[i]),
                    "ours_max_amp": float(ours_amp[i]),
                    "ours_v_mae": float(ours_mae[i]),
                    "ours_boundary_jump": float(ours_jump[i]),
                    "leader_dv": float(target_v[i, -1, 0] - target_v[i, 0, 0]),
                }
            )

    vehicles: list[int] | None
    if args.vehicles.strip().lower() == "all":
        vehicles = None
    else:
        vehicles = [int(x) for x in args.vehicles.split(",") if x.strip()]

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": args.split,
        "excitation_floor": args.excitation_floor,
        "min_gt_amp": args.min_gt_amp,
        "max_ours_mae": args.max_ours_mae,
        "max_boundary_jump": args.max_boundary_jump,
        "channels": list(channels),
        "layout": str(args.layout),
        "ours_name": args.ours_name,
        "contrast_name": args.contrast_name,
        "models": model_order,
        "cases": cases,
        "note": (
            "History is shared GT; model curves are drawn only on the future "
            "horizon to avoid fake continuity across the hist|pred cut. "
            "Auto-select uses a stricter excitation floor than the 0.05 metric floor."
        ),
    }
    (out_dir / "case_selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for k, case in enumerate(cases, start=1):
        curve_path = out_dir / f"trajectory_case{k}_{case.get('mode', 'case')}_idx{case['index']}.png"
        heat_path = out_dir / f"heatmap_case{k}_{case.get('mode', 'case')}_idx{case['index']}.png"
        log.info("Writing %s", curve_path)
        _plot_case_curves(
            case=case,
            history=history,
            target=target,
            preds=preds,
            model_order=model_order,
            output_channels=output_channels,
            channels=channels,
            target_hz=target_hz,
            out_path=curve_path,
            vehicles=vehicles,
            relative_to_hist_end=bool(args.relative_to_hist_end),
            layout=str(args.layout),
        )
        log.info("Writing %s", heat_path)
        _plot_case_heatmap(
            case=case,
            target_v=target_v,
            pred_v=pred_v,
            model_order=model_order,
            target_hz=target_hz,
            out_path=heat_path,
        )

    log.info("Wrote %d case(s) to %s", len(cases), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
