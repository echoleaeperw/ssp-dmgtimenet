"""Aggregate evaluation utilities producing the scheme-C metric tables."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..losses.total import resolve_target_indices
from ..metrics.accuracy import (
    horizon_wise_errors,
    mae_per_variable,
    rmse_per_variable,
    tail_vehicle_error,
    vehicle_wise_errors,
)
from ..metrics.safety import (
    collision_risk,
    comfort_jerk,
    energy_consumption,
    gap_violation_rate,
)
from ..metrics.stability import (
    adjacent_amplification,
    aggregate_amplification_distribution,
    conditional_internal_amplification_stats,
    disturbance_detection_stats,
    fft_band_gain_stats,
    gt_referenced_amplification_stats,
    gt_referenced_fft_band_gain_stats,
    phase_delay,
    strict_joint_stability_metrics,
    subplatoon_amplification,
    unstable_window_metrics,
)


@dataclass(slots=True, frozen=True)
class EvaluatorConfig:
    target_hz: float = 10.0
    detrend_window_steps: int = 8
    horizons_seconds: tuple[float, ...] = (1.0, 2.0, 3.0)
    fft_band_hz: tuple[float, float] = (0.05, 0.5)
    # FFT and phase-delay metrics run on the concatenated history+prediction
    # velocity (8s @ 10Hz instead of the bare 3s prediction window): the 3s
    # window leaves a single FFT bin inside the band and lets the
    # cross-correlation peak hit the ±2.9s boundary lags. The zero-padded
    # n_fft refines the frequency grid; the lag cap keeps the delay search
    # inside the physically plausible range (tau_max = 2.5s).
    fft_n_fft: int = 256
    phase_max_lag_seconds: float = 4.0
    delta_unstable: float = 0.0
    # Amplification statistics are only meaningful when the upstream signal
    # carries excitation: windows whose detrended leader RMS amplitude falls
    # below this floor (m/s) are excluded from the unstable-window statistics
    # (constant-leader rollouts otherwise inject x/eps division artefacts).
    excitation_floor: float = 0.05
    output_channels: tuple[str, ...] = ("v", "s", "a", "x_rel_leader")


@dataclass(slots=True)
class EvaluationReport:
    accuracy: dict[str, float]
    accuracy_by_horizon: dict[str, dict[str, float]]
    accuracy_by_vehicle: dict[str, dict[str, float]]
    tail_accuracy: dict[str, float]
    stability: dict[str, float]
    stability_distribution: dict[str, np.ndarray] = field(default_factory=dict)
    safety: dict[str, float] = field(default_factory=dict)
    comfort: dict[str, float] = field(default_factory=dict)
    energy: dict[str, float] = field(default_factory=dict)
    gap: dict[str, float] = field(default_factory=dict)
    phase_delay_seconds: np.ndarray | None = None


class Evaluator:
    """Run a model over a loader and assemble the full metric report."""

    def __init__(self, config: EvaluatorConfig) -> None:
        self.config = config

    def _gather(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> dict[str, np.ndarray]:
        model.eval()
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        history_vs: list[np.ndarray] = []
        hist_v_idx = resolve_target_indices(("v",))[0]
        with torch.no_grad():
            for batch in loader:
                history_raw = batch["history_raw"].to(device, non_blocking=True)
                history_mask = batch["history_mask"].to(device, non_blocking=True)
                future_raw = batch["future_raw"].to(device, non_blocking=True)
                future_mask = batch["future_mask"].to(device, non_blocking=True)
                model_out = model(history_raw, history_mask)
                pred = model_out["predictions"].detach().cpu().numpy()
                idx = list(resolve_target_indices(self.config.output_channels))
                target = future_raw[..., idx].detach().cpu().numpy()
                mask = future_mask[..., idx].detach().cpu().numpy()
                preds.append(pred)
                targets.append(target)
                masks.append(mask)
                history_vs.append(history_raw[..., hist_v_idx].detach().cpu().numpy())
        if not preds:
            raise RuntimeError("No samples produced during evaluation")
        return {
            "pred": np.concatenate(preds, axis=0),
            "target": np.concatenate(targets, axis=0),
            "mask": np.concatenate(masks, axis=0),
            "history_v": np.concatenate(history_vs, axis=0),
        }

    def report(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device | None = None,
    ) -> EvaluationReport:
        if device is None:
            device = next(model.parameters()).device
        cache = self._gather(model, loader, device)
        pred = cache["pred"]
        target = cache["target"]
        mask = cache["mask"]
        var_idx = {name: i for i, name in enumerate(self.config.output_channels)}
        accuracy = mae_per_variable(pred, target, mask, var_idx) | {
            f"rmse_{k}": v for k, v in rmse_per_variable(pred, target, mask, var_idx).items()
        }
        by_horizon = horizon_wise_errors(
            pred,
            target,
            target_hz=self.config.target_hz,
            horizons_seconds=self.config.horizons_seconds,
            mask=mask,
            variable_index=var_idx,
        )
        by_vehicle = vehicle_wise_errors(pred, target, mask, var_idx)
        tail = tail_vehicle_error(pred, target, mask, var_idx)

        v = pred[..., var_idx["v"]]  # (B, T, N)
        v_gt = target[..., var_idx["v"]]
        s_full = pred[..., var_idx["s"]]
        a = pred[..., var_idx["a"]]
        s_follow = s_full[..., 1:]
        # 8s long window (history + prediction) for the spectral/lag metrics.
        v_long = np.concatenate([cache["history_v"], v], axis=1)
        v_gt_long = np.concatenate([cache["history_v"], v_gt], axis=1)
        adj_amp = adjacent_amplification(v, self.config.detrend_window_steps)
        sub_amp = subplatoon_amplification(v, self.config.detrend_window_steps)
        fft_stats = fft_band_gain_stats(
            v_long,
            target_hz=self.config.target_hz,
            band=self.config.fft_band_hz,
            detrend_window_steps=self.config.detrend_window_steps,
            n_fft=self.config.fft_n_fft,
            excitation_floor=self.config.excitation_floor,
        )
        strict_report = strict_joint_stability_metrics(
            v,
            detrend_window_steps=self.config.detrend_window_steps,
            target_hz=self.config.target_hz,
            band=self.config.fft_band_hz,
            delta=self.config.delta_unstable,
            excitation_floor=self.config.excitation_floor,
            v_frequency=v_long,
            n_fft=self.config.fft_n_fft,
        )
        strict_gt_subset_report = strict_joint_stability_metrics(
            v,
            detrend_window_steps=self.config.detrend_window_steps,
            target_hz=self.config.target_hz,
            band=self.config.fft_band_hz,
            delta=self.config.delta_unstable,
            excitation_floor=self.config.excitation_floor,
            floor_reference_v=v_gt,
            v_frequency=v_long,
            n_fft=self.config.fft_n_fft,
        )
        gt_ref_stats = gt_referenced_amplification_stats(
            v,
            v_gt,
            detrend_window_steps=self.config.detrend_window_steps,
            delta=self.config.delta_unstable,
            excitation_floor=self.config.excitation_floor,
        )
        gt_ref_fft_stats = gt_referenced_fft_band_gain_stats(
            v_long,
            v_gt_long,
            excitation_reference_v=v_gt,
            target_hz=self.config.target_hz,
            band=self.config.fft_band_hz,
            detrend_window_steps=self.config.detrend_window_steps,
            n_fft=self.config.fft_n_fft,
            excitation_floor=self.config.excitation_floor,
        )
        detection_stats = disturbance_detection_stats(
            v,
            v_gt,
            detrend_window_steps=self.config.detrend_window_steps,
            excitation_floor=self.config.excitation_floor,
        )
        conditional_internal_stats = conditional_internal_amplification_stats(
            v,
            v_gt,
            detrend_window_steps=self.config.detrend_window_steps,
            delta=self.config.delta_unstable,
            excitation_floor=self.config.excitation_floor,
        )
        win_report = unstable_window_metrics(
            v,
            self.config.detrend_window_steps,
            delta=self.config.delta_unstable,
            excitation_floor=self.config.excitation_floor,
        )
        # Unified GT-subset protocol: restrict the support to windows whose
        # *ground-truth* leader carries excitation (intersected with the
        # per-model floor that keeps the ratio denominator well-conditioned),
        # so every model row is compared on the same excitation events. The
        # gt_self report doubles as the GT reference row of the main table and
        # its n_retained equals the GT-excited window count of the split.
        if self.config.excitation_floor > 0.0:
            gt_subset_report = unstable_window_metrics(
                v,
                self.config.detrend_window_steps,
                delta=self.config.delta_unstable,
                excitation_floor=self.config.excitation_floor,
                floor_reference_v=v_gt,
            )
            gt_self_report = unstable_window_metrics(
                v_gt,
                self.config.detrend_window_steps,
                delta=self.config.delta_unstable,
                excitation_floor=self.config.excitation_floor,
            )
        else:
            gt_subset_report = None
            gt_self_report = None
        adj_distribution = aggregate_amplification_distribution(adj_amp)
        delay = phase_delay(
            v_long,
            target_hz=self.config.target_hz,
            detrend_window_steps=self.config.detrend_window_steps,
            max_lag_seconds=self.config.phase_max_lag_seconds,
        )
        stability = {
            "unstable_window_ratio": float(win_report.unstable_window_ratio),
            "time_adjacent_unstable_window_ratio": float(win_report.unstable_window_ratio),
            "exceedance_area": float(win_report.exceedance_area),
            "max_amplification": float(win_report.max_amplification),
            "excitation_retained_ratio": float(win_report.excitation_retained_ratio),
            "excitation_n_retained": int(win_report.excitation_n_retained),
            "strict_time_unstable_window_ratio": float(
                strict_report.time_unstable_window_ratio
            ),
            "strict_frequency_unstable_window_ratio": float(
                strict_report.frequency_unstable_window_ratio
            ),
            "strict_joint_unstable_window_ratio": float(
                strict_report.joint_unstable_window_ratio
            ),
            "strict_max_time_amplification": float(strict_report.max_time_amplification),
            "strict_max_frequency_gain": float(strict_report.max_frequency_gain),
            "strict_excitation_retained_ratio": float(
                strict_report.excitation_retained_ratio
            ),
            "strict_excitation_n_retained": int(strict_report.excitation_n_retained),
            "strict_fft_n_bins": int(strict_report.fft_n_bins),
            "strict_gt_subset_time_unstable_window_ratio": float(
                strict_gt_subset_report.time_unstable_window_ratio
            ),
            "strict_gt_subset_frequency_unstable_window_ratio": float(
                strict_gt_subset_report.frequency_unstable_window_ratio
            ),
            "strict_gt_subset_joint_unstable_window_ratio": float(
                strict_gt_subset_report.joint_unstable_window_ratio
            ),
            "strict_gt_subset_max_time_amplification": float(
                strict_gt_subset_report.max_time_amplification
            ),
            "strict_gt_subset_max_frequency_gain": float(
                strict_gt_subset_report.max_frequency_gain
            ),
            "strict_gt_subset_n_retained": int(
                strict_gt_subset_report.excitation_n_retained
            ),
            **fft_stats,
            **{f"detection_{k}": float(value) for k, value in detection_stats.items()},
            **{f"gt_ref_{k}": float(value) for k, value in gt_ref_stats.items()},
            **{f"gt_ref_fft_{k}": float(value) for k, value in gt_ref_fft_stats.items()},
            **{
                f"conditional_internal_{k}": float(value)
                for k, value in conditional_internal_stats.items()
            },
            **{f"adj_{k}": float(np.mean(v_)) for k, v_ in win_report.pair_unstable_ratio.items()},
        }
        if gt_subset_report is not None and gt_self_report is not None:
            stability.update(
                {
                    "gt_subset_unstable_window_ratio": float(gt_subset_report.unstable_window_ratio),
                    "gt_subset_exceedance_area": float(gt_subset_report.exceedance_area),
                    "gt_subset_max_amplification": float(gt_subset_report.max_amplification),
                    "gt_subset_retained_ratio": float(gt_subset_report.excitation_retained_ratio),
                    "gt_subset_n_retained": int(gt_subset_report.excitation_n_retained),
                    "gt_excited_n_windows": int(gt_self_report.excitation_n_retained),
                    "gt_self_unstable_window_ratio": float(gt_self_report.unstable_window_ratio),
                    "gt_self_exceedance_area": float(gt_self_report.exceedance_area),
                    "gt_self_max_amplification": float(gt_self_report.max_amplification),
                    **{
                        f"gt_subset_adj_{k}": float(val)
                        for k, val in gt_subset_report.pair_unstable_ratio.items()
                    },
                }
            )
        stability_distribution = {
            "adj_distribution": adj_distribution,
            "subplatoon_mean": np.asarray(sub_amp.mean(axis=0)),
        }

        safety = collision_risk(s_follow, v, target_hz=self.config.target_hz)
        comfort = comfort_jerk(a, target_hz=self.config.target_hz)
        energy = energy_consumption(v, a, target_hz=self.config.target_hz)
        gap = gap_violation_rate(s_follow, v, target_hz=self.config.target_hz)
        return EvaluationReport(
            accuracy=accuracy,
            accuracy_by_horizon=by_horizon,
            accuracy_by_vehicle=by_vehicle,
            tail_accuracy=tail,
            stability=stability,
            stability_distribution=stability_distribution,
            safety=safety,
            comfort=comfort,
            energy=energy,
            gap=gap,
            phase_delay_seconds=np.asarray(delay),
        )


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    config: EvaluatorConfig,
    device: torch.device | None = None,
) -> EvaluationReport:
    return Evaluator(config).report(model, loader, device)


def report_to_flat_dict(report: EvaluationReport) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in report.accuracy.items():
        flat[f"acc/{k}"] = float(v)
    for h, sub in report.accuracy_by_horizon.items():
        for k, v in sub.items():
            flat[f"acc/{h}/{k}"] = float(v)
    for veh, sub in report.accuracy_by_vehicle.items():
        for k, v in sub.items():
            flat[f"acc/{veh}/{k}"] = float(v)
    for k, v in report.tail_accuracy.items():
        flat[f"tail/{k}"] = float(v)
    for k, v in report.stability.items():
        flat[f"stab/{k}"] = float(v)
    for k, v in report.safety.items():
        flat[f"safe/{k}"] = float(v)
    for k, v in report.comfort.items():
        flat[f"comfort/{k}"] = float(v)
    for k, v in report.energy.items():
        flat[f"energy/{k}"] = float(v)
    for k, v in report.gap.items():
        flat[f"gap/{k}"] = float(v)
    return flat


def write_report_markdown(report: EvaluationReport, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Evaluation Report", ""]
    lines.append("## Accuracy (per variable)")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in sorted(report.accuracy):
        lines.append(f"| {k} | {report.accuracy[k]:.6f} |")
    lines.append("")
    lines.append("## Accuracy (per horizon)")
    lines.append("| horizon | metric | value |")
    lines.append("| --- | --- | --- |")
    for horizon, sub in report.accuracy_by_horizon.items():
        for k in sorted(sub):
            lines.append(f"| {horizon} | {k} | {sub[k]:.6f} |")
    lines.append("")
    lines.append("## Accuracy (per vehicle)")
    lines.append("| vehicle | metric | value |")
    lines.append("| --- | --- | --- |")
    for veh, sub in report.accuracy_by_vehicle.items():
        for k in sorted(sub):
            lines.append(f"| {veh} | {k} | {sub[k]:.6f} |")
    lines.append("")
    lines.append("## Tail vehicle accuracy")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in sorted(report.tail_accuracy):
        lines.append(f"| {k} | {report.tail_accuracy[k]:.6f} |")
    lines.append("")
    lines.append("## Stability")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for k in sorted(report.stability):
        lines.append(f"| {k} | {report.stability[k]:.6f} |")
    lines.append("")
    lines.append("## Safety / comfort / energy / gap")
    for section_name, section in [
        ("safety", report.safety),
        ("comfort", report.comfort),
        ("energy", report.energy),
        ("gap", report.gap),
    ]:
        lines.append(f"### {section_name}")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for k in sorted(section):
            lines.append(f"| {k} | {section[k]:.6f} |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def merge_evaluator_with_loss_diagnostics(
    flat_report: dict[str, float],
    diag: Iterable[tuple[str, float]],
) -> dict[str, float]:
    out = dict(flat_report)
    for k, v in diag:
        out[f"diag/{k}"] = float(v)
    return out
