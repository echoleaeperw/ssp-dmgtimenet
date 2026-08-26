"""Safety, comfort and computational performance metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch


def collision_risk(
    s_pred: torch.Tensor | np.ndarray,
    v_pred: torch.Tensor | np.ndarray,
    target_hz: float,
    ttc_threshold: float = 1.5,
    min_gap: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Two-pronged collision risk: gap violation and small-TTC fraction.

    ``s_pred`` is the predicted gap-to-predecessor (m), shape ``(B, T, N-1)``
    where vehicle index 0 corresponds to ``C2`` (followers only).
    ``v_pred`` is the predicted speed (m/s) for *all* vehicles, shape
    ``(B, T, N)``.
    """

    if isinstance(s_pred, torch.Tensor):
        s = s_pred.detach().cpu().numpy()
    else:
        s = np.asarray(s_pred)
    if isinstance(v_pred, torch.Tensor):
        v = v_pred.detach().cpu().numpy()
    else:
        v = np.asarray(v_pred)
    if s.ndim != 3 or v.ndim != 3:
        raise ValueError("collision_risk expects 3D tensors")
    if s.shape[2] != v.shape[2] - 1:
        raise ValueError("s_pred should have N-1 vehicles, v_pred N vehicles")

    relative_v = v[:, :, :-1] - v[:, :, 1:]  # follower closes gap when v_lead < v_follow
    closing_speed = -relative_v
    safe_close = np.where(closing_speed > eps, closing_speed, np.nan)
    ttc = s / safe_close
    ttc = np.where(np.isnan(ttc), np.inf, ttc)

    return {
        "gap_violation_ratio": float((s < min_gap).mean()),
        "low_ttc_ratio": float(((ttc > 0) & (ttc < ttc_threshold)).mean()),
        "min_ttc_p05": float(np.percentile(ttc[np.isfinite(ttc)], 5)) if np.isfinite(ttc).any() else float("inf"),
        "max_closing_speed": float(closing_speed.max()),
    }


def comfort_jerk(
    a_pred: torch.Tensor | np.ndarray,
    target_hz: float,
) -> dict[str, float]:
    """Aggregate jerk-based comfort metrics."""

    if isinstance(a_pred, torch.Tensor):
        a = a_pred.detach().cpu().numpy()
    else:
        a = np.asarray(a_pred)
    if a.ndim != 3:
        raise ValueError("comfort_jerk expects shape (B, T, N)")
    jerk = np.diff(a, axis=1) * float(target_hz)
    abs_jerk = np.abs(jerk)
    return {
        "rms_jerk": float(np.sqrt((jerk ** 2).mean())),
        "mean_abs_jerk": float(abs_jerk.mean()),
        "max_abs_jerk": float(abs_jerk.max()),
        "p95_abs_jerk": float(np.percentile(abs_jerk, 95)),
    }


def gap_violation_rate(
    s_pred: torch.Tensor | np.ndarray,
    v_pred: torch.Tensor | np.ndarray,
    target_hz: float,
    minimum_time_headway: float = 0.6,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Fraction of frames where predicted gap or time-headway violates safety."""

    if isinstance(s_pred, torch.Tensor):
        s = s_pred.detach().cpu().numpy()
    else:
        s = np.asarray(s_pred)
    if isinstance(v_pred, torch.Tensor):
        v = v_pred.detach().cpu().numpy()
    else:
        v = np.asarray(v_pred)
    if s.shape[2] != v.shape[2] - 1:
        raise ValueError("s_pred should have N-1 vehicles, v_pred N vehicles")
    follower_speed = np.maximum(v[:, :, 1:], eps)
    th = s / follower_speed
    return {
        "negative_gap_ratio": float((s < 0).mean()),
        "small_th_ratio": float((th < minimum_time_headway).mean()),
        "mean_time_headway": float(th.mean()),
    }


def energy_consumption(
    v_pred: torch.Tensor | np.ndarray,
    a_pred: torch.Tensor | np.ndarray,
    target_hz: float,
    rolling_resistance: float = 0.012,
    drag_coefficient: float = 0.3,
    frontal_area: float = 2.2,
    air_density: float = 1.225,
    mass_kg: float = 1500.0,
    powertrain_efficiency: float = 0.92,
) -> dict[str, float]:
    """Approximate energy footprint per vehicle using a simple longitudinal model.

    Power demand at the wheels:

        P = m * a * v + m * g * c_r * v + 0.5 * rho * C_d * A * v^3

    Negative power is regenerated at ``powertrain_efficiency`` (mechanical to
    battery); positive power is divided by ``powertrain_efficiency`` to convert
    to consumed battery energy.
    """

    if isinstance(v_pred, torch.Tensor):
        v = v_pred.detach().cpu().numpy()
    else:
        v = np.asarray(v_pred)
    if isinstance(a_pred, torch.Tensor):
        a = a_pred.detach().cpu().numpy()
    else:
        a = np.asarray(a_pred)
    if v.shape != a.shape:
        raise ValueError("v_pred and a_pred shapes must match")
    g = 9.81
    accel_power = mass_kg * a * v
    rolling_power = mass_kg * g * rolling_resistance * v
    drag_power = 0.5 * air_density * drag_coefficient * frontal_area * (v ** 3)
    p_total = accel_power + rolling_power + drag_power
    consumed = np.where(p_total > 0, p_total / powertrain_efficiency, p_total * powertrain_efficiency)
    dt = 1.0 / float(target_hz)
    energy = consumed.sum(axis=1) * dt  # (B, N)
    return {
        "mean_energy_kJ_per_vehicle": float(energy.mean() / 1000.0),
        "p95_energy_kJ_per_vehicle": float(np.percentile(energy / 1000.0, 95)),
        "mean_power_kW": float(consumed.mean() / 1000.0),
    }


@dataclass(slots=True)
class InferenceLatencyReport:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    n_runs: int
    device: str


def inference_latency(
    forward_callable,
    sample_input,
    n_warmup: int = 5,
    n_runs: int = 50,
    device: torch.device | None = None,
) -> InferenceLatencyReport:
    """Measure per-call wall-clock time of ``forward_callable(sample_input)``.

    The callable is responsible for moving the input to the right device.
    CUDA timings synchronise after each call.
    """

    is_cuda = isinstance(device, torch.device) and device.type == "cuda"
    timings: list[float] = []
    with torch.inference_mode():
        for _ in range(n_warmup):
            forward_callable(sample_input)
            if is_cuda:
                torch.cuda.synchronize()
        for _ in range(n_runs):
            start = time.perf_counter()
            forward_callable(sample_input)
            if is_cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(timings, dtype=np.float64)
    return InferenceLatencyReport(
        mean_ms=float(arr.mean()),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        n_runs=len(arr),
        device=str(device) if device else "cpu",
    )
