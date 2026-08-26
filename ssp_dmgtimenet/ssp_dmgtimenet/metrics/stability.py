"""String-stability metrics in time and frequency domains.

These metrics support both numpy and PyTorch tensors so that the very same
implementations can be reused for evaluation (numpy) and for the
differentiable training loss (torch). When a torch tensor is fed in we keep
the autograd graph so :mod:`ssp_dmgtimenet.losses.stability` can use these
helpers directly.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

ArrayOrTensor = np.ndarray | torch.Tensor


def _is_torch(x: ArrayOrTensor) -> bool:
    return isinstance(x, torch.Tensor)


def detrended_velocity(v: ArrayOrTensor, window_steps: int) -> ArrayOrTensor:
    """Subtract a running mean of length ``window_steps`` along the time axis.

    ``v`` can have shape ``(B, T, N)`` or ``(T, N)``. The trend is computed
    with reflective padding to avoid shrinking the time axis.
    """

    if window_steps < 2:
        if _is_torch(v):
            return v.clone()
        return np.array(v, copy=True)

    if _is_torch(v):
        return _detrend_torch(v, window_steps)
    return _detrend_numpy(v, window_steps)


def _detrend_torch(v: torch.Tensor, window_steps: int) -> torch.Tensor:
    if v.dim() not in (2, 3):
        raise ValueError(f"detrended_velocity expects 2D or 3D, got {v.shape}")
    added_batch = False
    if v.dim() == 2:
        v = v.unsqueeze(0)
        added_batch = True
    B, T, N = v.shape
    pad = window_steps // 2
    v_pad = torch.nn.functional.pad(v.transpose(1, 2), (pad, pad), mode="reflect")
    weight = torch.ones(N, 1, window_steps, device=v.device, dtype=v.dtype) / float(window_steps)
    trend = torch.nn.functional.conv1d(v_pad, weight, groups=N)
    trend = trend[..., :T].transpose(1, 2)
    out = v - trend
    if added_batch:
        out = out.squeeze(0)
    return out


def _detrend_numpy(v: np.ndarray, window_steps: int) -> np.ndarray:
    arr = np.ascontiguousarray(v)
    if arr.ndim not in (2, 3):
        raise ValueError(f"detrended_velocity expects 2D or 3D, got {arr.shape}")
    added_batch = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        added_batch = True
    B, T, N = arr.shape
    pad = window_steps // 2
    pad_arr = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(window_steps) / float(window_steps)
    trend = np.empty_like(arr)
    for b in range(B):
        for n in range(N):
            trend[b, :, n] = np.convolve(pad_arr[b, :, n], kernel, mode="valid")[:T]
    out = arr - trend
    if added_batch:
        out = out[0]
    return out


def adjacent_amplification(
    v: ArrayOrTensor,
    detrend_window_steps: int,
    eps: float = 1e-6,
) -> ArrayOrTensor:
    """Compute ``A_i = ||v_{i+1}|| / ||v_i||`` per (batch, vehicle-pair).

    Returns shape ``(B, N-1)`` (or ``(N-1,)`` if input is 2D).
    """

    detrended = detrended_velocity(v, detrend_window_steps)
    if _is_torch(detrended):
        if detrended.dim() == 2:
            detrended = detrended.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        norms = torch.linalg.vector_norm(detrended, dim=1)  # (B, N)
        ratios = norms[:, 1:] / (norms[:, :-1] + eps)
        return ratios.squeeze(0) if squeeze else ratios

    arr = detrended
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False
    norms = np.linalg.norm(arr, axis=1)
    ratios = norms[:, 1:] / (norms[:, :-1] + eps)
    return ratios[0] if squeeze else ratios


def subplatoon_amplification(
    v: ArrayOrTensor,
    detrend_window_steps: int,
    eps: float = 1e-6,
) -> ArrayOrTensor:
    """Compute ``A_{j -> i}`` for every ordered pair ``j < i``.

    Returns shape ``(B, N, N)`` where entry ``[j, i]`` for ``j < i`` is the
    amplification ratio and other entries are zero.
    """

    detrended = detrended_velocity(v, detrend_window_steps)
    if _is_torch(detrended):
        if detrended.dim() == 2:
            detrended = detrended.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, _, N = detrended.shape
        norms = torch.linalg.vector_norm(detrended, dim=1)
        ratios = torch.zeros(B, N, N, device=detrended.device, dtype=detrended.dtype)
        for j in range(N - 1):
            ratios[:, j, j + 1 :] = norms[:, j + 1 :] / (norms[:, j : j + 1] + eps)
        return ratios.squeeze(0) if squeeze else ratios

    arr = detrended
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False
    B, _, N = arr.shape
    norms = np.linalg.norm(arr, axis=1)
    ratios = np.zeros((B, N, N), dtype=arr.dtype)
    for j in range(N - 1):
        ratios[:, j, j + 1 :] = norms[:, j + 1 :] / (norms[:, j : j + 1] + eps)
    return ratios[0] if squeeze else ratios


def fft_gain(
    v: ArrayOrTensor,
    target_hz: float,
    band: tuple[float, float] = (0.05, 0.5),
    detrend_window_steps: int = 0,
    eps: float = 1e-6,
    n_fft: int | None = None,
    return_magnitudes: bool = False,
) -> tuple[ArrayOrTensor, np.ndarray] | tuple[ArrayOrTensor, np.ndarray, ArrayOrTensor]:
    """Compute ``G_{j -> i}(f) = |V_i(f)| / |V_j(f)|`` over a frequency band.

    Returns ``(gain, freqs)`` where ``gain`` has shape ``(B, N, N, F_band)``
    and ``freqs`` is the numpy array of frequencies retained in ``band``. The
    ``j == i`` and ``j > i`` entries are zero.

    ``n_fft`` zero-pads the (detrended) signal before the transform so the
    band is sampled on a finer frequency grid. With the native 3s prediction
    window (T=30 @ 10Hz) the rfft resolution is 0.333Hz and band (0.05, 0.5)
    contains a single bin; padding interpolates the spectrum so per-frequency
    gain curves become meaningful.

    With ``return_magnitudes=True`` the per-vehicle band magnitudes
    ``|V(f)|`` of shape ``(B, N, F_band)`` are appended to the return value so
    callers can gate the gain statistics on upstream spectral excitation (the
    ratio is a division artefact wherever the denominator magnitude is ~0).
    """

    if detrend_window_steps:
        v_eff = detrended_velocity(v, detrend_window_steps)
    else:
        v_eff = v
    if _is_torch(v_eff):
        if v_eff.dim() == 2:
            v_eff = v_eff.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        B, T, N = v_eff.shape
        n = _resolve_n_fft(T, n_fft)
        spectra = torch.fft.rfft(v_eff, n=n, dim=1)
        magnitudes = torch.abs(spectra)
        freqs = torch.fft.rfftfreq(n, d=1.0 / target_hz).cpu().numpy()
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(band_mask):
            raise ValueError(f"No FFT bins fall in band {band} given n_fft={n}, fs={target_hz}")
        magnitudes_band = magnitudes[:, band_mask, :]  # (B, F_band, N)
        magnitudes_band = magnitudes_band.transpose(1, 2)  # (B, N, F_band)
        gain = torch.zeros(B, N, N, magnitudes_band.shape[-1], device=v_eff.device, dtype=v_eff.dtype)
        for j in range(N - 1):
            denom = magnitudes_band[:, j : j + 1, :] + eps
            gain[:, j, j + 1 :, :] = magnitudes_band[:, j + 1 :, :] / denom
        if squeeze:
            gain = gain.squeeze(0)
            magnitudes_band = magnitudes_band.squeeze(0)
        if return_magnitudes:
            return gain, freqs[band_mask], magnitudes_band
        return gain, freqs[band_mask]

    arr = v_eff
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False
    B, T, N = arr.shape
    n = _resolve_n_fft(T, n_fft)
    spectra = np.fft.rfft(arr, n=n, axis=1)
    magnitudes = np.abs(spectra)
    freqs = np.fft.rfftfreq(n, d=1.0 / target_hz)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        raise ValueError(f"No FFT bins fall in band {band} given n_fft={n}, fs={target_hz}")
    mags = magnitudes[:, band_mask, :].transpose(0, 2, 1)
    gain = np.zeros((B, N, N, mags.shape[-1]), dtype=arr.dtype)
    for j in range(N - 1):
        denom = mags[:, j : j + 1, :] + eps
        gain[:, j, j + 1 :, :] = mags[:, j + 1 :, :] / denom
    if squeeze:
        gain = gain[0]
        mags = mags[0]
    if return_magnitudes:
        return gain, freqs[band_mask], mags
    return gain, freqs[band_mask]


def _resolve_n_fft(T: int, n_fft: int | None) -> int:
    if n_fft is None:
        return T
    if n_fft < T:
        raise ValueError(f"n_fft={n_fft} must be >= signal length T={T}")
    return n_fft


def phase_delay(
    v: ArrayOrTensor,
    target_hz: float,
    detrend_window_steps: int = 0,
    max_lag_seconds: float | None = None,
    subsample: bool = False,
) -> ArrayOrTensor:
    """Per-pair propagation delay via cross-correlation peak.

    Returns shape ``(B, N - 1)`` (seconds). Detrending is recommended.

    ``max_lag_seconds`` restricts the argmax search to lags within
    ``[-max_lag, +max_lag]``. Without it the estimator on short windows
    frequently locks onto the boundary lags ``±(T-1)`` where only a handful
    of samples overlap, producing the ±(T-1)/fs artefact bands observed in
    the v5 figures. The physical pair delay is bounded by tau_max (2.5s), so
    a cap of a few seconds keeps the search inside the well-supported region.

    ``subsample`` refines the integer-lag argmax with a parabolic fit of the
    three correlation samples around the peak, yielding sub-sample (finer than
    ``1/target_hz``) delay resolution. It is opt-in so the default evaluator
    numbers stay on the native integer-lag grid; the interpretability scatter
    enables it to avoid the quantisation banding at multiples of ``1/fs``.
    """

    if detrend_window_steps:
        v_eff = detrended_velocity(v, detrend_window_steps)
    else:
        v_eff = v
    arr = v_eff.detach().cpu().numpy() if isinstance(v_eff, torch.Tensor) else np.ascontiguousarray(v_eff)
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    else:
        squeeze = False
    B, T, N = arr.shape
    if max_lag_seconds is not None:
        if max_lag_seconds <= 0:
            raise ValueError(f"max_lag_seconds must be positive, got {max_lag_seconds}")
        max_lag = min(int(round(max_lag_seconds * target_hz)), T - 1)
    else:
        max_lag = T - 1
    centre = T - 1
    delays = np.zeros((B, N - 1), dtype=np.float32)
    for b in range(B):
        for i in range(N - 1):
            x = arr[b, :, i] - arr[b, :, i].mean()
            y = arr[b, :, i + 1] - arr[b, :, i + 1].mean()
            denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
            corr = np.correlate(y, x, mode="full") / denom
            window = corr[centre - max_lag : centre + max_lag + 1]
            k = int(np.argmax(window))
            lag = float(k - max_lag)
            if subsample and 0 < k < window.shape[0] - 1:
                cm, c0, cp = window[k - 1], window[k], window[k + 1]
                curvature = cm - 2.0 * c0 + cp
                if abs(curvature) > 1e-12:
                    lag += float(np.clip(0.5 * (cm - cp) / curvature, -0.5, 0.5))
            delays[b, i] = lag / float(target_hz)
    return delays[0] if squeeze else delays


@dataclass(slots=True)
class UnstableWindowReport:
    unstable_window_ratio: float
    exceedance_area: float
    max_amplification: float
    pair_unstable_ratio: dict[str, float]
    excitation_retained_ratio: float = 1.0
    excitation_n_retained: int = 0


@dataclass(slots=True)
class StrictJointStabilityReport:
    """Per-window implementation of the strict time-and-frequency criterion.

    A retained window is jointly stable only when both the maximum
    sub-platoon time-domain amplification and the maximum pointwise spectral
    gain are no greater than ``1 + delta``.
    """

    time_unstable_window_ratio: float
    frequency_unstable_window_ratio: float
    joint_unstable_window_ratio: float
    max_time_amplification: float
    max_frequency_gain: float
    excitation_retained_ratio: float
    excitation_n_retained: int
    fft_n_bins: int


def vehicle_excitation_amplitude(
    v: ArrayOrTensor,
    detrend_window_steps: int,
) -> np.ndarray:
    """Per-(window, vehicle) RMS amplitude (m/s) of the detrended velocity.

    ``v`` has shape ``(B, T, N)`` (or ``(T, N)`` for a single window).
    Returns shape ``(B, N)``.
    """

    arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.ascontiguousarray(v)
    if arr.ndim == 2:
        arr = arr[None, ...]
    detrended = detrended_velocity(arr, detrend_window_steps)
    return np.sqrt(np.mean(np.square(detrended), axis=1))


def leader_excitation_amplitude(
    v: ArrayOrTensor,
    detrend_window_steps: int,
) -> np.ndarray:
    """Per-window RMS amplitude (m/s) of the detrended leader velocity.

    ``v`` has shape ``(B, T, N)`` (or ``(T, N)`` for a single window); vehicle
    index 0 is the platoon leader. Returns shape ``(B,)``.
    """

    return vehicle_excitation_amplitude(v, detrend_window_steps)[:, 0]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        return float("nan"), float("nan")
    proportion = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius))


def disturbance_detection_stats(
    v_pred: ArrayOrTensor,
    v_gt: ArrayOrTensor,
    detrend_window_steps: int,
    excitation_floor: float = 0.05,
) -> dict[str, float]:
    """Confusion-matrix statistics for predicted leader excitation.

    A positive event is defined independently for prediction and ground truth
    by the same detrended leader RMS threshold.  ``coverage`` is recall on GT
    disturbances; ``fpr`` measures spurious predicted disturbances on GT-quiet
    windows.
    """

    pred_excited = (
        leader_excitation_amplitude(v_pred, detrend_window_steps) >= excitation_floor
    )
    gt_excited = leader_excitation_amplitude(v_gt, detrend_window_steps) >= excitation_floor
    if pred_excited.shape != gt_excited.shape:
        raise ValueError(
            "v_pred and v_gt must contain the same number of windows, "
            f"got {pred_excited.shape} and {gt_excited.shape}"
        )

    tp = int(np.logical_and(gt_excited, pred_excited).sum())
    fp = int(np.logical_and(~gt_excited, pred_excited).sum())
    fn = int(np.logical_and(gt_excited, ~pred_excited).sum())
    tn = int(np.logical_and(~gt_excited, ~pred_excited).sum())
    n_gt = tp + fn
    n_quiet = fp + tn
    n_pred = tp + fp
    total = tp + fp + fn + tn
    coverage = _safe_rate(tp, n_gt)
    fpr = _safe_rate(fp, n_quiet)
    precision = _safe_rate(tp, n_pred)
    coverage_low, coverage_high = wilson_interval(tp, n_gt)
    fpr_low, fpr_high = wilson_interval(fp, n_quiet)

    return {
        "n_total": float(total),
        "n_gt_excited": float(n_gt),
        "n_gt_quiet": float(n_quiet),
        "n_pred_excited": float(n_pred),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "coverage": coverage,
        "coverage_ci95_low": coverage_low,
        "coverage_ci95_high": coverage_high,
        "fpr": fpr,
        "fpr_ci95_low": fpr_low,
        "fpr_ci95_high": fpr_high,
        "precision": precision,
        "specificity": _safe_rate(tn, n_quiet),
        "predicted_positive_rate": _safe_rate(n_pred, total),
        "gt_prevalence": _safe_rate(n_gt, total),
    }


def conditional_internal_amplification_stats(
    v_pred: ArrayOrTensor,
    v_gt: ArrayOrTensor,
    detrend_window_steps: int,
    *,
    delta: float = 0.0,
    excitation_floor: float = 0.05,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Leader-referenced internal gain on ``GT-excited AND pred-excited`` windows."""

    pred_amp = vehicle_excitation_amplitude(v_pred, detrend_window_steps)
    gt_leader_amp = leader_excitation_amplitude(v_gt, detrend_window_steps)
    if pred_amp.shape[0] != gt_leader_amp.shape[0]:
        raise ValueError(
            "v_pred and v_gt must contain the same number of windows, "
            f"got {pred_amp.shape[0]} and {gt_leader_amp.shape[0]}"
        )
    if pred_amp.shape[1] < 2:
        raise ValueError("v_pred must contain a leader and at least one follower")

    gt_excited = gt_leader_amp >= excitation_floor
    pred_excited = pred_amp[:, 0] >= excitation_floor
    keep = np.logical_and(gt_excited, pred_excited)
    n_gt = int(gt_excited.sum())
    n_windows = int(keep.sum())
    if n_windows == 0:
        return {
            "n_windows": 0.0,
            "support_ratio": _safe_rate(0, n_gt),
            "unstable_window_ratio": float("nan"),
            "unstable_ci95_low": float("nan"),
            "unstable_ci95_high": float("nan"),
            "max_gain": float("nan"),
            "p95_gain": float("nan"),
            "mean_gain": float("nan"),
            "mean_exceedance": float("nan"),
        }

    gains = pred_amp[keep, 1:] / (pred_amp[keep, 0:1] + eps)
    threshold = 1.0 + delta
    unstable = (gains > threshold).any(axis=1)
    n_unstable = int(unstable.sum())
    ci_low, ci_high = wilson_interval(n_unstable, n_windows)
    result = {
        "n_windows": float(n_windows),
        "support_ratio": _safe_rate(n_windows, n_gt),
        "unstable_window_ratio": float(unstable.mean()),
        "unstable_ci95_low": ci_low,
        "unstable_ci95_high": ci_high,
        "max_gain": float(gains.max()),
        "p95_gain": float(np.quantile(gains, 0.95)),
        "mean_gain": float(gains.mean()),
        "mean_exceedance": float(np.maximum(gains - threshold, 0.0).mean()),
    }
    for follower_idx in range(gains.shape[1]):
        label = f"C1_to_C{follower_idx + 2}"
        result[f"{label}_mean_gain"] = float(gains[:, follower_idx].mean())
        result[f"{label}_p95_gain"] = float(np.quantile(gains[:, follower_idx], 0.95))
    return result


def strict_joint_stability_metrics(
    v_time: ArrayOrTensor,
    *,
    detrend_window_steps: int,
    target_hz: float,
    band: tuple[float, float] = (0.05, 0.5),
    delta: float = 0.0,
    eps: float = 1e-6,
    excitation_floor: float = 0.05,
    floor_reference_v: ArrayOrTensor | None = None,
    v_frequency: ArrayOrTensor | None = None,
    n_fft: int | None = None,
) -> StrictJointStabilityReport:
    """Evaluate the strict conjunctive criterion from Eq. (5) per window.

    The time-domain term is ``max_{j<i} A_{j->i}``, evaluated on ``v_time``.
    The frequency-domain term is ``max_{j<i,f in band} G_{j->i}(f)``,
    evaluated on ``v_frequency`` when supplied, otherwise on ``v_time``.
    The pointwise spectral maximum intentionally follows the strict equation;
    unlike :func:`fft_band_gain_stats`, it is sensitive to near-empty
    denominator bins. Reporting both metrics makes that distinction explicit.

    Windows are retained when the predicted leader clears
    ``excitation_floor``. If ``floor_reference_v`` is supplied, its leader
    must clear the same floor as well.
    """

    time_arr = (
        v_time.detach().cpu().numpy() if isinstance(v_time, torch.Tensor) else np.asarray(v_time)
    )
    if time_arr.ndim == 2:
        time_arr = time_arr[None, ...]
    if time_arr.ndim != 3:
        raise ValueError(f"v_time must be 2D or 3D, got {time_arr.shape}")

    frequency_source = v_time if v_frequency is None else v_frequency
    frequency_arr = (
        frequency_source.detach().cpu().numpy()
        if isinstance(frequency_source, torch.Tensor)
        else np.asarray(frequency_source)
    )
    if frequency_arr.ndim == 2:
        frequency_arr = frequency_arr[None, ...]
    if frequency_arr.ndim != 3:
        raise ValueError(f"v_frequency must be 2D or 3D, got {frequency_arr.shape}")
    if frequency_arr.shape[0] != time_arr.shape[0] or frequency_arr.shape[2] != time_arr.shape[2]:
        raise ValueError(
            "v_time and v_frequency must agree on batch and vehicle dimensions, "
            f"got {time_arr.shape} and {frequency_arr.shape}"
        )

    sub_amp = np.asarray(
        subplatoon_amplification(time_arr, detrend_window_steps, eps=eps)
    )
    gain, freqs = fft_gain(
        frequency_arr,
        target_hz=target_hz,
        band=band,
        detrend_window_steps=detrend_window_steps,
        eps=eps,
        n_fft=n_fft,
    )
    gain_np = gain.detach().cpu().numpy() if isinstance(gain, torch.Tensor) else np.asarray(gain)

    n_windows, _, n_vehicles = time_arr.shape
    upper = np.triu(np.ones((n_vehicles, n_vehicles), dtype=bool), k=1)
    time_window_max = sub_amp[:, upper].max(axis=1)
    frequency_window_max = gain_np[:, upper, :].max(axis=(1, 2))

    if excitation_floor > 0.0:
        keep = leader_excitation_amplitude(time_arr, detrend_window_steps) >= excitation_floor
        if floor_reference_v is not None:
            reference_arr = (
                floor_reference_v.detach().cpu().numpy()
                if isinstance(floor_reference_v, torch.Tensor)
                else np.asarray(floor_reference_v)
            )
            if reference_arr.ndim == 2:
                reference_arr = reference_arr[None, ...]
            if reference_arr.shape[0] != n_windows:
                raise ValueError(
                    "floor_reference_v must share the batch dimension with v_time, "
                    f"got {reference_arr.shape} and {time_arr.shape}"
                )
            keep &= (
                leader_excitation_amplitude(reference_arr, detrend_window_steps)
                >= excitation_floor
            )
    else:
        if floor_reference_v is not None:
            raise ValueError("floor_reference_v requires excitation_floor > 0")
        keep = np.ones(n_windows, dtype=bool)

    n_retained = int(keep.sum())
    retained_ratio = float(keep.mean())
    if n_retained == 0:
        return StrictJointStabilityReport(
            time_unstable_window_ratio=0.0,
            frequency_unstable_window_ratio=0.0,
            joint_unstable_window_ratio=0.0,
            max_time_amplification=0.0,
            max_frequency_gain=0.0,
            excitation_retained_ratio=retained_ratio,
            excitation_n_retained=0,
            fft_n_bins=int(freqs.size),
        )

    threshold = 1.0 + delta
    time_unstable = time_window_max[keep] > threshold
    frequency_unstable = frequency_window_max[keep] > threshold
    return StrictJointStabilityReport(
        time_unstable_window_ratio=float(time_unstable.mean()),
        frequency_unstable_window_ratio=float(frequency_unstable.mean()),
        joint_unstable_window_ratio=float(np.logical_or(time_unstable, frequency_unstable).mean()),
        max_time_amplification=float(time_window_max[keep].max()),
        max_frequency_gain=float(frequency_window_max[keep].max()),
        excitation_retained_ratio=retained_ratio,
        excitation_n_retained=n_retained,
        fft_n_bins=int(freqs.size),
    )


def gt_referenced_amplification_stats(
    v_pred: ArrayOrTensor,
    v_gt: ArrayOrTensor,
    detrend_window_steps: int,
    *,
    delta: float = 0.0,
    excitation_floor: float = 0.05,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Follower response gains on one fixed, ground-truth-excited support.

    Every model is evaluated on the same windows selected solely by the
    ground-truth leader RMS.  Each predicted follower uses that same
    ground-truth leader as denominator:

    ``R_i = ||v_i^pred||_2 / (||v_1^GT||_2 + eps)``, ``i = 2, ..., N``.

    This is an externally referenced response-gain metric, not the model's
    internal adjacent-vehicle amplification.  It is intended for fair
    cross-model evaluation; the internal ratio remains useful as a diagnostic
    and as a differentiable training regulariser.
    """

    pred = v_pred.detach().cpu().numpy() if isinstance(v_pred, torch.Tensor) else np.asarray(v_pred)
    gt = v_gt.detach().cpu().numpy() if isinstance(v_gt, torch.Tensor) else np.asarray(v_gt)
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    if pred.ndim != 3 or gt.ndim != 3:
        raise ValueError(f"v_pred and v_gt must be 2D or 3D, got {pred.shape} and {gt.shape}")
    if pred.shape[:2] != gt.shape[:2]:
        raise ValueError(f"v_pred and v_gt must agree on (B, T), got {pred.shape} and {gt.shape}")
    if pred.shape[2] < 2 or gt.shape[2] < 1:
        raise ValueError("v_pred needs at least two vehicles and v_gt needs a leader")

    pred_det = np.asarray(detrended_velocity(pred, detrend_window_steps))
    gt_det = np.asarray(detrended_velocity(gt, detrend_window_steps))
    pred_norm = np.linalg.norm(pred_det, axis=1)
    gt_leader_norm = np.linalg.norm(gt_det[:, :, 0], axis=1)
    gt_leader_rms = np.sqrt(np.mean(np.square(gt_det[:, :, 0]), axis=1))
    keep = gt_leader_rms >= excitation_floor
    n_windows = int(keep.sum())
    if n_windows == 0:
        return {
            "n_windows": 0.0,
            "unstable_window_ratio": float("nan"),
            "unstable_ci95_low": float("nan"),
            "unstable_ci95_high": float("nan"),
            "max_gain": float("nan"),
            "p95_gain": float("nan"),
            "mean_gain": float("nan"),
            "mean_exceedance": float("nan"),
        }

    gains = pred_norm[keep, 1:] / (gt_leader_norm[keep, None] + eps)
    threshold = 1.0 + delta
    unstable = (gains > threshold).any(axis=1)
    n_unstable = int(unstable.sum())
    ci_low, ci_high = wilson_interval(n_unstable, n_windows)
    result = {
        "n_windows": float(n_windows),
        "unstable_window_ratio": float(unstable.mean()),
        "unstable_ci95_low": ci_low,
        "unstable_ci95_high": ci_high,
        "max_gain": float(gains.max()),
        "p95_gain": float(np.quantile(gains, 0.95)),
        "mean_gain": float(gains.mean()),
        "mean_exceedance": float(np.maximum(gains - threshold, 0.0).mean()),
    }
    for follower_idx in range(gains.shape[1]):
        label = f"C1_to_C{follower_idx + 2}"
        result[f"{label}_mean_gain"] = float(gains[:, follower_idx].mean())
        result[f"{label}_p95_gain"] = float(np.quantile(gains[:, follower_idx], 0.95))
    return result


def gt_referenced_fft_band_gain_stats(
    v_pred: ArrayOrTensor,
    v_gt: ArrayOrTensor,
    excitation_reference_v: ArrayOrTensor,
    *,
    target_hz: float,
    band: tuple[float, float] = (0.05, 0.5),
    detrend_window_steps: int = 8,
    n_fft: int | None = None,
    excitation_floor: float = 0.05,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Band-L2 follower gains with a shared GT-leader denominator and mask."""

    pred = v_pred.detach().cpu().numpy() if isinstance(v_pred, torch.Tensor) else np.asarray(v_pred)
    gt = v_gt.detach().cpu().numpy() if isinstance(v_gt, torch.Tensor) else np.asarray(v_gt)
    gate_v = (
        excitation_reference_v.detach().cpu().numpy()
        if isinstance(excitation_reference_v, torch.Tensor)
        else np.asarray(excitation_reference_v)
    )
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    if gate_v.ndim == 2:
        gate_v = gate_v[None, ...]
    if pred.shape[:2] != gt.shape[:2] or pred.shape[0] != gate_v.shape[0]:
        raise ValueError(
            "v_pred/v_gt must agree on (B, T), and excitation_reference_v must share B; "
            f"got {pred.shape}, {gt.shape}, {gate_v.shape}"
        )

    _, _, pred_mags = fft_gain(
        pred,
        target_hz=target_hz,
        band=band,
        detrend_window_steps=detrend_window_steps,
        n_fft=n_fft,
        return_magnitudes=True,
    )
    _, _, gt_mags = fft_gain(
        gt,
        target_hz=target_hz,
        band=band,
        detrend_window_steps=detrend_window_steps,
        n_fft=n_fft,
        return_magnitudes=True,
    )
    pred_band_norm = np.linalg.norm(np.asarray(pred_mags), axis=-1)
    gt_leader_band_norm = np.linalg.norm(np.asarray(gt_mags)[:, 0, :], axis=-1)
    keep = leader_excitation_amplitude(gate_v, detrend_window_steps) >= excitation_floor
    n_windows = int(keep.sum())
    if n_windows == 0:
        return {
            "n_windows": 0.0,
            "max_gain": 0.0,
            "p95_gain": 0.0,
            "mean_gain": 0.0,
        }

    gains = pred_band_norm[keep, 1:] / (gt_leader_band_norm[keep, None] + eps)
    return {
        "n_windows": float(n_windows),
        "max_gain": float(gains.max()),
        "p95_gain": float(np.quantile(gains, 0.95)),
        "mean_gain": float(gains.mean()),
    }


def fft_band_gain_stats(
    v: ArrayOrTensor,
    target_hz: float,
    band: tuple[float, float] = (0.05, 0.5),
    detrend_window_steps: int = 8,
    n_fft: int | None = None,
    excitation_floor: float = 0.05,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Band-restricted L2 spectral gain statistics over ordered pairs ``j < i``.

    For each window and pair the gain is ``||V_i||_band / ||V_j||_band`` (L2
    norm of the rfft magnitudes inside ``band``), the band-limited L2
    string-stability gain. Pairs whose *upstream* vehicle ``j`` has a
    detrended time-domain RMS amplitude below ``excitation_floor`` (m/s) are
    excluded: a flat upstream signal makes the ratio a division artefact.
    Aggregating the band norm (instead of per-frequency ratios) keeps a single
    well-conditioned ratio per pair, so isolated near-empty frequency bins
    cannot blow up the statistic either.

    Returns ``fft_gain_max`` / ``fft_gain_mean`` over retained pair-windows
    (``0.0`` when none survive), ``fft_pairs_retained_ratio`` and
    ``fft_n_bins``.
    """

    _, freqs, mags = fft_gain(
        v,
        target_hz=target_hz,
        band=band,
        detrend_window_steps=detrend_window_steps,
        eps=eps,
        n_fft=n_fft,
        return_magnitudes=True,
    )
    mags_np = mags.detach().cpu().numpy() if isinstance(mags, torch.Tensor) else np.asarray(mags)
    if mags_np.ndim == 2:
        mags_np = mags_np[None, ...]
    band_norm = np.linalg.norm(mags_np, axis=-1)  # (B, N)
    excitation = vehicle_excitation_amplitude(v, detrend_window_steps)
    B, N = band_norm.shape
    ratios: list[np.ndarray] = []
    total_pairs = 0
    for j in range(N - 1):
        total_pairs += B * (N - 1 - j)
        if excitation_floor > 0.0:
            keep = excitation[:, j] >= excitation_floor
        else:
            keep = np.ones(B, dtype=bool)
        if not np.any(keep):
            continue
        g = band_norm[keep, j + 1 :] / (band_norm[keep, j : j + 1] + eps)
        ratios.append(g.ravel())
    retained = np.concatenate(ratios) if ratios else np.empty(0, dtype=band_norm.dtype)
    if retained.size == 0:
        return {
            "fft_gain_max": 0.0,
            "fft_gain_mean": 0.0,
            "fft_pairs_retained_ratio": 0.0,
            "fft_n_bins": float(freqs.size),
        }
    return {
        "fft_gain_max": float(retained.max()),
        "fft_gain_mean": float(retained.mean()),
        "fft_pairs_retained_ratio": float(retained.size / total_pairs),
        "fft_n_bins": float(freqs.size),
    }


def unstable_window_metrics(
    v: ArrayOrTensor,
    detrend_window_steps: int,
    delta: float = 0.0,
    eps: float = 1e-6,
    excitation_floor: float = 0.05,
    floor_reference_v: ArrayOrTensor | None = None,
) -> UnstableWindowReport:
    """Aggregate adjacent-amplification statistics across the batch.

    A *window* is one element of the batch dimension. It is unstable if any
    adjacent pair has ``A_i > 1 + delta``.

    Windows whose detrended leader RMS amplitude is below ``excitation_floor``
    (m/s) are excluded from every statistic: with a (near-)constant upstream
    signal the amplification ratio degenerates to ``x / eps`` and reports the
    structural division artefact instead of platoon dynamics. Pass
    ``excitation_floor=0.0`` to keep every window. When no window survives the
    floor, the ratio/area/max statistics are reported as ``0.0`` and
    ``excitation_retained_ratio`` / ``excitation_n_retained`` expose the empty
    support so downstream tables can mark the cell as undefined.

    ``floor_reference_v`` enables the *unified reference subset* protocol: a
    second velocity tensor (typically the ground truth) of the same shape
    whose leader excitation must also clear ``excitation_floor`` for a window
    to be retained. The support becomes the intersection ``{pred leader
    excited} AND {reference leader excited}``, so every model is compared on
    the same externally-defined excitation events while the per-model floor
    still guards the ratio denominator against division artefacts.
    ``excitation_retained_ratio`` stays relative to the *full* batch.
    """

    A = adjacent_amplification(v, detrend_window_steps, eps=eps)
    if isinstance(A, torch.Tensor):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)
    if A_np.ndim == 1:
        A_np = A_np[None, ...]
    n_pairs = A_np.shape[1]
    if floor_reference_v is not None and excitation_floor <= 0.0:
        raise ValueError("floor_reference_v requires excitation_floor > 0")
    if excitation_floor > 0.0:
        amplitude = leader_excitation_amplitude(v, detrend_window_steps)
        keep = amplitude >= excitation_floor
        if floor_reference_v is not None:
            ref_amplitude = leader_excitation_amplitude(floor_reference_v, detrend_window_steps)
            if ref_amplitude.shape != amplitude.shape:
                raise ValueError(
                    f"floor_reference_v window count {ref_amplitude.shape} does not "
                    f"match v window count {amplitude.shape}"
                )
            keep = keep & (ref_amplitude >= excitation_floor)
    else:
        keep = np.ones(A_np.shape[0], dtype=bool)
    retained = A_np[keep]
    retained_ratio = float(keep.mean())
    n_retained = int(keep.sum())
    threshold = 1.0 + delta
    if n_retained == 0:
        pair_ratios = {f"C{i + 1}->C{i + 2}": 0.0 for i in range(n_pairs)}
        return UnstableWindowReport(
            unstable_window_ratio=0.0,
            exceedance_area=0.0,
            max_amplification=0.0,
            pair_unstable_ratio=pair_ratios,
            excitation_retained_ratio=retained_ratio,
            excitation_n_retained=0,
        )
    pair_unstable = (retained > threshold).any(axis=1)
    pair_ratios = {}
    for i in range(n_pairs):
        pair_ratios[f"C{i + 1}->C{i + 2}"] = float((retained[:, i] > threshold).mean())
    return UnstableWindowReport(
        unstable_window_ratio=float(pair_unstable.mean()),
        exceedance_area=float(np.maximum(0.0, retained - threshold).sum()),
        max_amplification=float(retained.max()),
        pair_unstable_ratio=pair_ratios,
        excitation_retained_ratio=retained_ratio,
        excitation_n_retained=n_retained,
    )


def aggregate_amplification_distribution(
    A: ArrayOrTensor,
    quantiles: Iterable[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> dict[str, np.ndarray]:
    """Per-pair quantiles of adjacent amplification across the batch."""

    if isinstance(A, torch.Tensor):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)
    if A_np.ndim == 1:
        A_np = A_np[None, ...]
    out: dict[str, np.ndarray] = {}
    for q in quantiles:
        out[f"q{q:.2f}"] = np.quantile(A_np, q, axis=0)
    out["mean"] = A_np.mean(axis=0)
    out["max"] = A_np.max(axis=0)
    return out
