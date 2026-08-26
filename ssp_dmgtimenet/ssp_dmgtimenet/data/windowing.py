"""Resample HighD platoon segments to 10 Hz and slice sliding windows.

Each ``PlatoonSegment`` produced by :mod:`ssp_dmgtimenet.data.platoons` is
materialised into a dense per-vehicle tensor of shape ``(T, N, F_raw)`` and
then chopped into history+predict windows.

Feature layout (in absolute units, before normalisation):

* ``x_abs`` (m)            longitudinal position (signed by direction)
* ``x_rel_leader`` (m)     ``x_abs - x_abs_leader`` so cross-platoon offsets cancel
* ``v`` (m/s)              speed magnitude (always non-negative)
* ``a`` (m/s^2)            longitudinal acceleration
* ``s`` (m)                gap to predecessor; ``NaN`` for the leader; later
  filled with 0 and accompanied by a mask
* ``dv`` (m/s)             v_i - v_{i-1}; ``NaN`` for leader -> 0 + mask
* ``da`` (m/s^2)           a_i - a_{i-1}; ``NaN`` for leader -> 0 + mask
* ``time_headway`` (s)     ``s / max(v, eps)``; same NaN treatment

The window slicer keeps a ``mask`` channel so downstream code never
silently treats NaNs as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils.filters import low_pass_filter, savgol_smooth
from .highd import HighDRecording
from .platoons import PlatoonSegment


FEATURE_NAMES: tuple[str, ...] = (
    "x_abs",
    "x_rel_leader",
    "v",
    "a",
    "s",
    "dv",
    "da",
    "time_headway",
)


@dataclass(slots=True, frozen=True)
class WindowingConfig:
    """Resampling and sliding-window hyper-parameters."""

    target_hz: float = 10.0
    history_seconds: float = 5.0
    predict_seconds: float = 3.0
    stride_seconds: float = 1.0
    minimum_total_seconds: float = 10.0
    smoothing_window_sec: float = 0.5
    savgol_polyorder: int = 3
    lowpass_cutoff_hz: float = 4.0
    epsilon_velocity: float = 0.1

    @property
    def history_steps(self) -> int:
        return int(round(self.history_seconds * self.target_hz))

    @property
    def predict_steps(self) -> int:
        return int(round(self.predict_seconds * self.target_hz))

    @property
    def stride_steps(self) -> int:
        return max(1, int(round(self.stride_seconds * self.target_hz)))

    @property
    def total_steps(self) -> int:
        return self.history_steps + self.predict_steps


@dataclass(slots=True)
class PlatoonSample:
    """A history+future window for a single platoon."""

    recording_id: int
    lane_id: int
    track_ids: tuple[int, ...]
    history: np.ndarray  # shape (T_hist, N, F_raw)
    future: np.ndarray   # shape (T_fut, N, F_raw)
    history_mask: np.ndarray  # shape (T_hist, N, F_raw); 1 where finite
    future_mask: np.ndarray   # shape (T_fut, N, F_raw)
    vehicle_lengths: np.ndarray  # shape (N,)
    start_time: float
    target_hz: float
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def N(self) -> int:
        return self.history.shape[1]


def _segment_dataframe(
    recording: HighDRecording,
    segment: PlatoonSegment,
) -> pd.DataFrame:
    """Slice ``recording.tracks`` for the given platoon segment."""

    sub = recording.tracks[
        recording.tracks["id"].isin(segment.track_ids)
        & (recording.tracks["frame"] >= segment.start_frame)
        & (recording.tracks["frame"] <= segment.end_frame)
    ].copy()
    if sub.empty:
        raise ValueError(f"Empty slice for segment {segment.track_ids}")
    sub = sub.sort_values(["id", "frame"], kind="stable").reset_index(drop=True)
    return sub


def _per_vehicle_pivot(
    sub: pd.DataFrame,
    track_ids: Sequence[int],
    column: str,
) -> np.ndarray:
    pivot = sub.pivot(index="frame", columns="id", values=column).sort_index()
    pivot = pivot.reindex(columns=list(track_ids))
    if pivot.isna().any().any():
        missing_cols = pivot.columns[pivot.isna().any()].tolist()
        raise ValueError(
            f"Missing column={column} for trackIds={missing_cols} inside segment {tuple(track_ids)}"
        )
    return pivot.to_numpy(dtype=np.float64)


def _resample_to_target_hz(arr: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    """Time-axis resampling via linear interpolation onto a uniform grid."""

    if abs(source_hz - target_hz) < 1e-6:
        return arr
    n = arr.shape[0]
    if n < 2:
        raise ValueError("Cannot resample fewer than 2 frames")
    duration = (n - 1) / source_hz
    new_n = int(np.floor(duration * target_hz)) + 1
    if new_n < 2:
        raise ValueError(f"Resampled length {new_n} too small (source duration={duration:.3f}s)")
    src_t = np.arange(n) / source_hz
    tgt_t = np.arange(new_n) / target_hz
    out = np.empty((new_n, *arr.shape[1:]), dtype=arr.dtype)
    flat = arr.reshape(n, -1)
    for col in range(flat.shape[1]):
        out.reshape(new_n, -1)[:, col] = np.interp(tgt_t, src_t, flat[:, col])
    return out


def resample_segment_to_target_hz(
    recording: HighDRecording,
    segment: PlatoonSegment,
    config: WindowingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, vehicle_lengths)`` arrays at the target frame rate.

    ``features`` has shape ``(T_target, N, F_raw)`` with ``F_raw`` matching
    :data:`FEATURE_NAMES`. The ``v`` and ``a`` channels are smoothed with a
    Savitzky-Golay filter at the source frame rate, then low-pass filtered
    before down-sampling so that no aliased high-frequency content survives.
    """

    track_ids = segment.track_ids
    sub = _segment_dataframe(recording, segment)
    fr = recording.frame_rate
    direction = segment.driving_direction
    if direction not in (1, 2):
        raise ValueError(f"Unknown drivingDirection={direction}")
    sign = 1.0 if direction == 1 else -1.0

    x_signed = sign * _per_vehicle_pivot(sub, track_ids, "x")
    width = _per_vehicle_pivot(sub, track_ids, "width")
    vehicle_lengths = width.mean(axis=0)
    if not np.allclose(vehicle_lengths, width[0]):
        raise ValueError("HighD vehicle widths should be constant per trackId; got drift")

    v_signed = sign * _per_vehicle_pivot(sub, track_ids, "xVelocity")
    a_signed = sign * _per_vehicle_pivot(sub, track_ids, "xAcceleration")

    smoothing_window_frames = max(5, int(round(config.smoothing_window_sec * fr)))
    if smoothing_window_frames % 2 == 0:
        smoothing_window_frames += 1
    v_signed = savgol_smooth(v_signed, smoothing_window_frames, config.savgol_polyorder, axis=0)
    a_signed = savgol_smooth(a_signed, smoothing_window_frames, config.savgol_polyorder, axis=0)

    cutoff = min(config.lowpass_cutoff_hz, fr / 2.0 - 0.5)
    if cutoff > 0:
        v_signed = low_pass_filter(v_signed, fs=fr, cutoff_hz=cutoff, axis=0)
        a_signed = low_pass_filter(a_signed, fs=fr, cutoff_hz=cutoff, axis=0)

    leader_x = x_signed[:, [0]]
    x_rel = x_signed - leader_x

    s = np.full_like(x_signed, np.nan)
    if x_signed.shape[1] >= 2:
        s[:, 1:] = x_signed[:, 1:] - x_signed[:, :-1] - vehicle_lengths[np.newaxis, :-1]

    v_mag = np.abs(v_signed)
    dv = np.full_like(v_signed, np.nan)
    da = np.full_like(a_signed, np.nan)
    if v_signed.shape[1] >= 2:
        dv[:, 1:] = v_signed[:, 1:] - v_signed[:, :-1]
        da[:, 1:] = a_signed[:, 1:] - a_signed[:, :-1]

    time_headway = np.full_like(s, np.nan)
    safe_v = np.where(v_mag > config.epsilon_velocity, v_mag, np.nan)
    time_headway[:, 1:] = s[:, 1:] / safe_v[:, 1:]

    raw_features = np.stack(
        [
            x_signed,        # x_abs
            x_rel,           # x_rel_leader
            v_mag,           # v (always >= 0)
            a_signed,        # a
            s,               # s
            dv,              # dv
            da,              # da
            time_headway,    # time_headway
        ],
        axis=-1,
    )

    resampled = _resample_to_target_hz(raw_features, source_hz=fr, target_hz=config.target_hz)
    return resampled.astype(np.float32, copy=False), vehicle_lengths.astype(np.float32, copy=False)


def slice_windows(
    features: np.ndarray,
    vehicle_lengths: np.ndarray,
    segment: PlatoonSegment,
    config: WindowingConfig,
) -> list[PlatoonSample]:
    """Cut ``features`` into history/future windows and emit ``PlatoonSample``s."""

    total_steps = config.total_steps
    if features.shape[0] < total_steps:
        return []
    if features.shape[0] / config.target_hz < config.minimum_total_seconds:
        return []

    finite = np.isfinite(features)
    samples: list[PlatoonSample] = []
    stride = config.stride_steps
    history = config.history_steps

    for start in range(0, features.shape[0] - total_steps + 1, stride):
        h_slice = features[start : start + history]
        f_slice = features[start + history : start + total_steps]
        if not np.isfinite(h_slice[..., :4]).all() or not np.isfinite(f_slice[..., :4]).all():
            continue
        sample = PlatoonSample(
            recording_id=segment.recording_id,
            lane_id=segment.lane_id,
            track_ids=segment.track_ids,
            history=h_slice.astype(np.float32, copy=False),
            future=f_slice.astype(np.float32, copy=False),
            history_mask=finite[start : start + history].astype(np.float32, copy=False),
            future_mask=finite[start + history : start + total_steps].astype(np.float32, copy=False),
            vehicle_lengths=vehicle_lengths.astype(np.float32, copy=False),
            start_time=float(start) / config.target_hz,
            target_hz=config.target_hz,
        )
        samples.append(sample)
    return samples


def stack_samples(samples: Sequence[PlatoonSample]) -> dict[str, np.ndarray]:
    """Stack a list of ``PlatoonSample`` into batched arrays for serialisation."""

    if not samples:
        raise ValueError("Cannot stack an empty sample list")
    n_track_ids = {len(s.track_ids) for s in samples}
    if len(n_track_ids) != 1:
        raise ValueError(f"Inconsistent platoon size in samples: {n_track_ids}")

    history = np.stack([s.history for s in samples], axis=0)
    future = np.stack([s.future for s in samples], axis=0)
    history_mask = np.stack([s.history_mask for s in samples], axis=0)
    future_mask = np.stack([s.future_mask for s in samples], axis=0)
    track_ids = np.asarray([s.track_ids for s in samples], dtype=np.int64)
    vehicle_lengths = np.stack([s.vehicle_lengths for s in samples], axis=0)
    recording_ids = np.asarray([s.recording_id for s in samples], dtype=np.int64)
    lane_ids = np.asarray([s.lane_id for s in samples], dtype=np.int64)
    start_times = np.asarray([s.start_time for s in samples], dtype=np.float32)
    return {
        "history": history,
        "future": future,
        "history_mask": history_mask,
        "future_mask": future_mask,
        "track_ids": track_ids,
        "vehicle_lengths": vehicle_lengths,
        "recording_ids": recording_ids,
        "lane_ids": lane_ids,
        "start_times": start_times,
        "feature_names": np.asarray(FEATURE_NAMES, dtype=object),
    }
