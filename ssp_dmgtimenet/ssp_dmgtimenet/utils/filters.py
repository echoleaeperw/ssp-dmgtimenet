"""Signal smoothing helpers used during HighD/NGSIM resampling."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def savgol_smooth(x: np.ndarray, window_size: int, polyorder: int = 3, axis: int = -1) -> np.ndarray:
    """Savitzky-Golay smoothing with safe handling of short sequences.

    The ``scipy.signal.savgol_filter`` requires an odd window strictly larger
    than ``polyorder``. When a sequence is shorter than the requested window
    we shrink the window to the closest odd value that fits while remaining
    larger than ``polyorder``; if the sequence is shorter than that minimum we
    return the input unchanged (the caller is expected to drop such sequences
    earlier).
    """

    if window_size <= polyorder:
        raise ValueError("window_size must be greater than polyorder")
    if window_size % 2 == 0:
        window_size += 1
    length = x.shape[axis]
    if length < window_size:
        max_odd = length if length % 2 == 1 else length - 1
        if max_odd <= polyorder:
            return x.copy()
        window_size = max_odd
    return savgol_filter(x, window_length=window_size, polyorder=polyorder, axis=axis, mode="interp")


def central_difference(x: np.ndarray, dt: float, axis: int = -1) -> np.ndarray:
    """Second-order central difference along ``axis`` with edge fall-back."""

    return np.gradient(x, dt, axis=axis, edge_order=2)


def low_pass_filter(
    x: np.ndarray,
    fs: float,
    cutoff_hz: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter via ``filtfilt``.

    Used when down-sampling 25 Hz HighD signals to 10 Hz so that no aliased
    high-frequency content survives the down-sample.
    """

    if cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        raise ValueError(f"cutoff_hz must be in (0, fs/2). Got cutoff={cutoff_hz}, fs={fs}")
    nyq = 0.5 * fs
    normalised = cutoff_hz / nyq
    b, a = butter(order, normalised, btype="low", analog=False)
    pad = max(3 * max(len(a), len(b)), 9)
    if x.shape[axis] <= pad:
        return x.copy()
    return filtfilt(b, a, x, axis=axis, method="pad")
