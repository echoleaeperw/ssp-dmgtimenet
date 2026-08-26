import math

import numpy as np

from ssp_dmgtimenet.metrics.stability import (
    conditional_internal_amplification_stats,
    disturbance_detection_stats,
    gt_referenced_amplification_stats,
)


def _wave(scale: float, steps: int = 30) -> np.ndarray:
    time = np.arange(steps, dtype=np.float64) / 10.0
    return scale * np.sin(2.0 * np.pi * 0.5 * time)


def test_disturbance_detection_uses_one_gt_definition_for_all_models() -> None:
    gt = np.zeros((4, 30, 3), dtype=np.float64)
    pred = np.zeros_like(gt)
    gt[0, :, 0] = _wave(1.0)
    gt[1, :, 0] = _wave(1.0)
    pred[0, :, 0] = _wave(1.0)
    pred[2, :, 0] = _wave(1.0)

    stats = disturbance_detection_stats(
        pred,
        gt,
        detrend_window_steps=1,
        excitation_floor=0.05,
    )

    assert stats["tp"] == 1
    assert stats["fp"] == 1
    assert stats["fn"] == 1
    assert stats["tn"] == 1
    assert stats["coverage"] == 0.5
    assert stats["fpr"] == 0.5


def test_conditional_internal_gain_uses_predicted_leader_common_denominator() -> None:
    gt = np.zeros((2, 30, 3), dtype=np.float64)
    pred = np.zeros_like(gt)
    gt[:, :, 0] = _wave(1.0)
    pred[0, :, 0] = _wave(1.0)
    pred[0, :, 1] = _wave(0.8)
    pred[0, :, 2] = _wave(1.2)

    stats = conditional_internal_amplification_stats(
        pred,
        gt,
        detrend_window_steps=1,
        excitation_floor=0.05,
    )

    assert stats["n_windows"] == 1
    assert stats["support_ratio"] == 0.5
    assert stats["unstable_window_ratio"] == 1.0
    assert math.isclose(stats["max_gain"], 1.2, rel_tol=1e-5)
    assert math.isclose(stats["C1_to_C2_mean_gain"], 0.8, rel_tol=1e-5)
    assert math.isclose(stats["C1_to_C3_mean_gain"], 1.2, rel_tol=1e-5)


def test_gt_reference_gain_keeps_every_gt_excited_window() -> None:
    gt = np.zeros((2, 30, 3), dtype=np.float64)
    pred = np.zeros_like(gt)
    gt[:, :, 0] = _wave(1.0)
    pred[0, :, 1] = _wave(0.5)
    pred[0, :, 2] = _wave(0.8)
    pred[1, :, 1] = _wave(1.1)
    pred[1, :, 2] = _wave(0.9)

    stats = gt_referenced_amplification_stats(
        pred,
        gt,
        detrend_window_steps=1,
        excitation_floor=0.05,
    )

    assert stats["n_windows"] == 2
    assert stats["unstable_window_ratio"] == 0.5
    assert math.isclose(stats["max_gain"], 1.1, rel_tol=1e-5)
