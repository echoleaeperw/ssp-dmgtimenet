import numpy as np

from ssp_dmgtimenet.metrics.stability import strict_joint_stability_metrics


def _scaled_windows(scales: list[list[float]], steps: int = 80) -> np.ndarray:
    time = np.arange(steps, dtype=np.float64) / 10.0
    base = np.sin(2.0 * np.pi * 0.25 * time) + 0.4 * np.sin(2.0 * np.pi * 0.5 * time)
    return np.asarray([[factor * base for factor in window] for window in scales]).transpose(
        0, 2, 1
    )


def test_strict_joint_stability_separates_time_frequency_and_union() -> None:
    v_time = _scaled_windows(
        [
            [1.0, 0.8, 0.6],
            [1.0, 0.8, 1.2],
            [1.0, 0.8, 0.6],
        ]
    )
    v_frequency = _scaled_windows(
        [
            [1.0, 0.8, 0.6],
            [1.0, 0.8, 0.6],
            [1.0, 0.8, 1.2],
        ]
    )

    report = strict_joint_stability_metrics(
        v_time,
        detrend_window_steps=8,
        target_hz=10.0,
        band=(0.1, 1.0),
        excitation_floor=0.01,
        v_frequency=v_frequency,
        n_fft=128,
    )

    assert report.time_unstable_window_ratio == 1.0 / 3.0
    assert report.frequency_unstable_window_ratio == 1.0 / 3.0
    assert report.joint_unstable_window_ratio == 2.0 / 3.0
    assert report.excitation_n_retained == 3
    assert report.max_time_amplification > 1.0
    assert report.max_frequency_gain > 1.0


def test_strict_joint_stability_uses_reference_excitation_support() -> None:
    v = _scaled_windows([[1.0, 0.8, 0.6], [1.0, 0.8, 0.6]])
    reference = v.copy()
    reference[1] = 0.0

    report = strict_joint_stability_metrics(
        v,
        detrend_window_steps=8,
        target_hz=10.0,
        band=(0.1, 1.0),
        excitation_floor=0.01,
        floor_reference_v=reference,
        n_fft=128,
    )

    assert report.excitation_n_retained == 1
    assert report.excitation_retained_ratio == 0.5
    assert report.joint_unstable_window_ratio == 0.0
