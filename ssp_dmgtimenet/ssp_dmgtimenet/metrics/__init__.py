"""Numerical metrics for accuracy, stability and engineering performance."""

from .accuracy import (
    horizon_wise_errors,
    mae_per_variable,
    rmse_per_variable,
    tail_vehicle_error,
    vehicle_wise_errors,
)
from .safety import (
    collision_risk,
    comfort_jerk,
    energy_consumption,
    gap_violation_rate,
    inference_latency,
)
from .stability import (
    adjacent_amplification,
    detrended_velocity,
    fft_band_gain_stats,
    fft_gain,
    leader_excitation_amplitude,
    phase_delay,
    strict_joint_stability_metrics,
    subplatoon_amplification,
    unstable_window_metrics,
    vehicle_excitation_amplitude,
)

__all__ = [
    "horizon_wise_errors",
    "mae_per_variable",
    "rmse_per_variable",
    "tail_vehicle_error",
    "vehicle_wise_errors",
    "adjacent_amplification",
    "detrended_velocity",
    "fft_band_gain_stats",
    "fft_gain",
    "leader_excitation_amplitude",
    "phase_delay",
    "strict_joint_stability_metrics",
    "subplatoon_amplification",
    "unstable_window_metrics",
    "vehicle_excitation_amplitude",
    "collision_risk",
    "comfort_jerk",
    "gap_violation_rate",
    "inference_latency",
    "energy_consumption",
]
