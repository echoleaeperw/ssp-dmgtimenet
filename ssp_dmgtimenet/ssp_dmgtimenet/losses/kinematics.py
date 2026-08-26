"""Kinematic consistency: x_t+1 - x_t = v dt; v_t+1 - v_t = a dt."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class KinematicsLossConfig:
    target_hz: float = 10.0
    weight_x: float = 1.0
    weight_v: float = 1.0
    weight_s: float = 0.5
    enforce_velocity_nonnegative: bool = True
    speed_lower_bound: float = 0.0


class KinematicsLoss(nn.Module):
    """Penalise inconsistencies between predicted (x_rel, v, a) and the gap.

    When ``output_std`` is set via :meth:`set_normalisation`, each sub-loss
    is divided by the squared standard deviation of the relevant output
    channel (times ``1/dt²`` for rate-based terms).  This ensures the
    back-propagated gradient through the model's denormalisation layer has
    the same magnitude as the normalised prediction-loss gradient.

    Without this scaling the kin gradient at the normalised-output level is
    amplified by ``output_std / dt`` which can be 100–500× larger than the
    prediction-loss gradient and destroys the learned prediction.
    """

    def __init__(self, config: KinematicsLossConfig) -> None:
        super().__init__()
        self.config = config
        self._output_std: torch.Tensor | None = None

    def set_normalisation(self, output_std: torch.Tensor) -> None:
        self._output_std = output_std.clone()

    def forward(
        self,
        predictions: torch.Tensor,
        vehicle_lengths: torch.Tensor,
        target_predictions: torch.Tensor | None = None,
        leader_velocity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predictions.shape[-1] < 4:
            raise ValueError("predictions must contain at least 4 channels: v, s, a, x_rel_leader")
        v = predictions[..., 0]
        s = predictions[..., 1]
        a = predictions[..., 2]
        x_rel = predictions[..., 3]
        dt = 1.0 / float(self.config.target_hz)

        # ---- velocity-position: d(x_rel)/dt = v_leader - v_i
        dx = x_rel[:, 1:, :] - x_rel[:, :-1, :]
        dv = v[:, 1:, :] - v[:, :-1, :]
        v_sg = v.detach()
        expected_dx_dt = v_sg[:, :, [0]] - v_sg
        x_loss = torch.mean((dx / dt - expected_dx_dt[:, 1:, :]) ** 2)

        # ---- acceleration: d(v_mag)/dt = -a_signed
        a_consistency = (dv / dt + a.detach()[:, 1:, :]) ** 2
        v_loss = a_consistency.mean()

        # ---- gap: s_i = x_rel_i - x_rel_{i-1} - L_{i-1}
        if vehicle_lengths.dim() != 2 or vehicle_lengths.shape[1] != predictions.shape[2]:
            raise ValueError("vehicle_lengths must be (B, N)")
        predecessor_lengths = vehicle_lengths[:, :-1].unsqueeze(1)
        gap_pred_from_x = x_rel[:, :, 1:] - x_rel[:, :, :-1] - predecessor_lengths
        gap_diff = (s[:, :, 1:] - gap_pred_from_x) ** 2
        s_loss = gap_diff.mean()

        if self._output_std is not None:
            std = self._output_std.to(predictions.device)
            std_v = std[0].clamp(min=1e-6)
            std_s = std[1].clamp(min=1e-6)
            std_xrel = std[3].clamp(min=1e-6)
            x_loss = x_loss * (dt ** 2) / (std_xrel ** 2)
            v_loss = v_loss * (dt ** 2) / (std_v ** 2)
            s_loss = s_loss / (std_s ** 2)

        # ---- speed lower bound penalty (one-sided)
        if self.config.enforce_velocity_nonnegative:
            below = torch.clamp(self.config.speed_lower_bound - v, min=0.0)
            speed_penalty = (below ** 2).mean()
        else:
            speed_penalty = torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        total = (
            self.config.weight_x * x_loss
            + self.config.weight_v * v_loss
            + self.config.weight_s * s_loss
            + speed_penalty
        )
        return total, {
            "kin_x": x_loss.detach(),
            "kin_v": v_loss.detach(),
            "kin_s": s_loss.detach(),
            "speed_lb": speed_penalty.detach(),
        }
