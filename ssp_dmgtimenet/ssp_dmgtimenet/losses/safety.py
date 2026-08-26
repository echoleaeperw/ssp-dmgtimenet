"""Soft constraints for non-negative gaps and small TTC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class SafetyLossConfig:
    min_gap: float = 0.5
    min_time_headway: float = 0.6
    ttc_threshold: float = 1.5
    weight_gap: float = 1.0
    weight_th: float = 0.5
    weight_ttc: float = 0.5
    eps: float = 1e-3


class SafetyLoss(nn.Module):
    """Hinge penalties on predicted gaps, time-headway and TTC."""

    def __init__(self, config: SafetyLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, predictions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predictions.shape[-1] < 3:
            raise ValueError("predictions must include at least v (0), s (1), a (2)")
        v = predictions[..., 0]
        s = predictions[..., 1]
        # follower indices are 1..N-1 (vehicle 0 is the leader without gap)
        s_follow = s[..., 1:]
        v_follow = v[..., 1:]
        v_lead = v[..., :-1]
        relative_v = v_lead - v_follow  # negative when follower is approaching

        gap_violation = torch.clamp(self.config.min_gap - s_follow, min=0.0)
        gap_loss = (gap_violation ** 2).mean()

        th = s_follow / torch.clamp(v_follow, min=self.config.eps)
        th_violation = torch.clamp(self.config.min_time_headway - th, min=0.0)
        th_loss = (th_violation ** 2).mean()

        closing = torch.clamp(-relative_v, min=self.config.eps)
        ttc = s_follow / closing
        approaching_mask = (relative_v < 0).float()
        ttc_violation = torch.clamp(self.config.ttc_threshold - ttc, min=0.0)
        ttc_loss = ((ttc_violation ** 2) * approaching_mask).sum() / (approaching_mask.sum().clamp_min(1.0))

        total = (
            self.config.weight_gap * gap_loss
            + self.config.weight_th * th_loss
            + self.config.weight_ttc * ttc_loss
        )
        return total, {
            "safety_gap": gap_loss.detach(),
            "safety_th": th_loss.detach(),
            "safety_ttc": ttc_loss.detach(),
        }
