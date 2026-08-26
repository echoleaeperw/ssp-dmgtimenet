"""Differentiable string-stability losses (adjacent / sub-platoon / FFT)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..metrics.stability import (
    adjacent_amplification,
    detrended_velocity,
    fft_gain,
    subplatoon_amplification,
)


@dataclass(slots=True, frozen=True)
class StabilityLossConfig:
    detrend_window_steps: int = 8
    delta: float = 0.0
    eps: float = 1e-6
    target_hz: float = 10.0
    fft_band_hz: tuple[float, float] = (0.05, 0.5)
    excitation_quantile: float = 0.25
    excitation_floor: float = 0.05
    weight_adj: float = 1.0
    weight_sub: float = 1.0
    weight_fft: float = 1.0


class LowExcitationGate:
    """Mask out batch entries whose upstream excitation is below a threshold.

    The threshold is taken as the 25-th percentile of the *true* (non-detrended)
    leader velocity standard deviation observed during training. We accept
    a running estimator-style update so the trainer can keep refreshing the
    quantile from the observed batches.
    """

    def __init__(self, quantile: float = 0.25, floor: float = 0.05) -> None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must be in [0, 1]")
        self.quantile = quantile
        self.floor = floor
        self.threshold: float = floor

    def update(self, std_v_leader: torch.Tensor) -> None:
        if std_v_leader.numel() == 0:
            return
        candidate = float(torch.quantile(std_v_leader.detach().abs(), self.quantile).item())
        self.threshold = max(self.floor, candidate)

    def mask(self, std_v_leader: torch.Tensor) -> torch.Tensor:
        return (std_v_leader >= self.threshold).float()


class AdjacentAmplificationLoss(nn.Module):
    """L_adj from scheme C §5.5.1."""

    def __init__(self, config: StabilityLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        v_pred: torch.Tensor,
        excitation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        amp = adjacent_amplification(
            v_pred,
            self.config.detrend_window_steps,
            eps=self.config.eps,
        )
        if not isinstance(amp, torch.Tensor):
            amp = torch.as_tensor(amp, dtype=v_pred.dtype, device=v_pred.device)
        excess = torch.clamp(amp - 1.0 - self.config.delta, min=0.0)
        squared = excess ** 2
        if excitation_mask is not None:
            mask = excitation_mask.view(amp.shape[0], 1).expand_as(squared)
            denom = mask.sum().clamp_min(1.0)
            loss = (squared * mask).sum() / denom
        else:
            loss = squared.mean()
        return self.config.weight_adj * loss, {
            "adj_loss_raw": loss.detach(),
            "adj_amplification_mean": amp.detach().mean(),
            "adj_amplification_max": amp.detach().max(),
        }


class SubplatoonAmplificationLoss(nn.Module):
    """L_sub from scheme C §5.5.2."""

    def __init__(self, config: StabilityLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        v_pred: torch.Tensor,
        excitation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        amp = subplatoon_amplification(
            v_pred,
            self.config.detrend_window_steps,
            eps=self.config.eps,
        )
        if not isinstance(amp, torch.Tensor):
            amp = torch.as_tensor(amp, dtype=v_pred.dtype, device=v_pred.device)
        # zero entries are j>=i; keep only valid upper-triangular pairs
        N = amp.shape[-1]
        mask_upper = torch.triu(torch.ones(N, N, device=v_pred.device, dtype=v_pred.dtype), diagonal=1)
        valid = amp * mask_upper
        excess = torch.clamp(valid - 1.0 - self.config.delta, min=0.0)
        squared = excess ** 2
        if excitation_mask is not None:
            mask = excitation_mask.view(amp.shape[0], 1, 1).expand_as(squared) * mask_upper
            denom = mask.sum().clamp_min(1.0)
            loss = (squared * mask).sum() / denom
        else:
            denom = mask_upper.sum().clamp_min(1.0) * amp.shape[0]
            loss = squared.sum() / denom
        return self.config.weight_sub * loss, {
            "sub_loss_raw": loss.detach(),
            "sub_amp_mean_above_diag": (valid.sum() / mask_upper.sum().clamp_min(1.0) / amp.shape[0]).detach(),
            "sub_amp_max": valid.max().detach(),
        }


class FFTAmplificationLoss(nn.Module):
    """L_fft from scheme C §5.5.3."""

    def __init__(self, config: StabilityLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        v_pred: torch.Tensor,
        excitation_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gain, freqs = fft_gain(
            v_pred,
            target_hz=self.config.target_hz,
            band=self.config.fft_band_hz,
            detrend_window_steps=self.config.detrend_window_steps,
            eps=self.config.eps,
        )
        if not isinstance(gain, torch.Tensor):
            gain = torch.as_tensor(gain, dtype=v_pred.dtype, device=v_pred.device)
        N = gain.shape[1]
        mask_upper = torch.triu(torch.ones(N, N, device=gain.device, dtype=gain.dtype), diagonal=1)
        valid = gain * mask_upper.unsqueeze(-1)
        excess = torch.clamp(valid - 1.0 - self.config.delta, min=0.0)
        squared = excess ** 2
        if excitation_mask is not None:
            mask = excitation_mask.view(gain.shape[0], 1, 1, 1).expand_as(squared) * mask_upper.unsqueeze(-1)
            denom = mask.sum().clamp_min(1.0)
            loss = (squared * mask).sum() / denom
        else:
            denom = mask_upper.unsqueeze(-1).expand_as(squared).sum().clamp_min(1.0)
            loss = squared.sum() / denom
        return self.config.weight_fft * loss, {
            "fft_loss_raw": loss.detach(),
            "fft_gain_max": valid.max().detach(),
            "fft_gain_mean": (valid.sum() / mask_upper.sum().clamp_min(1.0) / gain.shape[0] / gain.shape[-1]).detach(),
            "fft_n_bins": torch.tensor(freqs.shape[0], dtype=torch.float32),
        }
