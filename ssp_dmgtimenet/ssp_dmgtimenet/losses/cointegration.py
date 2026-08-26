"""Cross-vehicle cointegration variance penalty.

Following scheme C §5.3, we penalise the variance of the residual ``r_i``
and the variance of its first difference ``Δr_i``. The first term keeps the
equilibrium tight, while the second penalises drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class CointegrationLossConfig:
    weight_var: float = 1.0
    weight_diff: float = 0.5
    eps: float = 1e-6


class CointegrationLoss(nn.Module):
    def __init__(self, config: CointegrationLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        residuals: torch.Tensor,
        residual_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if residuals.dim() != 4:
            raise ValueError(f"residuals must be (B, T, N, 3), got {residuals.shape}")
        if residual_mask.shape != (residuals.shape[2],):
            raise ValueError(f"residual_mask must be (N,), got {residual_mask.shape}")
        if residual_mask.sum() == 0:
            zero = torch.zeros((), device=residuals.device, dtype=residuals.dtype)
            return zero, {"coint_var": zero, "coint_diff": zero}

        valid = residuals[:, :, residual_mask, :]
        var = valid.var(dim=1, unbiased=False)  # (B, N_valid, 3)
        var_loss = var.mean()
        diff = valid[:, 1:, :, :] - valid[:, :-1, :, :]
        diff_var = diff.var(dim=1, unbiased=False).mean()
        total = self.config.weight_var * var_loss + self.config.weight_diff * diff_var
        return total, {
            "coint_var": var_loss.detach(),
            "coint_diff": diff_var.detach(),
        }
