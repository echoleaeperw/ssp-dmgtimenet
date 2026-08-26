"""Cross-Vehicle Cointegration Feature Extraction (CFE).

Given raw per-vehicle observations of speed, gap and acceleration, we model
*cointegration-inspired equilibrium residuals* of the form

    r_i^v(t) = v_i(t) - alpha_i^v * v_{i-1}(t) - beta_i^v
    r_i^s(t) = s_i(t) - alpha_i^s * s_{i-1}(t) - beta_i^s     (i >= 2 only)
    r_i^a(t) = a_i(t) - alpha_i^a * a_{i-1}(t) - beta_i^a

where ``alpha`` and ``beta`` are learnable per-pair scalars. The leader
(``i = 1``) does not have an upstream vehicle and therefore contributes
zero residuals; we expose this via a ``residual_mask``.

The module returns two things:

* ``token``: a ``(B, T, N, d_model)`` tensor that embeds the residuals back
  into the main feature stream;
* ``residuals``: ``(B, T, N, 3)`` ordered as ``(v, s, a)`` for use inside
  the cointegration loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class CrossVehicleCFEConfig:
    d_model: int = 96
    num_vehicles: int = 5
    raw_v_index: int = 2
    raw_a_index: int = 3
    raw_s_index: int = 4
    project_hidden: int = 96
    init_alpha: float = 1.0
    init_beta: float = 0.0
    bound_alpha: tuple[float, float] = (0.5, 1.5)
    learn_per_vehicle_beta: bool = True
    dropout: float = 0.1


class CrossVehicleCFE(nn.Module):
    def __init__(self, config: CrossVehicleCFEConfig) -> None:
        super().__init__()
        if config.num_vehicles < 2:
            raise ValueError("num_vehicles must be >= 2")
        if config.bound_alpha[0] >= config.bound_alpha[1]:
            raise ValueError("bound_alpha must be a strictly increasing tuple")
        self.config = config

        n_pairs = config.num_vehicles - 1
        init_unit = (config.init_alpha - config.bound_alpha[0]) / (config.bound_alpha[1] - config.bound_alpha[0])
        init_unit = min(max(init_unit, 1e-4), 1 - 1e-4)
        init_logit = float(torch.logit(torch.tensor(init_unit)))
        self.alpha_logits_v = nn.Parameter(torch.full((n_pairs,), init_logit, dtype=torch.float32))
        self.alpha_logits_s = nn.Parameter(torch.full((n_pairs,), init_logit, dtype=torch.float32))
        self.alpha_logits_a = nn.Parameter(torch.full((n_pairs,), init_logit, dtype=torch.float32))

        beta_shape = (n_pairs,)
        self.beta_v = nn.Parameter(torch.full(beta_shape, config.init_beta, dtype=torch.float32))
        self.beta_s = nn.Parameter(torch.full(beta_shape, config.init_beta, dtype=torch.float32))
        self.beta_a = nn.Parameter(torch.full(beta_shape, config.init_beta, dtype=torch.float32))

        self.project = nn.Sequential(
            nn.Linear(3, config.project_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.project_hidden, config.d_model),
        )
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)

    def _alpha(self, logits: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(logits)
        lo, hi = self.config.bound_alpha
        return lo + (hi - lo) * s

    def forward(
        self,
        raw_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if raw_features.dim() != 4:
            raise ValueError(f"CFE expects (B, T, N, F_raw), got {raw_features.shape}")
        B, T, N, F = raw_features.shape
        if N != self.config.num_vehicles:
            raise ValueError(f"Expected N={self.config.num_vehicles}, got {N}")
        if F <= max(self.config.raw_v_index, self.config.raw_a_index, self.config.raw_s_index):
            raise ValueError("raw_features does not contain the required v/s/a indices")

        v = raw_features[..., self.config.raw_v_index]  # (B, T, N)
        a = raw_features[..., self.config.raw_a_index]
        s = raw_features[..., self.config.raw_s_index]

        alpha_v = self._alpha(self.alpha_logits_v)
        alpha_s = self._alpha(self.alpha_logits_s)
        alpha_a = self._alpha(self.alpha_logits_a)

        residual_v = torch.zeros_like(v)
        residual_s = torch.zeros_like(s)
        residual_a = torch.zeros_like(a)
        for i in range(1, N):
            residual_v[:, :, i] = v[:, :, i] - alpha_v[i - 1] * v[:, :, i - 1] - self.beta_v[i - 1]
            residual_s[:, :, i] = s[:, :, i] - alpha_s[i - 1] * s[:, :, i - 1] - self.beta_s[i - 1]
            residual_a[:, :, i] = a[:, :, i] - alpha_a[i - 1] * a[:, :, i - 1] - self.beta_a[i - 1]
        residuals = torch.stack([residual_v, residual_s, residual_a], dim=-1)
        residual_mask = torch.zeros(N, dtype=torch.bool, device=v.device)
        residual_mask[1:] = True

        token = self.project(residuals)  # (B, T, N, d_model)
        token = self.layer_norm(token)
        token = self.residual_dropout(token)
        # Mask the leader residual to zero in the token stream as well.
        token = token * residual_mask.view(1, 1, N, 1)
        return token, residuals, residual_mask
