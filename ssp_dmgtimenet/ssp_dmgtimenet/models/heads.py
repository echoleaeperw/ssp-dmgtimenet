"""Multi-task forecast heads for SSP-DMGTimeNet.

The decoder maps the encoder output ``(B, T_hist, N, D)`` to a future window
``(B, T_fut, N, D_out)`` with ``D_out`` containing four channels:
``v, s, a, x_rel``. We use a temporal MLP that consumes the last few
encoder steps plus learnable per-vehicle "future" tokens, similar to the
DMGTimeNet decoder but extended over the platoon dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class PlatoonForecastHeadConfig:
    d_model: int = 96
    history_steps: int = 50
    predict_steps: int = 30
    num_vehicles: int = 5
    output_channels: int = 4
    pool_window: int = 8
    hidden_dim: int = 192
    dropout: float = 0.1


class PlatoonForecastHead(nn.Module):
    def __init__(self, config: PlatoonForecastHeadConfig) -> None:
        super().__init__()
        if config.pool_window <= 0 or config.pool_window > config.history_steps:
            raise ValueError("pool_window must be in (0, history_steps]")
        self.config = config

        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, config.num_vehicles, config.d_model, dtype=torch.float32)
        )
        nn.init.normal_(self.future_query, std=0.02)

        in_dim = config.d_model * 2  # context pooling + future query
        self.decoder_mlp = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_channels),
        )
        self.context_norm = nn.LayerNorm(config.d_model)
        self.future_norm = nn.LayerNorm(config.d_model)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.dim() != 4:
            raise ValueError(f"Head expects (B, T_hist, N, D), got {encoded.shape}")
        B, T, N, D = encoded.shape
        if T != self.config.history_steps:
            raise ValueError(f"Expected history_steps={self.config.history_steps}, got T={T}")
        if N != self.config.num_vehicles:
            raise ValueError(f"Expected num_vehicles={self.config.num_vehicles}, got N={N}")
        if D != self.config.d_model:
            raise ValueError(f"Expected d_model={self.config.d_model}, got D={D}")

        context_pool = encoded[:, -self.config.pool_window :].mean(dim=1)  # (B, N, D)
        context_pool = self.context_norm(context_pool)
        context = context_pool.unsqueeze(1).expand(B, self.config.predict_steps, N, D)

        future_query = self.future_norm(self.future_query)  # (T_fut, N, D)
        future_query = future_query.unsqueeze(0).expand(B, -1, -1, -1)

        merged = torch.cat([context, future_query], dim=-1)  # (B, T_fut, N, 2D)
        return self.decoder_mlp(merged)
