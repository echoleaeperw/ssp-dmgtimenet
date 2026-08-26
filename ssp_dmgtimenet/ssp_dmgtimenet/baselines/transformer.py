"""Transformer baseline for platoon prediction.

A vanilla encoder-only transformer that flattens the (T, N) grid into a
single sequence of length ``T * N`` with sinusoidal time embeddings and
learnable per-vehicle position embeddings. The decoder side is a small
MLP fed by a per-step / per-vehicle learnable query.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import BaselineConfigBase, _NormalisedBackbone


@dataclass(slots=True, frozen=True)
class PlatoonTransformerConfig(BaselineConfigBase):
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 192
    dropout: float = 0.1


class PlatoonTransformer(_NormalisedBackbone):
    def __init__(self, config: PlatoonTransformerConfig) -> None:
        super().__init__(config)
        self.config: PlatoonTransformerConfig = config
        self.input_proj = nn.Linear(config.num_features_in, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.vehicle_pos = nn.Parameter(torch.zeros(config.num_vehicles, config.d_model))
        nn.init.normal_(self.vehicle_pos, std=0.02)
        self.register_buffer(
            "time_pos",
            self._sinusoidal_time_encoding(config.history_steps, config.d_model),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, config.num_vehicles, config.d_model)
        )
        nn.init.normal_(self.future_query, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_output_channels),
        )

    @staticmethod
    def _sinusoidal_time_encoding(T: int, d_model: int) -> torch.Tensor:
        position = torch.arange(T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(T, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        B, T, N, F = history_raw.shape
        normed = self.normalise(history_raw) * history_mask
        x = self.input_proj(normed)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        x = x + self.vehicle_pos.view(1, 1, N, -1) + self.time_pos.view(1, T, 1, -1)
        x = x.reshape(B, T * N, -1)
        encoded = self.encoder(x)
        encoded = encoded.reshape(B, T, N, -1)

        future_q = self.future_query.unsqueeze(0).expand(B, -1, -1, -1)
        last_ctx = encoded[:, -1, :, :].unsqueeze(1).expand_as(future_q)
        merged = torch.cat([last_ctx, future_q], dim=-1)
        decoded = self.decoder(merged)
        return self.denormalise(decoded)
