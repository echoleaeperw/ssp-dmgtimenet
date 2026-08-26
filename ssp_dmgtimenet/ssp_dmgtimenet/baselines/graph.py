"""Full-graph attention baseline.

This baseline drops the chain-causal mask of SP-DACA and uses an
*undirected* full-graph attention over all vehicles at every time step.
It serves as the head-to-head ablation isolating the impact of strict
chain causality and learnable propagation delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import BaselineConfigBase, _NormalisedBackbone


@dataclass(slots=True, frozen=True)
class FullGraphAttentionConfig(BaselineConfigBase):
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 192
    dropout: float = 0.1
    time_causal: bool = True


class _FullGraphLayer(nn.Module):
    def __init__(self, config: FullGraphAttentionConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config
        self.head_dim = config.d_model // config.num_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        H = self.config.num_heads
        Hd = self.head_dim
        normed = self.attn_norm(x)
        qkv = self.qkv(normed).reshape(B, T, N, 3, H, Hd).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, N, d)
        q_flat = q.reshape(B, H, T * N, Hd)
        k_flat = k.reshape(B, H, T * N, Hd)
        v_flat = v.reshape(B, H, T * N, Hd)
        scores = torch.matmul(q_flat, k_flat.transpose(-1, -2)) / (Hd ** 0.5)
        if self.config.time_causal:
            t_idx = torch.arange(T, device=x.device)
            time_mask = t_idx[None, :] <= t_idx[:, None]
            n_idx = torch.arange(N, device=x.device)
            full_mask = time_mask[:, None, :, None].expand(T, N, T, N)
            full_mask = full_mask.reshape(T * N, T * N)
            scores = scores.masked_fill(~full_mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v_flat).reshape(B, H, T, N, Hd)
        out = out.permute(0, 2, 3, 1, 4).reshape(B, T, N, D)
        out = self.proj_drop(self.out_proj(out))
        x = x + out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FullGraphAttention(_NormalisedBackbone):
    def __init__(self, config: FullGraphAttentionConfig) -> None:
        super().__init__(config)
        self.config: FullGraphAttentionConfig = config
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
        self.layers = nn.ModuleList([_FullGraphLayer(config) for _ in range(config.num_layers)])
        self.encoder_norm = nn.LayerNorm(config.d_model)
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
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
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
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        future_q = self.future_query.unsqueeze(0).expand(B, -1, -1, -1)
        last_ctx = x[:, -1, :, :].unsqueeze(1).expand_as(future_q)
        merged = torch.cat([last_ctx, future_q], dim=-1)
        decoded = self.decoder(merged)
        return self.denormalise(decoded)
