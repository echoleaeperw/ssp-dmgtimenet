"""DMGTimeNet cascade baseline.

The original DMGTimeNet is a *single-pair* (leader-follower) non-stationary
car-following predictor. To turn it into a platoon-level baseline we
cascade its per-pair forecast along the chain ``C_1 -> C_2 -> ... -> C_N``:

* ``C_1`` (the leader) is rolled out with a constant-velocity assumption,
  matching the IDM/OVM baselines.
* Every other follower ``C_i`` consumes the historical leader-follower
  pair ``(C_{i-1}, C_i)`` and produces ``C_i``'s future prediction with a
  shared 2-vehicle DACA + HGF + CFE block.

This gives a strong neural baseline that uses delay-aware attention but
*without* the platoon-wide chain causality, head-to-tail stability loss,
or cross-vehicle CFE that scheme C introduces.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..models.cross_cfe import CrossVehicleCFE, CrossVehicleCFEConfig
from ..models.hgf import HGFConfig, HierarchicalGatedFusion, MultiScaleConfig, MultiScaleTimeTokens
from ..models.sp_daca import SequentialPropagationDACA, SequentialPropagationDACAConfig
from .common import BaselineConfigBase, FEATURE_INDEX, _NormalisedBackbone


@dataclass(slots=True, frozen=True)
class DMGCascadeConfig(BaselineConfigBase):
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 192
    dropout: float = 0.1
    scales_seconds: tuple[float, ...] = (0.4, 0.8, 1.6)
    leader_mode: str = "constant_velocity"


class _PairBlock(nn.Module):
    """A 2-vehicle SP-DACA encoder used in the DMG cascade."""

    def __init__(self, config: DMGCascadeConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.num_features_in, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.multiscale = MultiScaleTimeTokens(
            MultiScaleConfig(
                target_hz=config.target_hz,
                scales_seconds=config.scales_seconds,
                d_model=config.d_model,
                dropout=config.dropout,
            )
        )
        self.hgf = HierarchicalGatedFusion(
            HGFConfig(
                d_model=config.d_model,
                num_scales=len(config.scales_seconds),
                num_vehicles=2,
                dropout=config.dropout,
            )
        )
        self.cross_cfe = CrossVehicleCFE(
            CrossVehicleCFEConfig(
                d_model=config.d_model,
                num_vehicles=2,
                raw_v_index=FEATURE_INDEX["v"],
                raw_a_index=FEATURE_INDEX["a"],
                raw_s_index=FEATURE_INDEX["s"],
                dropout=config.dropout,
            )
        )
        self.layers = nn.ModuleList(
            [
                SequentialPropagationDACA(
                    SequentialPropagationDACAConfig(
                        d_model=config.d_model,
                        num_heads=config.num_heads,
                        num_vehicles=2,
                        target_hz=config.target_hz,
                        dropout=config.dropout,
                    )
                )
                for _ in range(config.num_layers)
            ]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_layers)]
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(config.d_model) for _ in range(config.num_layers)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.d_model, config.ffn_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.ffn_dim, config.d_model),
                    nn.Dropout(config.dropout),
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(self, pair_normed: torch.Tensor, pair_raw: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(pair_normed)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        scales = self.multiscale(x)
        fused, _ = self.hgf(scales)
        cfe_token, _, _ = self.cross_cfe(pair_raw)
        x = fused + cfe_token
        for attn, ln, fn, ffn in zip(self.layers, self.layer_norms, self.ffn_norms, self.ffns, strict=True):
            attn_out, _ = attn(ln(x))
            x = x + attn_out
            x = x + ffn(fn(x))
        return self.final_norm(x)


class DMGCascade(_NormalisedBackbone):
    def __init__(self, config: DMGCascadeConfig) -> None:
        super().__init__(config)
        self.config: DMGCascadeConfig = config
        if config.num_vehicles < 2:
            raise ValueError("DMGCascade requires at least 2 vehicles")
        self.pair_block = _PairBlock(config)
        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, 2, config.d_model)
        )
        nn.init.normal_(self.future_query, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_output_channels),
        )

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        B, T, N, F = history_raw.shape
        T_fut = self.config.predict_steps
        device = history_raw.device
        dtype = history_raw.dtype
        normed = self.normalise(history_raw) * history_mask

        out = torch.zeros(B, T_fut, N, self.config.num_output_channels, device=device, dtype=dtype)
        leader_v_last = history_raw[:, -1, 0, FEATURE_INDEX["v"]]
        leader_x_rel = torch.zeros(B, T_fut, device=device, dtype=dtype)
        v_idx = FEATURE_INDEX["v"]
        if self.config.leader_mode == "constant_velocity":
            leader_v_future = leader_v_last.unsqueeze(1).expand(B, T_fut)
            leader_a_future = torch.zeros_like(leader_v_future)
        elif self.config.leader_mode == "constant_acceleration":
            leader_a_last = history_raw[:, -1, 0, FEATURE_INDEX["a"]]
            t_grid = torch.arange(1, T_fut + 1, device=device, dtype=dtype) / float(self.config.target_hz)
            leader_v_future = leader_v_last.unsqueeze(1) + leader_a_last.unsqueeze(1) * t_grid.unsqueeze(0)
            leader_a_future = leader_a_last.unsqueeze(1).expand(B, T_fut)
        else:
            raise ValueError(f"Unknown leader_mode {self.config.leader_mode}")
        out[:, :, 0, 0] = leader_v_future
        out[:, :, 0, 2] = leader_a_future
        out[:, :, 0, 3] = leader_x_rel  # x_rel for leader is identically 0

        leader_s_zero = torch.zeros(B, T_fut, device=device, dtype=dtype)
        out[:, :, 0, 1] = leader_s_zero

        for i in range(1, N):
            pair_raw = history_raw[:, :, [i - 1, i], :]
            pair_norm = normed[:, :, [i - 1, i], :]
            encoded = self.pair_block(pair_norm, pair_raw)  # (B, T, 2, D)
            future_q = self.future_query.unsqueeze(0).expand(B, -1, -1, -1)
            last_ctx = encoded[:, -1, :, :].unsqueeze(1).expand_as(future_q)
            merged = torch.cat([last_ctx, future_q], dim=-1)
            decoded = self.decoder(merged)
            decoded = self.denormalise(decoded)
            out[:, :, i, :] = decoded[:, :, 1, :]
        return out
