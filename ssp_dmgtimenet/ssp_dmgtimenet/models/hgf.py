"""Multi-scale time-token extraction and Hierarchical Gated Fusion (HGF).

The temporal scales are specified in *seconds*; the module converts them to
frame counts at the configured ``target_hz``. We keep the time axis length
unchanged so that downstream blocks can stack scales freely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class MultiScaleConfig:
    target_hz: float = 10.0
    scales_seconds: tuple[float, ...] = (0.4, 0.8, 1.6, 2.4)
    d_model: int = 96
    use_dilation: bool = True
    dropout: float = 0.1


class MultiScaleTimeTokens(nn.Module):
    """A bank of depth-wise 1D convolutions applied per vehicle.

    Input shape is ``(B, T, N, D)`` and the output is a list of M tensors of
    the same shape, one per scale. Each scale uses a kernel covering
    ``round(scale_seconds * target_hz)`` frames; if ``use_dilation`` is true
    we keep the kernel size at 3 and use dilation to span the same receptive
    field, which is cheaper for long scales.
    """

    def __init__(self, config: MultiScaleConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Conv1d] = []
        for s in config.scales_seconds:
            kernel_total = max(1, int(round(s * config.target_hz)))
            if config.use_dilation:
                kernel_size = 3
                dilation = max(1, (kernel_total - 1) // (kernel_size - 1)) if kernel_total > 1 else 1
                effective = (kernel_size - 1) * dilation + 1
                padding = effective // 2
            else:
                kernel_size = kernel_total
                if kernel_size % 2 == 0:
                    kernel_size += 1
                dilation = 1
                padding = kernel_size // 2
            layers.append(
                nn.Conv1d(
                    in_channels=config.d_model,
                    out_channels=config.d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    groups=config.d_model,  # depth-wise to keep cost small
                )
            )
        self.depthwise_convs = nn.ModuleList(layers)
        self.point_convs = nn.ModuleList(
            [nn.Conv1d(config.d_model, config.d_model, kernel_size=1) for _ in config.scales_seconds]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in config.scales_seconds])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    @property
    def num_scales(self) -> int:
        return len(self.config.scales_seconds)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"MultiScaleTimeTokens expects (B, T, N, D), got {x.shape}")
        B, T, N, D = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, D, T)
        outputs: list[torch.Tensor] = []
        for dwc, pwc, ln in zip(self.depthwise_convs, self.point_convs, self.norms, strict=True):
            h = dwc(x_flat)
            h = pwc(h)
            h = h.reshape(B, N, D, T).permute(0, 3, 1, 2)
            h = ln(h)
            h = self.activation(h)
            h = self.dropout(h)
            outputs.append(h)
        return outputs


@dataclass(slots=True, frozen=True)
class HGFConfig:
    d_model: int = 96
    num_scales: int = 4
    num_vehicles: int = 5
    gate_hidden: int = 64
    use_position_aware_gate: bool = True
    dropout: float = 0.1


class HierarchicalGatedFusion(nn.Module):
    """Position-aware softmax gating that mixes multi-scale tokens."""

    def __init__(self, config: HGFConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.d_model + (config.num_vehicles if config.use_position_aware_gate else 0)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, config.gate_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden, config.num_scales),
        )
        if config.use_position_aware_gate:
            self.register_buffer("vehicle_one_hot", torch.eye(config.num_vehicles), persistent=False)
        self.fuse_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, scale_tokens: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(scale_tokens) != self.config.num_scales:
            raise ValueError(
                f"HGF expected {self.config.num_scales} scales, got {len(scale_tokens)}"
            )
        first = scale_tokens[0]
        if first.dim() != 4:
            raise ValueError(f"Each scale must be (B, T, N, D), got {first.shape}")
        B, T, N, D = first.shape
        if N != self.config.num_vehicles:
            raise ValueError(f"Expected N={self.config.num_vehicles}, got {N}")
        if D != self.config.d_model:
            raise ValueError(f"Expected D={self.config.d_model}, got {D}")
        for s in scale_tokens:
            if s.shape != first.shape:
                raise ValueError("All scales must share the same shape")
        stacked = torch.stack(list(scale_tokens), dim=-1)  # (B, T, N, D, M)
        gate_input = scale_tokens[0]
        if self.config.use_position_aware_gate:
            one_hot = self.vehicle_one_hot[None, None, :, :].expand(B, T, N, N)
            gate_input = torch.cat([gate_input, one_hot], dim=-1)
        logits = self.gate_mlp(gate_input)  # (B, T, N, M)
        weights = torch.softmax(logits, dim=-1)
        fused = (stacked * weights[..., None, :]).sum(dim=-1)
        fused = self.fuse_norm(fused)
        fused = self.dropout(fused)
        return fused, weights
