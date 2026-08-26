"""SSP-DMGTimeNet: top-level model wiring for scheme C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from .cross_cfe import CrossVehicleCFE, CrossVehicleCFEConfig
from .hgf import HGFConfig, HierarchicalGatedFusion, MultiScaleConfig, MultiScaleTimeTokens
from .heads import PlatoonForecastHead, PlatoonForecastHeadConfig
from .sp_daca import SequentialPropagationDACA, SequentialPropagationDACAConfig


@dataclass(slots=True, frozen=True)
class SSPDMGTimeNetConfig:
    num_vehicles: int = 5
    num_features_in: int = 8
    target_hz: float = 10.0
    history_steps: int = 50
    predict_steps: int = 30
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ffn_dim: int = 192
    dropout: float = 0.1
    scales_seconds: tuple[float, ...] = (0.4, 0.8, 1.6, 2.4)
    output_channels: tuple[str, ...] = ("v", "s", "a", "x_rel_leader")
    cfe_v_index: int = 2
    cfe_a_index: int = 3
    cfe_s_index: int = 4
    sp_daca: dict = field(default_factory=lambda: {})
    # Ablation switches. With use_hgf=False the multi-scale tokens are fused
    # by a uniform mean (isolates the learned gating); with use_cfe=False the
    # cointegration branch is removed entirely (zero residuals, all-False
    # residual mask, no CFE token added to the stream).
    use_hgf: bool = True
    use_cfe: bool = True


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SPDACABlock(nn.Module):
    """Residual block: PreNorm -> SP-DACA -> Add -> PreNorm -> FFN -> Add."""

    def __init__(self, sp_daca: SequentialPropagationDACA, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(sp_daca.config.d_model)
        self.ffn_norm = nn.LayerNorm(sp_daca.config.d_model)
        self.attn = sp_daca
        self.ffn = _FeedForward(sp_daca.config.d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        a, diag = self.attn(self.attn_norm(x))
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, diag


class SSPDMGTimeNet(nn.Module):
    """End-to-end platoon forecaster.

    Forward signature returns a dict with the following entries:

    * ``predictions``: ``(B, T_fut, N, D_out)`` un-normalised predictions in
      physical units (m/s, m, m/s^2, m).
    * ``predictions_norm``: the same tensor before un-normalisation (z-scored).
    * ``cfe_residuals``: ``(B, T_hist, N, 3)`` cross-vehicle residuals.
    * ``cfe_residual_mask``: ``(N,)`` boolean mask (False for the leader).
    * ``hgf_weights``: ``(B, T_hist, N, num_scales)`` gating weights.
    * ``layer_diagnostics``: list of per-layer diagnostics from SP-DACA.
    """

    def __init__(self, config: SSPDMGTimeNetConfig) -> None:
        super().__init__()
        self.config = config

        # Normalisation buffers; populated via ``set_normalisation``.
        self.register_buffer("input_mean", torch.zeros(config.num_features_in), persistent=True)
        self.register_buffer("input_std", torch.ones(config.num_features_in), persistent=True)

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
        self.hgf: HierarchicalGatedFusion | None = None
        if config.use_hgf:
            self.hgf = HierarchicalGatedFusion(
                HGFConfig(
                    d_model=config.d_model,
                    num_scales=len(config.scales_seconds),
                    num_vehicles=config.num_vehicles,
                    dropout=config.dropout,
                )
            )

        self.cross_cfe: CrossVehicleCFE | None = None
        if config.use_cfe:
            self.cross_cfe = CrossVehicleCFE(
                CrossVehicleCFEConfig(
                    d_model=config.d_model,
                    num_vehicles=config.num_vehicles,
                    raw_v_index=config.cfe_v_index,
                    raw_a_index=config.cfe_a_index,
                    raw_s_index=config.cfe_s_index,
                    dropout=config.dropout,
                )
            )

        sp_daca_kwargs = dict(config.sp_daca) if config.sp_daca else {}
        sp_daca_kwargs.setdefault("d_model", config.d_model)
        sp_daca_kwargs.setdefault("num_heads", config.num_heads)
        sp_daca_kwargs.setdefault("num_vehicles", config.num_vehicles)
        sp_daca_kwargs.setdefault("target_hz", config.target_hz)
        sp_daca_kwargs.setdefault("dropout", config.dropout)
        self.sp_daca_blocks = nn.ModuleList(
            [
                _SPDACABlock(
                    SequentialPropagationDACA(SequentialPropagationDACAConfig(**sp_daca_kwargs)),
                    ffn_dim=config.ffn_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.head = PlatoonForecastHead(
            PlatoonForecastHeadConfig(
                d_model=config.d_model,
                history_steps=config.history_steps,
                predict_steps=config.predict_steps,
                num_vehicles=config.num_vehicles,
                output_channels=len(config.output_channels),
                dropout=config.dropout,
            )
        )

        # Output un-normalisation statistics; populated via ``set_normalisation``.
        self.register_buffer("output_mean", torch.zeros(len(config.output_channels)), persistent=True)
        self.register_buffer("output_std", torch.ones(len(config.output_channels)), persistent=True)

    def set_normalisation(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        if input_mean.shape != self.input_mean.shape:
            raise ValueError(f"Expected input_mean shape {self.input_mean.shape}, got {input_mean.shape}")
        if input_std.shape != self.input_std.shape:
            raise ValueError(f"Expected input_std shape {self.input_std.shape}, got {input_std.shape}")
        if output_mean.shape != self.output_mean.shape:
            raise ValueError(f"Expected output_mean shape {self.output_mean.shape}, got {output_mean.shape}")
        if output_std.shape != self.output_std.shape:
            raise ValueError(f"Expected output_std shape {self.output_std.shape}, got {output_std.shape}")
        self.input_mean.copy_(input_mean.to(self.input_mean.dtype))
        self.input_std.copy_(input_std.to(self.input_std.dtype))
        self.output_mean.copy_(output_mean.to(self.output_mean.dtype))
        self.output_std.copy_(output_std.to(self.output_std.dtype))

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.input_mean.view(1, 1, 1, -1)
        std = self.input_std.view(1, 1, 1, -1)
        return (x - mean) / std

    def denormalise_output(self, y_norm: torch.Tensor) -> torch.Tensor:
        mean = self.output_mean.view(1, 1, 1, -1)
        std = self.output_std.view(1, 1, 1, -1)
        return y_norm * std + mean

    def forward(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if history_raw.dim() != 4:
            raise ValueError(f"history_raw must be (B, T, N, F_raw), got {history_raw.shape}")
        if history_raw.shape != history_mask.shape:
            raise ValueError("history_mask must match history_raw shape")
        if history_raw.shape[1] != self.config.history_steps:
            raise ValueError(f"history_steps mismatch: expected {self.config.history_steps}, got {history_raw.shape[1]}")
        if history_raw.shape[2] != self.config.num_vehicles:
            raise ValueError(f"num_vehicles mismatch: expected {self.config.num_vehicles}, got {history_raw.shape[2]}")
        if history_raw.shape[3] != self.config.num_features_in:
            raise ValueError(f"num_features_in mismatch: expected {self.config.num_features_in}, got {history_raw.shape[3]}")

        # Replace masked-out (NaN) entries with zero AFTER normalisation; the
        # mask is concatenated as auxiliary information so the network can
        # detect it.
        history_filled = torch.where(history_mask > 0, history_raw, torch.zeros_like(history_raw))
        normed = self.normalise(history_filled)
        normed = torch.where(history_mask > 0, normed, torch.zeros_like(normed))
        x = self.input_proj(normed)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        scale_tokens = self.multiscale(x)
        if self.hgf is not None:
            fused, hgf_weights = self.hgf(scale_tokens)
        else:
            stacked = torch.stack(list(scale_tokens), dim=-1)
            fused = stacked.mean(dim=-1)
            num_scales = len(scale_tokens)
            hgf_weights = torch.full(
                (*fused.shape[:3], num_scales),
                1.0 / num_scales,
                device=fused.device,
                dtype=fused.dtype,
            )

        if self.cross_cfe is not None:
            cfe_token, cfe_residuals, cfe_mask = self.cross_cfe(history_filled)
            x = fused + cfe_token
        else:
            B_, T_, N_, _ = history_filled.shape
            cfe_residuals = torch.zeros(B_, T_, N_, 3, device=fused.device, dtype=fused.dtype)
            cfe_mask = torch.zeros(N_, dtype=torch.bool, device=fused.device)
            x = fused

        layer_diagnostics: list[dict[str, torch.Tensor]] = []
        for block in self.sp_daca_blocks:
            x, diag = block(x)
            layer_diagnostics.append(diag)

        x = self.encoder_norm(x)
        predictions_norm = self.head(x)
        predictions = self.denormalise_output(predictions_norm)
        return {
            "predictions": predictions,
            "predictions_norm": predictions_norm,
            "cfe_residuals": cfe_residuals,
            "cfe_residual_mask": cfe_mask,
            "hgf_weights": hgf_weights,
            "layer_diagnostics": layer_diagnostics,
        }
