"""Recurrent baselines for platoon prediction.

* :class:`PlatoonLSTM` / :class:`PlatoonGRU` -- flatten the platoon dimension
  into the feature vector and run a single sequence-to-sequence recurrent
  encoder; the decoder is an MLP that consumes the last hidden state plus a
  learnable per-future-step query token. This is the standard RNN baseline
  used in trajectory prediction work.
* :class:`InteractionLSTM` -- per-vehicle LSTM with shared weights whose
  inputs include explicit interaction features with the predecessor (gap,
  relative speed, time headway), following the "social"/"interaction"
  baselines in the trajectory prediction literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import (
    BaselineConfigBase,
    FEATURE_INDEX,
    _NormalisedBackbone,
)


@dataclass(slots=True, frozen=True)
class PlatoonLSTMConfig(BaselineConfigBase):
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    rnn_type: str = "lstm"


class _RecurrentBackbone(_NormalisedBackbone):
    def __init__(self, config: PlatoonLSTMConfig) -> None:
        super().__init__(config)
        self.config: PlatoonLSTMConfig = config
        in_dim = config.num_vehicles * config.num_features_in
        if config.rnn_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=in_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
        elif config.rnn_type == "gru":
            self.encoder = nn.GRU(
                input_size=in_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown rnn_type {config.rnn_type!r}")

        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, config.hidden_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.future_query, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_vehicles * config.num_output_channels),
        )

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        B, T, N, F = history_raw.shape
        normed = self.normalise(history_raw) * history_mask
        flat = normed.reshape(B, T, N * F)
        encoded, hidden = self.encoder(flat)
        last_hidden = encoded[:, -1, :]  # (B, H)
        future = self.future_query.unsqueeze(0).expand(B, -1, -1)  # (B, T_fut, H)
        merged = torch.cat([last_hidden.unsqueeze(1).expand_as(future), future], dim=-1)
        decoded = self.decoder(merged)  # (B, T_fut, N*D_out)
        decoded = decoded.reshape(B, self.config.predict_steps, N, self.config.num_output_channels)
        return self.denormalise(decoded)


class PlatoonLSTM(_RecurrentBackbone):
    pass


@dataclass(slots=True, frozen=True)
class PlatoonGRUConfig(PlatoonLSTMConfig):
    rnn_type: str = "gru"


class PlatoonGRU(_RecurrentBackbone):
    def __init__(self, config: PlatoonGRUConfig) -> None:
        super().__init__(config)


@dataclass(slots=True, frozen=True)
class InteractionLSTMConfig(BaselineConfigBase):
    hidden_dim: int = 96
    num_layers: int = 2
    dropout: float = 0.1
    interaction_features: tuple[str, ...] = ("v", "s", "dv", "time_headway")


class InteractionLSTM(_NormalisedBackbone):
    """Per-vehicle LSTM with interaction features (Int-LSTM style)."""

    def __init__(self, config: InteractionLSTMConfig) -> None:
        super().__init__(config)
        self.config: InteractionLSTMConfig = config
        for name in config.interaction_features:
            if name not in FEATURE_INDEX:
                raise ValueError(f"Interaction feature {name!r} not in FEATURE_INDEX")
        in_dim = config.num_features_in + len(config.interaction_features)
        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, config.num_vehicles, config.hidden_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.future_query, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_output_channels),
        )

    def _interaction_block(self, normed: torch.Tensor) -> torch.Tensor:
        feats: list[torch.Tensor] = []
        for name in self.config.interaction_features:
            idx = FEATURE_INDEX[name]
            full = normed[..., idx]
            shifted = torch.zeros_like(full)
            shifted[..., 1:] = full[..., :-1]
            feats.append(shifted.unsqueeze(-1))
        return torch.cat(feats, dim=-1)  # (B, T, N, K)

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        B, T, N, F = history_raw.shape
        normed = self.normalise(history_raw) * history_mask
        interactions = self._interaction_block(normed)
        combined = torch.cat([normed, interactions], dim=-1)  # (B, T, N, F + K)
        per_vehicle = combined.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        encoded, _ = self.encoder(per_vehicle)
        last_hidden = encoded[:, -1, :].reshape(B, N, -1)  # (B, N, H)
        future = self.future_query.unsqueeze(0).expand(B, -1, -1, -1)  # (B, T_fut, N, H)
        last_expand = last_hidden.unsqueeze(1).expand(-1, self.config.predict_steps, -1, -1)
        merged = torch.cat([last_expand, future], dim=-1)
        decoded = self.decoder(merged)
        return self.denormalise(decoded)
