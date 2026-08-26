"""CNN-Int-LSTM-IDM hybrid baseline.

This is our re-implementation of the Physica A 2025 model
"CNN-Int-LSTM-IDM" referenced in scheme C §2.2. The original paper combines
a 1D-CNN feature extractor, an interaction-aware LSTM and an IDM physical
prior. We adopt the same recipe:

* 1D-CNN over time, applied per vehicle with shared weights, extracts local
  temporal features.
* An interaction LSTM consumes the CNN features together with the
  predecessor's last hidden state, mirroring the *Int-LSTM* interaction
  channel.
* A small MLP produces a per-step *residual* acceleration.
* IDM rolls the platoon out using learnable parameters, and the final
  acceleration is ``a_IDM + a_residual``. ``v``, ``s`` and ``x_rel`` are
  obtained by Euler integration with a non-negativity clamp on speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import BaselineConfigBase, FEATURE_INDEX, _NormalisedBackbone
from .physics import _bounded, _invert


@dataclass(slots=True, frozen=True)
class CNNIntLSTMIDMConfig(BaselineConfigBase):
    cnn_channels: int = 64
    cnn_kernel: int = 5
    cnn_layers: int = 2
    lstm_hidden: int = 96
    lstm_layers: int = 2
    dropout: float = 0.1
    init_v0: float = 30.0
    init_T: float = 1.5
    init_a_max: float = 1.5
    init_b: float = 2.0
    init_s0: float = 2.0
    delta: float = 4.0
    bound_v0: tuple[float, float] = (10.0, 45.0)
    bound_T: tuple[float, float] = (0.4, 3.0)
    bound_a_max: tuple[float, float] = (0.3, 4.0)
    bound_b: tuple[float, float] = (0.3, 4.0)
    bound_s0: tuple[float, float] = (0.5, 5.0)
    speed_clamp_lower: float = 0.0
    leader_mode: str = "constant_velocity"
    interaction_features: tuple[str, ...] = ("v", "s", "dv")
    # Physical saturation of the commanded acceleration (IDM prior + learned
    # residual): close-cut-in initial gaps in HighD blow the raw IDM term up
    # to |a| ~ 1e4 m/s^2, which no vehicle can realise.
    accel_clamp: tuple[float, float] = (-8.0, 4.0)


class _PerVehicleCNN(nn.Module):
    def __init__(self, config: CNNIntLSTMIDMConfig) -> None:
        super().__init__()
        in_dim = config.num_features_in
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(config.cnn_layers):
            layers.append(
                nn.Conv1d(prev, config.cnn_channels, kernel_size=config.cnn_kernel, padding=config.cnn_kernel // 2)
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            prev = config.cnn_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, T, F) -> conv expects (B*N, F, T)
        B, N, T, F = x.shape
        flat = x.reshape(B * N, T, F).transpose(1, 2)
        out = self.net(flat).transpose(1, 2)
        return out.reshape(B, N, T, -1)


class CNNIntLSTMIDM(_NormalisedBackbone):
    def __init__(self, config: CNNIntLSTMIDMConfig) -> None:
        super().__init__(config)
        self.config: CNNIntLSTMIDMConfig = config
        self.cnn = _PerVehicleCNN(config)
        for name in config.interaction_features:
            if name not in FEATURE_INDEX:
                raise ValueError(f"Interaction feature {name!r} missing from FEATURE_INDEX")
        in_dim = config.cnn_channels + len(config.interaction_features)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.future_query = nn.Parameter(
            torch.zeros(config.predict_steps, config.lstm_hidden)
        )
        nn.init.normal_(self.future_query, std=0.02)
        self.residual_mlp = nn.Sequential(
            nn.Linear(config.lstm_hidden * 2, config.lstm_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden, 1),
        )

        self.logit_v0 = nn.Parameter(torch.tensor(_invert(config.init_v0, *config.bound_v0)))
        self.logit_T = nn.Parameter(torch.tensor(_invert(config.init_T, *config.bound_T)))
        self.logit_a_max = nn.Parameter(torch.tensor(_invert(config.init_a_max, *config.bound_a_max)))
        self.logit_b = nn.Parameter(torch.tensor(_invert(config.init_b, *config.bound_b)))
        self.logit_s0 = nn.Parameter(torch.tensor(_invert(config.init_s0, *config.bound_s0)))

        self.dt = 1.0 / float(config.target_hz)

    @property
    def v0(self) -> torch.Tensor:
        return _bounded(self.logit_v0, *self.config.bound_v0)

    @property
    def T(self) -> torch.Tensor:
        return _bounded(self.logit_T, *self.config.bound_T)

    @property
    def a_max(self) -> torch.Tensor:
        return _bounded(self.logit_a_max, *self.config.bound_a_max)

    @property
    def b(self) -> torch.Tensor:
        return _bounded(self.logit_b, *self.config.bound_b)

    @property
    def s0(self) -> torch.Tensor:
        return _bounded(self.logit_s0, *self.config.bound_s0)

    def _idm_acceleration(self, v_lead: torch.Tensor, v_follow: torch.Tensor, s_follow: torch.Tensor) -> torch.Tensor:
        delta_v = v_follow - v_lead
        denom = 2.0 * torch.sqrt(self.a_max * self.b + 1e-6)
        s_star = self.s0 + torch.clamp(v_follow * self.T + (v_follow * delta_v) / denom, min=0.0)
        return self.a_max * (
            1.0
            - torch.pow(torch.clamp(v_follow / self.v0, min=1e-6), self.config.delta)
            - torch.pow(s_star / torch.clamp(s_follow, min=1e-3), 2)
        )

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        B, T, N, F = history_raw.shape
        T_fut = self.config.predict_steps
        device = history_raw.device
        dtype = history_raw.dtype

        normed = self.normalise(history_raw) * history_mask
        per_vehicle = normed.permute(0, 2, 1, 3)  # (B, N, T, F)
        cnn_feat = self.cnn(per_vehicle)  # (B, N, T, C)
        interaction_feats: list[torch.Tensor] = []
        for name in self.config.interaction_features:
            idx = FEATURE_INDEX[name]
            full = normed[..., idx]  # (B, T, N)
            shifted = torch.zeros_like(full)
            shifted[..., 1:] = full[..., :-1]
            interaction_feats.append(shifted.unsqueeze(-1).permute(0, 2, 1, 3))
        if interaction_feats:
            interaction_feats_tensor = torch.cat(interaction_feats, dim=-1)
            lstm_in = torch.cat([cnn_feat, interaction_feats_tensor], dim=-1)
        else:
            lstm_in = cnn_feat
        flat_in = lstm_in.reshape(B * N, T, -1)
        encoded, _ = self.lstm(flat_in)
        last = encoded[:, -1, :].reshape(B, N, -1)
        future_q = self.future_query.unsqueeze(0).expand(B, -1, -1)  # (B, T_fut, H)
        out_v = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_s = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_a = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_xrel = torch.empty(B, T_fut, N, device=device, dtype=dtype)

        v = history_raw[:, -1, :, FEATURE_INDEX["v"]]
        s = history_raw[:, -1, :, FEATURE_INDEX["s"]]
        a_prev = history_raw[:, -1, :, FEATURE_INDEX["a"]]
        x_rel = history_raw[:, -1, :, FEATURE_INDEX["x_rel_leader"]]

        for t in range(T_fut):
            future_token = future_q[:, t, :]  # (B, H)
            future_token_n = future_token.unsqueeze(1).expand(-1, N, -1)
            merged = torch.cat([last, future_token_n], dim=-1)
            residual_a = self.residual_mlp(merged).squeeze(-1)  # (B, N)

            v_lead = v[:, :-1]
            v_follow = v[:, 1:]
            s_follow = s[:, 1:]
            a_idm_follow = self._idm_acceleration(v_lead, v_follow, s_follow)
            if self.config.leader_mode == "constant_velocity":
                a_idm_lead = torch.zeros_like(v[:, [0]])
            elif self.config.leader_mode == "constant_acceleration":
                a_idm_lead = a_prev[:, [0]]
            else:
                raise ValueError(f"Unknown leader_mode {self.config.leader_mode}")
            a_idm = torch.cat([a_idm_lead, a_idm_follow], dim=1)
            a = a_idm + residual_a
            a = torch.clamp(a, min=self.config.accel_clamp[0], max=self.config.accel_clamp[1])
            v_new = torch.clamp(v + a * self.dt, min=self.config.speed_clamp_lower)
            v_lead_new = v_new[:, [0]]
            x_rel_new = x_rel + (v_new - v_lead_new) * self.dt
            v_lead_new_full = v_new[:, :-1]
            v_follow_new = v_new[:, 1:]
            s_new_follow = s[:, 1:] + (v_lead_new_full - v_follow_new) * self.dt
            s_new = torch.cat([torch.zeros(B, 1, device=device, dtype=dtype), s_new_follow], dim=1)

            out_v[:, t, :] = v_new
            out_s[:, t, :] = s_new
            out_a[:, t, :] = a
            out_xrel[:, t, :] = x_rel_new

            v = v_new
            s = s_new
            a_prev = a
            x_rel = x_rel_new
        return torch.stack([out_v, out_s, out_a, out_xrel], dim=-1)
