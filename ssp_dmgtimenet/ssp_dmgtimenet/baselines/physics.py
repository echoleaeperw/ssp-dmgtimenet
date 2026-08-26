"""Physics-based car-following baselines: IDM, OVM, FVDM.

Each baseline rolls a platoon forward by integrating an acceleration rule
with the configured time step. The leader is rolled out with a constant
velocity assumption (its last observed velocity), which is the standard
choice when the leader's future is unknown but is necessary for a
self-contained prediction. All parameters are stored as learnable scalars
constrained to physically-plausible intervals via a sigmoid mapping; this
matches "data-driven calibration" baselines reported in the trajectory
prediction literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .common import FEATURE_INDEX, BaselineBase, BaselineConfigBase


def _bounded(logit: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(logit)


def _invert(value: float, lo: float, hi: float) -> float:
    p = (value - lo) / max(hi - lo, 1e-6)
    p = min(max(p, 1e-3), 1 - 1e-3)
    return float(torch.logit(torch.tensor(p)))


class _PlatoonRoller(BaselineBase):
    """Common rollout scaffold for physics baselines."""

    def __init__(self, config: BaselineConfigBase) -> None:
        super().__init__(config)
        self.dt = 1.0 / float(config.target_hz)

    def _initial_state(
        self,
        history_raw: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v_idx = FEATURE_INDEX["v"]
        s_idx = FEATURE_INDEX["s"]
        x_rel_idx = FEATURE_INDEX["x_rel_leader"]
        a_idx = FEATURE_INDEX["a"]
        v = history_raw[..., v_idx]  # (B, T, N)
        s = history_raw[..., s_idx]
        x_rel = history_raw[..., x_rel_idx]
        a = history_raw[..., a_idx]
        last_mask = history_mask[..., v_idx]  # (B, T, N)
        if last_mask.numel() == 0:
            raise ValueError("history is empty")
        return {
            "v0": v[:, -1, :],          # (B, N)
            "s0": s[:, -1, :],          # (B, N); leader column is zero/NaN-filled
            "x_rel0": x_rel[:, -1, :],  # (B, N); leader column = 0
            "a0": a[:, -1, :],          # (B, N)
        }

    def _accumulate_x_rel(self, x_rel: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Time-step update: x_rel[t+1] = x_rel[t] + (v - v_leader) * dt."""

        v_lead = v[:, [0]]
        return x_rel + (v - v_lead) * self.dt


@dataclass(slots=True, frozen=True)
class IDMCascadeConfig(BaselineConfigBase):
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
    leader_mode: str = "constant_velocity"  # or "constant_acceleration"
    # Physical output clamp (emergency-braking floor / max drive accel).
    # HighD contains close-cut-in initial states (s < 1m at ~26 m/s) where the
    # raw IDM repulsion term diverges to |a| ~ 1e4 m/s^2; every published
    # simulation pipeline saturates commanded acceleration to a physically
    # realisable interval before integrating.
    accel_clamp: tuple[float, float] = (-8.0, 4.0)


class IDMCascade(_PlatoonRoller):
    """Cascade of Intelligent Driver Models with learnable shared parameters."""

    def __init__(self, config: IDMCascadeConfig) -> None:
        super().__init__(config)
        self.config: IDMCascadeConfig = config
        self.logit_v0 = nn.Parameter(torch.tensor(_invert(config.init_v0, *config.bound_v0)))
        self.logit_T = nn.Parameter(torch.tensor(_invert(config.init_T, *config.bound_T)))
        self.logit_a_max = nn.Parameter(torch.tensor(_invert(config.init_a_max, *config.bound_a_max)))
        self.logit_b = nn.Parameter(torch.tensor(_invert(config.init_b, *config.bound_b)))
        self.logit_s0 = nn.Parameter(torch.tensor(_invert(config.init_s0, *config.bound_s0)))

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

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        state = self._initial_state(history_raw, history_mask)
        B = history_raw.shape[0]
        N = self.config.num_vehicles
        T_fut = self.config.predict_steps
        device = history_raw.device
        dtype = history_raw.dtype

        v = state["v0"].clone()
        s = state["s0"].clone()
        x_rel = state["x_rel0"].clone()
        a_prev = state["a0"].clone()

        out_v = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_s = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_a = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_xrel = torch.empty(B, T_fut, N, device=device, dtype=dtype)

        for t in range(T_fut):
            v_lead = v[:, :-1]  # vehicles 0..N-2 act as leader for follower 1..N-1
            v_follow = v[:, 1:]
            delta_v = v_follow - v_lead
            s_follow = s[:, 1:]
            denom = 2.0 * torch.sqrt(self.a_max * self.b + 1e-6)
            s_star = self.s0 + torch.clamp(
                v_follow * self.T + (v_follow * delta_v) / denom,
                min=0.0,
            )
            a_follow = self.a_max * (
                1.0
                - torch.pow(torch.clamp(v_follow / self.v0, min=1e-6), self.config.delta)
                - torch.pow(s_star / torch.clamp(s_follow, min=1e-3), 2)
            )
            if self.config.leader_mode == "constant_velocity":
                a_lead = torch.zeros_like(v[:, [0]])
            elif self.config.leader_mode == "constant_acceleration":
                a_lead = a_prev[:, [0]]
            else:
                raise ValueError(f"Unknown leader_mode {self.config.leader_mode}")
            a = torch.cat([a_lead, a_follow], dim=1)
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


@dataclass(slots=True, frozen=True)
class OVMCascadeConfig(BaselineConfigBase):
    init_kappa: float = 0.5
    init_V1: float = 6.75
    init_V2: float = 7.91
    init_C1: float = 0.13
    init_C2: float = 1.57
    init_L: float = 5.0
    bound_kappa: tuple[float, float] = (0.05, 2.0)
    bound_V1: tuple[float, float] = (0.0, 30.0)
    bound_V2: tuple[float, float] = (0.0, 30.0)
    bound_C1: tuple[float, float] = (0.01, 0.5)
    bound_C2: tuple[float, float] = (0.0, 5.0)
    bound_L: tuple[float, float] = (3.0, 8.0)
    speed_clamp_lower: float = 0.0
    leader_mode: str = "constant_velocity"
    # Same physical saturation as IDMCascadeConfig.accel_clamp: close-cut-in
    # initial gaps in HighD make the raw OVM/FVDM commanded acceleration
    # exceed any realisable braking capability, so the simulation pipeline
    # saturates it before integration.
    accel_clamp: tuple[float, float] = (-8.0, 4.0)


class OVMCascade(_PlatoonRoller):
    """Optimal Velocity Model cascade."""

    def __init__(self, config: OVMCascadeConfig) -> None:
        super().__init__(config)
        self.config: OVMCascadeConfig = config
        self.logit_kappa = nn.Parameter(torch.tensor(_invert(config.init_kappa, *config.bound_kappa)))
        self.logit_V1 = nn.Parameter(torch.tensor(_invert(config.init_V1, *config.bound_V1)))
        self.logit_V2 = nn.Parameter(torch.tensor(_invert(config.init_V2, *config.bound_V2)))
        self.logit_C1 = nn.Parameter(torch.tensor(_invert(config.init_C1, *config.bound_C1)))
        self.logit_C2 = nn.Parameter(torch.tensor(_invert(config.init_C2, *config.bound_C2)))
        self.logit_L = nn.Parameter(torch.tensor(_invert(config.init_L, *config.bound_L)))

    @property
    def kappa(self) -> torch.Tensor:
        return _bounded(self.logit_kappa, *self.config.bound_kappa)

    @property
    def V1(self) -> torch.Tensor:
        return _bounded(self.logit_V1, *self.config.bound_V1)

    @property
    def V2(self) -> torch.Tensor:
        return _bounded(self.logit_V2, *self.config.bound_V2)

    @property
    def C1(self) -> torch.Tensor:
        return _bounded(self.logit_C1, *self.config.bound_C1)

    @property
    def C2(self) -> torch.Tensor:
        return _bounded(self.logit_C2, *self.config.bound_C2)

    @property
    def L(self) -> torch.Tensor:
        return _bounded(self.logit_L, *self.config.bound_L)

    def optimal_velocity(self, s: torch.Tensor) -> torch.Tensor:
        return self.V1 + self.V2 * torch.tanh(self.C1 * (s - self.L) - self.C2)

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        state = self._initial_state(history_raw, history_mask)
        B = history_raw.shape[0]
        N = self.config.num_vehicles
        T_fut = self.config.predict_steps
        device = history_raw.device
        dtype = history_raw.dtype

        v = state["v0"].clone()
        s = state["s0"].clone()
        x_rel = state["x_rel0"].clone()
        a_prev = state["a0"].clone()

        out_v = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_s = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_a = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_xrel = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        for t in range(T_fut):
            s_follow = s[:, 1:]
            v_follow = v[:, 1:]
            v_star = self.optimal_velocity(torch.clamp(s_follow, min=1e-3))
            a_follow = self.kappa * (v_star - v_follow)
            if self.config.leader_mode == "constant_velocity":
                a_lead = torch.zeros_like(v[:, [0]])
            elif self.config.leader_mode == "constant_acceleration":
                a_lead = a_prev[:, [0]]
            else:
                raise ValueError(f"Unknown leader_mode {self.config.leader_mode}")
            a = torch.cat([a_lead, a_follow], dim=1)
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


@dataclass(slots=True, frozen=True)
class FVDMCascadeConfig(OVMCascadeConfig):
    init_lambda: float = 0.5
    bound_lambda: tuple[float, float] = (0.0, 2.0)


class FVDMCascade(OVMCascade):
    """Full Velocity Difference Model = OVM + relative-velocity term."""

    def __init__(self, config: FVDMCascadeConfig) -> None:
        super().__init__(config)
        self.config: FVDMCascadeConfig = config
        self.logit_lambda = nn.Parameter(torch.tensor(_invert(config.init_lambda, *config.bound_lambda)))

    @property
    def lam(self) -> torch.Tensor:
        return _bounded(self.logit_lambda, *self.config.bound_lambda)

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        state = self._initial_state(history_raw, history_mask)
        B = history_raw.shape[0]
        N = self.config.num_vehicles
        T_fut = self.config.predict_steps
        device = history_raw.device
        dtype = history_raw.dtype

        v = state["v0"].clone()
        s = state["s0"].clone()
        x_rel = state["x_rel0"].clone()
        a_prev = state["a0"].clone()

        out_v = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_s = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_a = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        out_xrel = torch.empty(B, T_fut, N, device=device, dtype=dtype)
        for t in range(T_fut):
            s_follow = s[:, 1:]
            v_follow = v[:, 1:]
            v_lead = v[:, :-1]
            v_star = self.optimal_velocity(torch.clamp(s_follow, min=1e-3))
            a_follow = self.kappa * (v_star - v_follow) + self.lam * (v_lead - v_follow)
            if self.config.leader_mode == "constant_velocity":
                a_lead = torch.zeros_like(v[:, [0]])
            elif self.config.leader_mode == "constant_acceleration":
                a_lead = a_prev[:, [0]]
            else:
                raise ValueError(f"Unknown leader_mode {self.config.leader_mode}")
            a = torch.cat([a_lead, a_follow], dim=1)
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
