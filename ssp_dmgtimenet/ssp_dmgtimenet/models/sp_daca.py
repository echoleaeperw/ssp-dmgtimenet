"""Sequential-Propagation Delay-Aware Causal Attention.

SP-DACA upgrades DMGTimeNet's per-vehicle DACA to platoon level. We expose
the platoon as a strict chain ``C_1 -> C_2 -> ... -> C_N`` and learn a
non-negative gap ``tau_i`` between adjacent vehicles (driver reaction +
inter-vehicle propagation delay). Cumulative delay between any upstream
``C_j`` and downstream ``C_i`` is ``sum_{k=j..i-1} tau_k``.

The attention bias uses a Gaussian kernel centred on the cumulative delay,

    K_tau(Delta_t) = exp(-(Delta_t - tau_{j -> i})^2 / (2 * sigma_tau^2))

so the layer can express:

* time causality (Delta_t >= 0)
* spatial upstream-only causality (j <= i)
* learnable propagation delay
* an uncertainty width via ``sigma_tau``

The implementation is multi-head, supports dropout, and operates on tensors
shaped ``(B, T, N, D)`` (batch, time, vehicle, feature).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class SequentialPropagationDACAConfig:
    d_model: int
    num_heads: int = 4
    num_vehicles: int = 5
    target_hz: float = 10.0
    tau_min: float = 0.3
    tau_max: float = 2.5
    tau_init: float = 1.0
    sigma_min: float = 0.1
    sigma_max: float = 1.5
    sigma_init: float = 0.5
    dropout: float = 0.1
    qkv_bias: bool = True
    use_relative_pe: bool = True
    # Ablation switches.
    use_delay_bias: bool = True
    spatial_mode: str = "chain"  # "chain" (upstream-only) or "full"
    learnable_tau: bool = True


class SequentialPropagationDACA(nn.Module):
    """A chain-causal multi-head attention layer with learnable delays."""

    def __init__(self, config: SequentialPropagationDACAConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if config.num_vehicles < 2:
            raise ValueError("num_vehicles must be >= 2")
        if not (0 < config.tau_min < config.tau_max):
            raise ValueError("require 0 < tau_min < tau_max")
        if config.spatial_mode not in ("chain", "full"):
            raise ValueError(f"spatial_mode must be 'chain' or 'full', got {config.spatial_mode!r}")
        self.config = config
        self.head_dim = config.d_model // config.num_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Per-pair raw parameters. We map them to (tau_min, tau_max) via sigmoid.
        init_logit = self._invert_sigmoid_unit(
            (config.tau_init - config.tau_min) / (config.tau_max - config.tau_min)
        )
        self.tau_logits = nn.Parameter(
            torch.full((config.num_heads, config.num_vehicles - 1), init_logit, dtype=torch.float32),
            requires_grad=config.learnable_tau,
        )

        sigma_init_unit = (config.sigma_init - config.sigma_min) / (config.sigma_max - config.sigma_min)
        sigma_init_unit = max(min(sigma_init_unit, 0.999), 0.001)
        self.sigma_logits = nn.Parameter(
            torch.full((config.num_heads,), self._invert_sigmoid_unit(sigma_init_unit), dtype=torch.float32)
        )

        self.use_relative_pe = config.use_relative_pe
        if config.use_relative_pe:
            self.time_pe_scale = nn.Parameter(torch.ones(config.num_heads))

    @staticmethod
    def _invert_sigmoid_unit(p: float) -> float:
        p_clamped = min(max(p, 1e-4), 1 - 1e-4)
        return float(torch.logit(torch.tensor(p_clamped)))

    @property
    def tau(self) -> torch.Tensor:
        """Per-head, per-pair learnable delay in seconds, shape ``(H, N-1)``."""

        s = torch.sigmoid(self.tau_logits)
        return self.config.tau_min + (self.config.tau_max - self.config.tau_min) * s

    @property
    def sigma_tau(self) -> torch.Tensor:
        """Per-head Gaussian width (s), shape ``(H,)``."""

        s = torch.sigmoid(self.sigma_logits)
        return self.config.sigma_min + (self.config.sigma_max - self.config.sigma_min) * s

    def cumulative_tau(self) -> torch.Tensor:
        """Cumulative delay between any pair ``(j, i)`` with ``j < i``.

        Returns shape ``(H, N, N)``. Entries with ``j >= i`` are zero (and
        will be masked out anyway).
        """

        tau = self.tau  # (H, N-1)
        H, _ = tau.shape
        N = self.config.num_vehicles
        cum = torch.zeros(H, N, N, device=tau.device, dtype=tau.dtype)
        for j in range(N):
            running = torch.zeros(H, device=tau.device, dtype=tau.dtype)
            for i in range(j + 1, N):
                running = running + tau[:, i - 1]
                cum[:, j, i] = running
        return cum

    def _spatial_causal_mask(self, device: torch.device) -> torch.Tensor:
        """Boolean mask ``(N_q, N_k)``; ``mask[i, j] = True`` means query
        vehicle ``i`` may attend key vehicle ``j``.

        Chain mode allows upstream-only attention ``j <= i`` (index 0 is the
        platoon leader), matching ``cumulative_tau()`` whose entries are
        non-zero exactly for ``j < i``. v5 shipped with the comparison
        inverted (``j >= i``), which masked out every tau-dependent bias
        entry and starved ``tau_logits`` of data gradient (the root cause of
        the observed tau collapse).
        """

        N = self.config.num_vehicles
        if self.config.spatial_mode == "full":
            return torch.ones(N, N, dtype=torch.bool, device=device)
        idx = torch.arange(N, device=device)
        return idx[None, :] <= idx[:, None]

    def _time_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(T, device=device)
        return idx[None, :] <= idx[:, None]  # True if t' <= t (allowed)

    def _delay_bias(self, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute attention bias of shape ``(H, T_q, N_q, T_k, N_k)``.

        ``dt[t_q, t_k] = (t_q - t_k) / fs`` is the time gap between the query
        timestamp and the key timestamp. ``cumulative_tau()`` returns the
        per-head cumulative delay indexed by ``(j, i)``; for a query at
        vehicle ``i_q`` and a key at vehicle ``j_k`` we therefore index it
        as ``cum[h, j_k, i_q]``. The kernel is then a Gaussian centred on the
        cumulative delay.
        """

        dt = (
            torch.arange(T, device=device, dtype=dtype)[:, None]
            - torch.arange(T, device=device, dtype=dtype)[None, :]
        ) / float(self.config.target_hz)
        cum = self.cumulative_tau().to(device=device, dtype=dtype)  # (H, j, i)
        cum_qk = cum.permute(0, 2, 1)  # (H, i_q, j_k)
        sigma = self.sigma_tau.to(device=device, dtype=dtype)
        # (H, T_q, N_q, T_k, N_k)
        delta = dt[None, :, None, :, None] - cum_qk[:, None, :, None, :]
        sigma_b = sigma[:, None, None, None, None]
        bias = -(delta ** 2) / (2.0 * sigma_b ** 2 + 1e-6)
        return bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 4:
            raise ValueError(f"SP-DACA expects (B, T, N, D), got {x.shape}")
        B, T, N, D = x.shape
        if N != self.config.num_vehicles:
            raise ValueError(f"Expected N={self.config.num_vehicles}, got {N}")
        if D != self.config.d_model:
            raise ValueError(f"Expected D={self.config.d_model}, got {D}")
        H = self.config.num_heads
        Hd = self.head_dim
        device = x.device
        dtype = x.dtype

        q = self.q_proj(x).reshape(B, T, N, H, Hd).permute(0, 3, 1, 2, 4)  # (B, H, T, N, d)
        k = self.k_proj(x).reshape(B, T, N, H, Hd).permute(0, 3, 1, 2, 4)
        v = self.v_proj(x).reshape(B, T, N, H, Hd).permute(0, 3, 1, 2, 4)

        q_flat = q.reshape(B, H, T * N, Hd)
        k_flat = k.reshape(B, H, T * N, Hd)
        v_flat = v.reshape(B, H, T * N, Hd)

        scale = 1.0 / (Hd ** 0.5)
        scores = torch.matmul(q_flat, k_flat.transpose(-1, -2)) * scale  # (B, H, T*N, T*N)

        if self.config.use_delay_bias:
            delay_bias = self._delay_bias(T, device, dtype)  # (H, T_q, N_q, T_k, N_k)
            if self.use_relative_pe:
                delay_bias = delay_bias * self.time_pe_scale.view(H, 1, 1, 1, 1)
            scores = scores + delay_bias.reshape(1, H, T * N, T * N)

        spatial_mask = self._spatial_causal_mask(device)  # (N_q, N_k)
        time_mask = self._time_causal_mask(T, device)  # (T_q, T_k)
        full_mask = time_mask[:, None, :, None] & spatial_mask[None, :, None, :]
        full_mask = full_mask.reshape(T * N, T * N)

        scores = scores.masked_fill(~full_mask[None, None, :, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v_flat)  # (B, H, T*N, d)
        out = out.reshape(B, H, T, N, Hd).permute(0, 2, 3, 1, 4).reshape(B, T, N, D)
        out = self.proj_dropout(self.out_proj(out))

        diagnostics = {
            "tau": self.tau.detach(),
            "sigma_tau": self.sigma_tau.detach(),
            "cumulative_tau": self.cumulative_tau().detach(),
            "attention": attn.detach(),
        }
        return out, diagnostics
