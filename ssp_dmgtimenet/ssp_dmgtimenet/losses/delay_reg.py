"""Regularisation on the SP-DACA learnable delays.

We penalise three things:

* Smoothness along the platoon: ``(tau_{i+1} - tau_i)^2`` so the chain delay
  cannot oscillate aggressively from one pair to the next, which is the
  physically-motivated prior in scheme C §5.6 (delay continuity).
* Soft boundary attraction towards the centre of the configured ``[tau_min,
  tau_max]`` window. This stops the sigmoid from saturating at either end
  and turning the gradient off.
* Variance across attention heads: heads are encouraged to disagree only as
  much as the data supports, otherwise individual heads quickly diverge to
  noisy delay estimates that hurt cross-vehicle CFE consistency.

The module operates on a list of ``SequentialPropagationDACA`` blocks; the
caller passes the model's ``layer_diagnostics`` already stacked or the
attention modules directly. We expose both an explicit "from blocks" forward
and a "from tensor" forward so the trainer can short-circuit one or the
other when convenient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from ..models.sp_daca import SequentialPropagationDACA


@dataclass(slots=True, frozen=True)
class DelayRegConfig:
    weight_smooth: float = 1.0
    weight_centre: float = 0.05
    weight_head_var: float = 0.1
    weight_sigma_centre: float = 0.05
    target_centre_unit: float = 0.5


class DelayRegulariser(nn.Module):
    """Compute the scheme-C tau-regularisation across all SP-DACA blocks."""

    def __init__(self, config: DelayRegConfig) -> None:
        super().__init__()
        self.config = config

    def forward_from_blocks(
        self,
        attention_blocks: Iterable[SequentialPropagationDACA],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        taus: list[torch.Tensor] = []
        sigmas: list[torch.Tensor] = []
        bounds_tau: tuple[float, float] | None = None
        bounds_sigma: tuple[float, float] | None = None
        for block in attention_blocks:
            taus.append(block.tau)
            sigmas.append(block.sigma_tau)
            if bounds_tau is None:
                bounds_tau = (block.config.tau_min, block.config.tau_max)
                bounds_sigma = (block.config.sigma_min, block.config.sigma_max)
            elif bounds_tau != (block.config.tau_min, block.config.tau_max):
                raise ValueError(
                    "All SP-DACA blocks must share the same tau_min/tau_max for delay reg"
                )
        if not taus:
            zero = torch.zeros((), dtype=torch.float32)
            return zero, {
                "delay_smooth": zero,
                "delay_centre": zero,
                "delay_head_var": zero,
                "delay_sigma_centre": zero,
            }
        if bounds_tau is None or bounds_sigma is None:
            raise RuntimeError("attention_blocks did not yield any block")
        return self._compute(taus, sigmas, bounds_tau, bounds_sigma)

    def forward(
        self,
        taus: torch.Tensor,
        sigmas: torch.Tensor,
        bounds_tau: tuple[float, float],
        bounds_sigma: tuple[float, float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if taus.dim() != 3:
            raise ValueError(f"taus must be (L, H, N-1), got {taus.shape}")
        if sigmas.dim() != 2:
            raise ValueError(f"sigmas must be (L, H), got {sigmas.shape}")
        if taus.shape[:2] != sigmas.shape:
            raise ValueError(
                f"taus and sigmas must agree on (L, H); got {taus.shape} vs {sigmas.shape}"
            )
        per_layer_tau = [taus[i] for i in range(taus.shape[0])]
        per_layer_sigma = [sigmas[i] for i in range(sigmas.shape[0])]
        return self._compute(per_layer_tau, per_layer_sigma, bounds_tau, bounds_sigma)

    def _compute(
        self,
        taus: list[torch.Tensor],
        sigmas: list[torch.Tensor],
        bounds_tau: tuple[float, float],
        bounds_sigma: tuple[float, float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = taus[0].device
        dtype = taus[0].dtype
        tau_min, tau_max = bounds_tau
        sigma_min, sigma_max = bounds_sigma

        smooth_terms: list[torch.Tensor] = []
        centre_terms: list[torch.Tensor] = []
        head_var_terms: list[torch.Tensor] = []
        sigma_centre_terms: list[torch.Tensor] = []
        target_unit = float(self.config.target_centre_unit)
        target_unit = min(max(target_unit, 0.0), 1.0)
        for tau, sigma in zip(taus, sigmas, strict=True):
            if tau.shape[1] >= 2:
                diffs = tau[:, 1:] - tau[:, :-1]
                smooth_terms.append((diffs ** 2).mean())
            else:
                smooth_terms.append(torch.zeros((), device=device, dtype=dtype))
            tau_unit = (tau - tau_min) / max(tau_max - tau_min, 1e-6)
            centre_terms.append(((tau_unit - target_unit) ** 2).mean())
            if tau.shape[0] >= 2:
                head_var_terms.append(tau.var(dim=0, unbiased=False).mean())
            else:
                head_var_terms.append(torch.zeros((), device=device, dtype=dtype))
            sigma_unit = (sigma - sigma_min) / max(sigma_max - sigma_min, 1e-6)
            sigma_centre_terms.append(((sigma_unit - target_unit) ** 2).mean())

        smooth = torch.stack(smooth_terms).mean()
        centre = torch.stack(centre_terms).mean()
        head_var = torch.stack(head_var_terms).mean()
        sigma_centre = torch.stack(sigma_centre_terms).mean()

        total = (
            self.config.weight_smooth * smooth
            + self.config.weight_centre * centre
            + self.config.weight_head_var * head_var
            + self.config.weight_sigma_centre * sigma_centre
        )
        return total, {
            "delay_smooth": smooth.detach(),
            "delay_centre": centre.detach(),
            "delay_head_var": head_var.detach(),
            "delay_sigma_centre": sigma_centre.detach(),
        }
