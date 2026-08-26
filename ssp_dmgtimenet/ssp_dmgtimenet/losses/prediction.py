"""Primary prediction loss with optional Huber on individual variables."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class PredictionLossConfig:
    variables: tuple[str, ...] = ("v", "s", "a", "x_rel_leader")
    weights: dict[str, float] = field(default_factory=lambda: {"v": 1.0, "s": 1.0, "a": 1.0, "x_rel_leader": 0.5})
    loss_type: str = "huber"
    huber_delta: float = 1.0
    leader_weight: float = 1.0


class PredictionLoss(nn.Module):
    """Per-variable, per-vehicle weighted regression loss.

    When ``output_std`` is registered via :meth:`set_normalisation`, the
    loss is computed in z-score space so that gradient magnitudes are
    naturally balanced across variables of different physical scales.

    Inputs:

    * ``predictions``: ``(B, T_fut, N, D_out)`` in physical units.
    * ``targets``: ``(B, T_fut, N, D_out)`` in physical units.
    * ``mask``: ``(B, T_fut, N, D_out)`` with 1 where the value is finite,
      0 otherwise. Required because the gap channel is zeroed for the leader.
    """

    def __init__(self, config: PredictionLossConfig) -> None:
        super().__init__()
        self.config = config
        if config.loss_type not in {"huber", "mae", "mse"}:
            raise ValueError(f"Unknown loss_type {config.loss_type}")
        self._output_mean: torch.Tensor | None = None
        self._output_std: torch.Tensor | None = None

    def set_normalisation(self, output_mean: torch.Tensor, output_std: torch.Tensor) -> None:
        self._output_mean = output_mean.clone()
        self._output_std = output_std.clone()

    def _to_norm_space(self, x: torch.Tensor) -> torch.Tensor:
        if self._output_std is None:
            return x
        mean = self._output_mean.to(x.device).view(1, 1, 1, -1)
        std = self._output_std.to(x.device).view(1, 1, 1, -1)
        return (x - mean) / std

    def _per_element(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.loss_type == "mae":
            return torch.abs(pred - target)
        if self.config.loss_type == "mse":
            return (pred - target) ** 2
        delta = self.config.huber_delta
        diff = pred - target
        absd = diff.abs()
        return torch.where(absd <= delta, 0.5 * diff ** 2, delta * (absd - 0.5 * delta))

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if predictions.shape != targets.shape:
            raise ValueError(f"predictions {predictions.shape} != targets {targets.shape}")
        if mask.shape != predictions.shape:
            raise ValueError("mask must match prediction shape")
        if predictions.shape[-1] != len(self.config.variables):
            raise ValueError(
                f"variables config has {len(self.config.variables)} entries, "
                f"but prediction last dim is {predictions.shape[-1]}"
            )

        pred_n = self._to_norm_space(predictions)
        tgt_n = self._to_norm_space(targets)

        safe_pred = torch.where(mask.bool(), pred_n, torch.zeros_like(pred_n))
        safe_target = torch.where(mask.bool(), tgt_n, torch.zeros_like(tgt_n))
        per_elem = self._per_element(safe_pred, safe_target)

        weights = torch.ones_like(predictions)
        for d, name in enumerate(self.config.variables):
            weights[..., d] = float(self.config.weights.get(name, 1.0))
        if self.config.leader_weight != 1.0 and predictions.shape[2] >= 1:
            weights[:, :, 0, :] *= self.config.leader_weight

        weighted = per_elem * weights * mask
        denom = (weights * mask).sum().clamp_min(1e-6)
        total = weighted.sum() / denom

        per_var: dict[str, torch.Tensor] = {}
        for d, name in enumerate(self.config.variables):
            num = (per_elem[..., d] * mask[..., d]).sum()
            den = mask[..., d].sum().clamp_min(1.0)
            per_var[name] = num / den
        return total, per_var
