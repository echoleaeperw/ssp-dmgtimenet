"""Shared utilities for the baseline models.

Every baseline exposes the same forward contract as
:class:`ssp_dmgtimenet.models.SSPDMGTimeNet`. The helpers in this module
make it easy to reuse the dataset normalisation logic, share the output
channel layout, and produce the empty placeholders for ``cfe_residuals``
and ``layer_diagnostics`` so the unified loss / trainer keep working.

We deliberately keep the leader's future trajectory as a *first-class*
prediction rather than feeding the ground-truth leader into the followers
(which would leak label information). All baselines therefore predict every
vehicle in the platoon, including the leader; in the physics-based
cascades the leader is rolled out with a constant-velocity / constant-
acceleration assumption, which mirrors how the original IDM/OVM/FVDM
papers initialise an isolated leader.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..data.windowing import FEATURE_NAMES


OUTPUT_CHANNELS: tuple[str, ...] = ("v", "s", "a", "x_rel_leader")


def _index_of(name: str) -> int:
    if name not in FEATURE_NAMES:
        raise KeyError(f"FEATURE_NAMES is missing required field {name!r}")
    return FEATURE_NAMES.index(name)


FEATURE_INDEX = {name: _index_of(name) for name in FEATURE_NAMES}


@dataclass(slots=True, frozen=True)
class BaselineConfigBase:
    num_vehicles: int = 5
    history_steps: int = 50
    predict_steps: int = 30
    target_hz: float = 10.0
    num_features_in: int = len(FEATURE_NAMES)
    num_output_channels: int = len(OUTPUT_CHANNELS)


def baseline_zero_extras(
    history_raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Produce the zero ``cfe_residuals`` / mask placeholders required by the loss."""

    if history_raw.dim() != 4:
        raise ValueError(f"history_raw must be (B, T, N, F), got {history_raw.shape}")
    B, T, N, _ = history_raw.shape
    return {
        "cfe_residuals": torch.zeros(B, T, N, 3, device=history_raw.device, dtype=history_raw.dtype),
        "cfe_residual_mask": torch.zeros(N, dtype=torch.bool, device=history_raw.device),
        "hgf_weights": torch.empty(0, device=history_raw.device),
        "layer_diagnostics": [],
    }


class BaselineBase(nn.Module):
    """Common scaffolding for baseline models.

    Subclasses must define :meth:`predict` returning a tensor of shape
    ``(B, T_fut, N, num_output_channels)`` in *physical* units. The base
    class will:

    * Validate input shapes/dtypes.
    * Replace masked entries with zero before passing to ``predict``.
    * Construct the SSP-style output dict with placeholder extras.
    """

    def __init__(self, config: BaselineConfigBase) -> None:
        super().__init__()
        self.config = config

    def _check(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> None:
        if history_raw.dim() != 4:
            raise ValueError(f"history_raw must be (B, T, N, F), got {history_raw.shape}")
        if history_raw.shape != history_mask.shape:
            raise ValueError("history_mask must match history_raw shape")
        if history_raw.shape[1] != self.config.history_steps:
            raise ValueError(
                f"history_steps mismatch: expected {self.config.history_steps}, got {history_raw.shape[1]}"
            )
        if history_raw.shape[2] != self.config.num_vehicles:
            raise ValueError(
                f"num_vehicles mismatch: expected {self.config.num_vehicles}, got {history_raw.shape[2]}"
            )
        if history_raw.shape[3] != self.config.num_features_in:
            raise ValueError(
                f"num_features_in mismatch: expected {self.config.num_features_in}, got {history_raw.shape[3]}"
            )

    def predict(self, history_raw: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        history_raw: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._check(history_raw, history_mask)
        history_filled = torch.where(history_mask > 0, history_raw, torch.zeros_like(history_raw))
        predictions = self.predict(history_filled, history_mask)
        if predictions.dim() != 4:
            raise ValueError(f"predict() must return (B, T_fut, N, D_out), got {predictions.shape}")
        if predictions.shape[1] != self.config.predict_steps:
            raise ValueError(
                f"predict() returned T_fut={predictions.shape[1]}, expected {self.config.predict_steps}"
            )
        if predictions.shape[3] != self.config.num_output_channels:
            raise ValueError(
                f"predict() returned D_out={predictions.shape[3]}, expected {self.config.num_output_channels}"
            )
        out = baseline_zero_extras(history_filled)
        out["predictions"] = predictions
        return out

    def init_normalisation(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        """Most baselines do not need normalisation buffers; this is a no-op so the trainer can call it uniformly.

        Subclasses that *do* need it (e.g. neural ones) should override.
        """

        return None


class _NormalisedBackbone(BaselineBase):
    """A baseline that normalises inputs / denormalises outputs internally."""

    def __init__(self, config: BaselineConfigBase) -> None:
        super().__init__(config)
        self.register_buffer("input_mean", torch.zeros(config.num_features_in), persistent=True)
        self.register_buffer("input_std", torch.ones(config.num_features_in), persistent=True)
        self.register_buffer("output_mean", torch.zeros(config.num_output_channels), persistent=True)
        self.register_buffer("output_std", torch.ones(config.num_output_channels), persistent=True)

    def init_normalisation(
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
        return (x - self.input_mean.view(1, 1, 1, -1)) / self.input_std.view(1, 1, 1, -1)

    def denormalise(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.output_std.view(1, 1, 1, -1) + self.output_mean.view(1, 1, 1, -1)
