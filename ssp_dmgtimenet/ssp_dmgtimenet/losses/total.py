"""Total loss aggregator with the scheme-C three-stage curriculum.

The aggregator wires together the components defined in this package and
exposes a small ``CurriculumScheduler`` that linearly ramps the scheme-C
weights according to the training progress (``epoch / total_epochs``):

* Stage 1 (``[0, stage_1)``): only ``L_pred + L_kin``;
* Stage 2 (``[stage_1, stage_2)``): ramp ``L_adj`` and ``L_sub`` from 0 to
  their full weights;
* Stage 3 (``[stage_2, 1.0]``): ramp ``L_fft`` and ``L_coint`` from 0 to
  their full weights and freeze ``L_adj``/``L_sub``.

The safety, delay-regularisation and gap penalties are always on but with
their static weights from the YAML config; the trainer can still freeze
individual components by passing ``weight_*=0`` in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from ..data.windowing import FEATURE_NAMES
from ..models.ssp_dmgtimenet import SSPDMGTimeNet
from .cointegration import CointegrationLoss, CointegrationLossConfig
from .delay_reg import DelayRegConfig, DelayRegulariser
from .kinematics import KinematicsLoss, KinematicsLossConfig
from .prediction import PredictionLoss, PredictionLossConfig
from .safety import SafetyLoss, SafetyLossConfig
from .stability import (
    AdjacentAmplificationLoss,
    FFTAmplificationLoss,
    LowExcitationGate,
    StabilityLossConfig,
    SubplatoonAmplificationLoss,
)


@dataclass(slots=True, frozen=True)
class CurriculumConfig:
    stage_1_fraction: float = 0.2
    stage_2_fraction: float = 0.6
    ramp_window_fraction: float = 0.1


@dataclass(slots=True, frozen=True)
class TotalLossConfig:
    prediction: PredictionLossConfig = field(default_factory=PredictionLossConfig)
    kinematics: KinematicsLossConfig = field(default_factory=KinematicsLossConfig)
    safety: SafetyLossConfig = field(default_factory=SafetyLossConfig)
    stability: StabilityLossConfig = field(default_factory=StabilityLossConfig)
    cointegration: CointegrationLossConfig = field(default_factory=CointegrationLossConfig)
    delay_reg: DelayRegConfig = field(default_factory=DelayRegConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    weight_pred: float = 1.0
    weight_kin: float = 0.5
    weight_safe: float = 0.5
    weight_adj: float = 1.0
    weight_sub: float = 1.0
    weight_fft: float = 1.0
    weight_coint: float = 0.5
    weight_delay: float = 0.05
    excitation_gate_quantile: float = 0.25
    excitation_gate_floor: float = 0.05
    excitation_gate_momentum: float = 0.1


def _ramp(progress: float, start: float, end: float) -> float:
    if progress <= start:
        return 0.0
    if progress >= end:
        return 1.0
    return float((progress - start) / max(end - start, 1e-6))


def stage_weights(
    progress: float,
    config: CurriculumConfig,
) -> dict[str, float]:
    s1 = config.stage_1_fraction
    s2 = config.stage_2_fraction
    if not 0.0 <= s1 <= s2 <= 1.0:
        raise ValueError("Need 0 <= stage_1_fraction <= stage_2_fraction <= 1.0")
    ramp_w = max(config.ramp_window_fraction, 0.0)
    kin_safe = _ramp(progress, s1, s1 + ramp_w)
    adj_sub = _ramp(progress, s2, s2 + ramp_w)
    fft_start = min(s2 + ramp_w, 1.0)
    fft_coint = _ramp(progress, fft_start, fft_start + ramp_w)
    return {
        "pred": 1.0,
        "kin": kin_safe,
        "safe": kin_safe,
        "adj": adj_sub,
        "sub": adj_sub,
        "fft": fft_coint,
        "coint": fft_coint,
        "delay": 1.0,
    }


def resolve_target_indices(
    output_channels: tuple[str, ...],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> tuple[int, ...]:
    """Map prediction-channel names to indices into the per-sample feature axis."""

    out: list[int] = []
    for name in output_channels:
        if name not in feature_names:
            raise ValueError(
                f"Output channel {name!r} is not present in feature_names {feature_names}"
            )
        out.append(feature_names.index(name))
    return tuple(out)


class TotalLoss(nn.Module):
    """Aggregate all scheme-C loss terms with curriculum scheduling."""

    def __init__(
        self,
        config: TotalLossConfig,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        super().__init__()
        self.config = config
        self.feature_names = tuple(feature_names)
        self.target_indices = resolve_target_indices(
            config.prediction.variables, self.feature_names
        )
        self.prediction = PredictionLoss(config.prediction)
        self.kinematics = KinematicsLoss(config.kinematics)
        self.safety = SafetyLoss(config.safety)
        self.adjacent = AdjacentAmplificationLoss(config.stability)
        self.subplatoon = SubplatoonAmplificationLoss(config.stability)
        self.fft = FFTAmplificationLoss(config.stability)
        self.cointegration = CointegrationLoss(config.cointegration)
        self.delay_reg = DelayRegulariser(config.delay_reg)
        self.excitation_gate = LowExcitationGate(
            quantile=config.excitation_gate_quantile,
            floor=config.excitation_gate_floor,
        )
        self._momentum = float(config.excitation_gate_momentum)
        if not 0.0 <= self._momentum <= 1.0:
            raise ValueError("excitation_gate_momentum must be in [0, 1]")

    def update_excitation_threshold(self, batch_leader_velocity: torch.Tensor) -> None:
        if batch_leader_velocity.dim() != 2:
            raise ValueError(
                f"batch_leader_velocity must be (B, T_fut), got {batch_leader_velocity.shape}"
            )
        std = batch_leader_velocity.detach().std(dim=1)
        previous = float(self.excitation_gate.threshold)
        self.excitation_gate.update(std)
        if 0.0 < self._momentum < 1.0 and previous > self.excitation_gate.floor:
            blended = (1.0 - self._momentum) * previous + self._momentum * self.excitation_gate.threshold
            self.excitation_gate.threshold = max(blended, self.excitation_gate.floor)

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        attention_blocks,
        progress: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future_raw = batch.get("future_raw", batch["future"])
        future_mask = batch["future_mask"]
        vehicle_lengths = batch["vehicle_lengths"]
        if future_raw.dim() != 4:
            raise ValueError(f"batch['future_raw'] must be (B, T_fut, N, F), got {future_raw.shape}")
        if future_mask.dim() != 4 or future_mask.shape != future_raw.shape:
            raise ValueError("future_mask shape must match future_raw shape")

        predictions = model_output["predictions"]  # (B, T_fut, N, D_out)
        var_names = self.config.prediction.variables
        idx = list(self.target_indices)
        target = future_raw[..., idx]
        target_mask = future_mask[..., idx]

        pred_loss, pred_diag = self.prediction(predictions, target, target_mask)

        kin_loss, kin_diag = self.kinematics(
            predictions=predictions,
            vehicle_lengths=vehicle_lengths,
            target_predictions=target,
            leader_velocity=target[..., var_names.index("v")][:, :, 0]
            if "v" in var_names
            else None,
        )

        safe_loss, safe_diag = self.safety(predictions)

        v_index = var_names.index("v") if "v" in var_names else 0
        leader_v = target[..., v_index][:, :, 0]
        self.update_excitation_threshold(leader_v)
        excitation_mask = self.excitation_gate.mask(leader_v.std(dim=1))

        v_pred = predictions[..., v_index]
        adj_loss, adj_diag = self.adjacent(v_pred, excitation_mask=excitation_mask)
        sub_loss, sub_diag = self.subplatoon(v_pred, excitation_mask=excitation_mask)
        fft_loss, fft_diag = self.fft(v_pred, excitation_mask=excitation_mask)

        coint_residuals = model_output["cfe_residuals"]
        coint_mask = model_output["cfe_residual_mask"]
        coint_loss, coint_diag = self.cointegration(coint_residuals, coint_mask)

        delay_loss, delay_diag = self.delay_reg.forward_from_blocks(attention_blocks)

        weights = stage_weights(progress, self.config.curriculum)
        total = (
            self.config.weight_pred * weights["pred"] * pred_loss
            + self.config.weight_kin * weights["kin"] * kin_loss
            + self.config.weight_safe * weights["safe"] * safe_loss
            + self.config.weight_adj * weights["adj"] * adj_loss
            + self.config.weight_sub * weights["sub"] * sub_loss
            + self.config.weight_fft * weights["fft"] * fft_loss
            + self.config.weight_coint * weights["coint"] * coint_loss
            + self.config.weight_delay * weights["delay"] * delay_loss
        )

        diag: dict[str, torch.Tensor] = {
            "loss_total": total.detach(),
            "loss_pred": pred_loss.detach(),
            "loss_kin": kin_loss.detach(),
            "loss_safe": safe_loss.detach(),
            "loss_adj": adj_loss.detach(),
            "loss_sub": sub_loss.detach(),
            "loss_fft": fft_loss.detach(),
            "loss_coint": coint_loss.detach(),
            "loss_delay": delay_loss.detach(),
            "weight_adj_active": torch.tensor(weights["adj"]),
            "weight_fft_active": torch.tensor(weights["fft"]),
            "weight_coint_active": torch.tensor(weights["coint"]),
            "excitation_threshold": torch.tensor(self.excitation_gate.threshold),
        }
        for prefix, sub in [
            ("pred", pred_diag),
            ("kin", kin_diag),
            ("safe", safe_diag),
            ("adj", adj_diag),
            ("sub", sub_diag),
            ("fft", fft_diag),
            ("coint", coint_diag),
            ("delay", delay_diag),
        ]:
            for k, v in sub.items():
                diag[f"{prefix}.{k}"] = v.detach() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        return total, diag


def attention_blocks_of(model: SSPDMGTimeNet):
    return [block.attn for block in model.sp_daca_blocks]
