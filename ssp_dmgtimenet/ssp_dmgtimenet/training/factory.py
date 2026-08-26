"""Build model + loss objects from a YAML-style config dictionary."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Mapping

import torch
from torch import nn

from ..baselines import (
    CNNIntLSTMIDM,
    CNNIntLSTMIDMConfig,
    DMGCascade,
    DMGCascadeConfig,
    FullGraphAttention,
    FullGraphAttentionConfig,
    FVDMCascade,
    FVDMCascadeConfig,
    IDMCascade,
    IDMCascadeConfig,
    InteractionLSTM,
    InteractionLSTMConfig,
    OVMCascade,
    OVMCascadeConfig,
    PlatoonGRU,
    PlatoonGRUConfig,
    PlatoonLSTM,
    PlatoonLSTMConfig,
    PlatoonTransformer,
    PlatoonTransformerConfig,
)
from ..losses.cointegration import CointegrationLossConfig
from ..losses.delay_reg import DelayRegConfig
from ..losses.kinematics import KinematicsLossConfig
from ..losses.prediction import PredictionLossConfig
from ..losses.safety import SafetyLossConfig
from ..losses.stability import StabilityLossConfig
from ..losses.total import CurriculumConfig, TotalLoss, TotalLossConfig
from ..models import SSPDMGTimeNet, SSPDMGTimeNetConfig


_MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "ssp_dmgtimenet": (SSPDMGTimeNet, SSPDMGTimeNetConfig),
    "idm_cascade": (IDMCascade, IDMCascadeConfig),
    "ovm_cascade": (OVMCascade, OVMCascadeConfig),
    "fvdm_cascade": (FVDMCascade, FVDMCascadeConfig),
    "platoon_lstm": (PlatoonLSTM, PlatoonLSTMConfig),
    "platoon_gru": (PlatoonGRU, PlatoonGRUConfig),
    "interaction_lstm": (InteractionLSTM, InteractionLSTMConfig),
    "platoon_transformer": (PlatoonTransformer, PlatoonTransformerConfig),
    "full_graph_attention": (FullGraphAttention, FullGraphAttentionConfig),
    "dmg_cascade": (DMGCascade, DMGCascadeConfig),
    "cnn_int_lstm_idm": (CNNIntLSTMIDM, CNNIntLSTMIDMConfig),
}


def _coerce_value(value: Any, type_hint: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, (list, tuple)):
        coerced = type(value)(_coerce_value(v, None) for v in value)
        return tuple(coerced) if isinstance(coerced, tuple) else list(coerced)
    if isinstance(value, dict):
        return {str(k): _coerce_value(v, None) for k, v in value.items()}
    return value


def _from_mapping(target_type: type, mapping: Mapping[str, Any]) -> Any:
    if not is_dataclass(target_type):
        raise TypeError(f"{target_type.__name__} is not a dataclass")
    valid = {f.name for f in fields(target_type)}
    extras = set(mapping.keys()) - valid
    if extras:
        raise ValueError(f"Unknown fields for {target_type.__name__}: {sorted(extras)}")
    converted: dict[str, Any] = {}
    for f in fields(target_type):
        if f.name not in mapping:
            continue
        raw = mapping[f.name]
        if is_dataclass(f.type) and isinstance(raw, Mapping):
            converted[f.name] = _from_mapping(f.type, raw)
        else:
            converted[f.name] = _coerce_value(raw, f.type)
    return target_type(**converted)


def build_model(model_section: Mapping[str, Any]) -> nn.Module:
    name = model_section.get("name")
    if name is None:
        raise KeyError("model_section is missing 'name'")
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {name!r}. Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    cls, cfg_cls = _MODEL_REGISTRY[name]
    cfg_section = model_section.get("config", {})
    if not isinstance(cfg_section, Mapping):
        raise TypeError(f"model.config must be a mapping for {name!r}")
    config = _from_mapping(cfg_cls, dict(cfg_section))
    return cls(config)


def _build_subconfig(target_cls: type, mapping: Mapping[str, Any] | None) -> Any:
    if mapping is None:
        return target_cls()
    return _from_mapping(target_cls, dict(mapping))


def build_total_loss(loss_section: Mapping[str, Any]) -> TotalLoss:
    prediction = _build_subconfig(PredictionLossConfig, loss_section.get("prediction"))
    kinematics = _build_subconfig(KinematicsLossConfig, loss_section.get("kinematics"))
    safety = _build_subconfig(SafetyLossConfig, loss_section.get("safety"))
    stability = _build_subconfig(StabilityLossConfig, loss_section.get("stability"))
    cointegration = _build_subconfig(CointegrationLossConfig, loss_section.get("cointegration"))
    delay_reg = _build_subconfig(DelayRegConfig, loss_section.get("delay_reg"))
    curriculum = _build_subconfig(CurriculumConfig, loss_section.get("curriculum"))
    weights = loss_section.get("weights", {})
    if not isinstance(weights, Mapping):
        raise TypeError("loss.weights must be a mapping")
    excitation = loss_section.get("excitation_gate", {})
    if not isinstance(excitation, Mapping):
        raise TypeError("loss.excitation_gate must be a mapping")
    config = TotalLossConfig(
        prediction=prediction,
        kinematics=kinematics,
        safety=safety,
        stability=stability,
        cointegration=cointegration,
        delay_reg=delay_reg,
        curriculum=curriculum,
        weight_pred=float(weights.get("pred", 1.0)),
        weight_kin=float(weights.get("kin", 0.5)),
        weight_safe=float(weights.get("safe", 0.5)),
        weight_adj=float(weights.get("adj", 1.0)),
        weight_sub=float(weights.get("sub", 1.0)),
        weight_fft=float(weights.get("fft", 1.0)),
        weight_coint=float(weights.get("coint", 0.5)),
        weight_delay=float(weights.get("delay", 0.05)),
        excitation_gate_quantile=float(excitation.get("quantile", 0.25)),
        excitation_gate_floor=float(excitation.get("floor", 0.05)),
        excitation_gate_momentum=float(excitation.get("momentum", 0.1)),
    )
    return TotalLoss(config)


def initialise_normalisation_for_model(
    model: nn.Module,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
) -> None:
    if hasattr(model, "set_normalisation"):
        model.set_normalisation(input_mean, input_std, output_mean, output_std)
    elif hasattr(model, "init_normalisation"):
        model.init_normalisation(input_mean, input_std, output_mean, output_std)
    else:
        raise AttributeError(
            f"Model {type(model).__name__} has neither set_normalisation nor init_normalisation"
        )
