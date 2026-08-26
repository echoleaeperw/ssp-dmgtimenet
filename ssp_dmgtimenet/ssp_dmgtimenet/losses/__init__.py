"""Loss functions for SSP-DMGTimeNet."""

from .prediction import PredictionLoss, PredictionLossConfig
from .kinematics import KinematicsLoss, KinematicsLossConfig
from .safety import SafetyLoss, SafetyLossConfig
from .stability import (
    AdjacentAmplificationLoss,
    SubplatoonAmplificationLoss,
    FFTAmplificationLoss,
    StabilityLossConfig,
    LowExcitationGate,
)
from .cointegration import CointegrationLoss, CointegrationLossConfig
from .delay_reg import DelayRegulariser, DelayRegConfig
from .total import (
    TotalLoss,
    TotalLossConfig,
    CurriculumConfig,
    stage_weights,
    attention_blocks_of,
    resolve_target_indices,
)

__all__ = [
    "PredictionLoss",
    "PredictionLossConfig",
    "KinematicsLoss",
    "KinematicsLossConfig",
    "SafetyLoss",
    "SafetyLossConfig",
    "AdjacentAmplificationLoss",
    "SubplatoonAmplificationLoss",
    "FFTAmplificationLoss",
    "StabilityLossConfig",
    "LowExcitationGate",
    "CointegrationLoss",
    "CointegrationLossConfig",
    "DelayRegulariser",
    "DelayRegConfig",
    "TotalLoss",
    "TotalLossConfig",
    "CurriculumConfig",
    "stage_weights",
    "attention_blocks_of",
    "resolve_target_indices",
]
