"""Baseline models for scheme C.

Each baseline exposes the *same* forward signature as
:class:`ssp_dmgtimenet.models.SSPDMGTimeNet` so the trainer can swap them in
without code changes:

>>> output = model(history_raw, history_mask)
>>> output["predictions"]            # (B, T_fut, N, 4) physical units
>>> output["cfe_residuals"]          # (B, T_hist, N, 3) zero placeholder
>>> output["cfe_residual_mask"]      # (N,) all-False if no CFE
>>> output["hgf_weights"]            # optional, may be missing
>>> output["layer_diagnostics"]      # list, may be empty

The total loss can therefore be reused across SSP-DMGTimeNet and every
baseline; the only adjustment is that for a baseline that lacks SP-DACA, the
``attention_blocks`` argument to :class:`TotalLoss` is an empty list (then
the delay regulariser term degrades to zero) and the cointegration loss is
zero because the residual mask is all-False.
"""

from .common import BaselineBase, BaselineConfigBase, baseline_zero_extras
from .physics import (
    IDMCascade,
    IDMCascadeConfig,
    OVMCascade,
    OVMCascadeConfig,
    FVDMCascade,
    FVDMCascadeConfig,
)
from .recurrent import (
    PlatoonLSTM,
    PlatoonLSTMConfig,
    PlatoonGRU,
    PlatoonGRUConfig,
    InteractionLSTM,
    InteractionLSTMConfig,
)
from .transformer import PlatoonTransformer, PlatoonTransformerConfig
from .graph import FullGraphAttention, FullGraphAttentionConfig
from .dmg_cascade import DMGCascade, DMGCascadeConfig
from .cnn_int_lstm_idm import CNNIntLSTMIDM, CNNIntLSTMIDMConfig

__all__ = [
    "BaselineBase",
    "BaselineConfigBase",
    "baseline_zero_extras",
    "IDMCascade",
    "IDMCascadeConfig",
    "OVMCascade",
    "OVMCascadeConfig",
    "FVDMCascade",
    "FVDMCascadeConfig",
    "PlatoonLSTM",
    "PlatoonLSTMConfig",
    "PlatoonGRU",
    "PlatoonGRUConfig",
    "InteractionLSTM",
    "InteractionLSTMConfig",
    "PlatoonTransformer",
    "PlatoonTransformerConfig",
    "FullGraphAttention",
    "FullGraphAttentionConfig",
    "DMGCascade",
    "DMGCascadeConfig",
    "CNNIntLSTMIDM",
    "CNNIntLSTMIDMConfig",
]
