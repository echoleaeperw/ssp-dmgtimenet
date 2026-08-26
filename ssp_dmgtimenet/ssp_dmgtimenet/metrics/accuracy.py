"""Prediction accuracy metrics aggregated by variable / horizon / vehicle."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import torch


VariableIndex = Mapping[str, int]
DEFAULT_VARIABLE_INDEX: VariableIndex = {"v": 0, "s": 1, "a": 2}


def _ensure_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _validate_shapes(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(f"mask shape {mask.shape} != pred shape {pred.shape}")


def mae_per_variable(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    variable_index: VariableIndex | None = None,
) -> dict[str, float]:
    """Mean absolute error per output variable across (B, T, N).

    ``pred`` and ``target`` are expected to have shape ``(B, T, N, D)`` where
    the last dimension is indexed by ``variable_index``.
    """

    pred_np = _ensure_numpy(pred)
    target_np = _ensure_numpy(target)
    mask_np = _ensure_numpy(mask) if mask is not None else None
    _validate_shapes(pred_np, target_np, mask_np)
    variable_index = variable_index or DEFAULT_VARIABLE_INDEX

    out: dict[str, float] = {}
    safe_pred = np.where(np.isfinite(target_np), pred_np, 0.0)
    safe_target = np.where(np.isfinite(target_np), target_np, 0.0)
    abs_err = np.abs(safe_pred - safe_target)
    for name, d in variable_index.items():
        e = abs_err[..., d]
        if mask_np is not None:
            m = mask_np[..., d]
            denom = max(float(m.sum()), 1.0)
            out[name] = float((e * m).sum() / denom)
        else:
            out[name] = float(e.mean())
    return out


def rmse_per_variable(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    variable_index: VariableIndex | None = None,
) -> dict[str, float]:
    pred_np = _ensure_numpy(pred)
    target_np = _ensure_numpy(target)
    mask_np = _ensure_numpy(mask) if mask is not None else None
    _validate_shapes(pred_np, target_np, mask_np)
    variable_index = variable_index or DEFAULT_VARIABLE_INDEX

    out: dict[str, float] = {}
    safe_pred = np.where(np.isfinite(target_np), pred_np, 0.0)
    safe_target = np.where(np.isfinite(target_np), target_np, 0.0)
    sq_err = (safe_pred - safe_target) ** 2
    for name, d in variable_index.items():
        e = sq_err[..., d]
        if mask_np is not None:
            m = mask_np[..., d]
            denom = max(float(m.sum()), 1.0)
            out[name] = float(np.sqrt((e * m).sum() / denom))
        else:
            out[name] = float(np.sqrt(e.mean()))
    return out


def horizon_wise_errors(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    target_hz: float,
    horizons_seconds: Iterable[float] = (1.0, 2.0, 3.0, 5.0),
    mask: torch.Tensor | np.ndarray | None = None,
    variable_index: VariableIndex | None = None,
) -> dict[str, dict[str, float]]:
    """For each horizon, MAE/RMSE on the *prefix* ``[:t_h]`` of the predict window."""

    pred_np = _ensure_numpy(pred)
    target_np = _ensure_numpy(target)
    mask_np = _ensure_numpy(mask) if mask is not None else None
    _validate_shapes(pred_np, target_np, mask_np)

    total_steps = pred_np.shape[1]
    results: dict[str, dict[str, float]] = {}
    for sec in horizons_seconds:
        steps = max(1, int(round(sec * target_hz)))
        if steps > total_steps:
            continue
        sl_pred = pred_np[:, :steps]
        sl_target = target_np[:, :steps]
        sl_mask = mask_np[:, :steps] if mask_np is not None else None
        mae = mae_per_variable(sl_pred, sl_target, sl_mask, variable_index)
        rmse = rmse_per_variable(sl_pred, sl_target, sl_mask, variable_index)
        results[f"horizon_{sec:.1f}s"] = {
            **{f"mae_{k}": v for k, v in mae.items()},
            **{f"rmse_{k}": v for k, v in rmse.items()},
        }
    return results


def vehicle_wise_errors(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    variable_index: VariableIndex | None = None,
) -> dict[str, dict[str, float]]:
    """Per-vehicle MAE/RMSE; vehicle 0 is the leader."""

    pred_np = _ensure_numpy(pred)
    target_np = _ensure_numpy(target)
    mask_np = _ensure_numpy(mask) if mask is not None else None
    _validate_shapes(pred_np, target_np, mask_np)

    N = pred_np.shape[2]
    out: dict[str, dict[str, float]] = {}
    for n in range(N):
        sl_pred = pred_np[:, :, n : n + 1]
        sl_target = target_np[:, :, n : n + 1]
        sl_mask = mask_np[:, :, n : n + 1] if mask_np is not None else None
        mae = mae_per_variable(sl_pred, sl_target, sl_mask, variable_index)
        rmse = rmse_per_variable(sl_pred, sl_target, sl_mask, variable_index)
        out[f"C{n + 1}"] = {**{f"mae_{k}": v for k, v in mae.items()}, **{f"rmse_{k}": v for k, v in rmse.items()}}
    return out


def tail_vehicle_error(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray | None = None,
    variable_index: VariableIndex | None = None,
) -> dict[str, float]:
    """Errors on the last vehicle (C_N) - the "tail" indicator."""

    pred_np = _ensure_numpy(pred)
    target_np = _ensure_numpy(target)
    mask_np = _ensure_numpy(mask) if mask is not None else None
    _validate_shapes(pred_np, target_np, mask_np)

    N = pred_np.shape[2]
    sl_pred = pred_np[:, :, N - 1 : N]
    sl_target = target_np[:, :, N - 1 : N]
    sl_mask = mask_np[:, :, N - 1 : N] if mask_np is not None else None
    mae = mae_per_variable(sl_pred, sl_target, sl_mask, variable_index)
    rmse = rmse_per_variable(sl_pred, sl_target, sl_mask, variable_index)
    return {**{f"mae_{k}": v for k, v in mae.items()}, **{f"rmse_{k}": v for k, v in rmse.items()}}
