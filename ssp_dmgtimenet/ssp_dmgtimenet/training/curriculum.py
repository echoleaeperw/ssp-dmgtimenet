"""Curriculum scheduling helpers for the scheme-C three-stage loss ramp."""

from __future__ import annotations

from dataclasses import dataclass

from ..losses.total import CurriculumConfig, stage_weights


@dataclass(slots=True, frozen=True)
class CurriculumStage:
    name: str
    start_progress: float
    end_progress: float
    description: str


@dataclass(slots=True, frozen=True)
class CurriculumStageReport:
    stage: CurriculumStage
    progress: float
    weights: dict[str, float]


def progress_for_epoch(epoch: int, total_epochs: int) -> float:
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    return min(1.0, max(0.0, float(epoch) / float(total_epochs)))


def stage_for_progress(progress: float, config: CurriculumConfig) -> CurriculumStage:
    s1 = config.stage_1_fraction
    s2 = config.stage_2_fraction
    ramp_w = max(config.ramp_window_fraction, 0.0)
    if not 0.0 <= s1 <= s2 <= 1.0:
        raise ValueError("Need 0 <= stage_1_fraction <= stage_2_fraction <= 1.0")
    if progress < s1:
        return CurriculumStage(
            name="pred_only",
            start_progress=0.0,
            end_progress=s1,
            description="Stage 0: only L_pred + L_delay (pure prediction learning).",
        )
    if progress < s2:
        return CurriculumStage(
            name="ramp_kin_safe",
            start_progress=s1,
            end_progress=s2,
            description="Stage 1: ramp L_kin and L_safe from 0 to full weight.",
        )
    fft_start = min(s2 + ramp_w, 1.0)
    if progress < fft_start:
        return CurriculumStage(
            name="ramp_adj_sub",
            start_progress=s2,
            end_progress=fft_start,
            description="Stage 2: ramp L_adj and L_sub from 0 to full weight.",
        )
    return CurriculumStage(
        name="ramp_fft_coint",
        start_progress=fft_start,
        end_progress=1.0,
        description="Stage 3: ramp L_fft and L_coint to full weight.",
    )


def report_curriculum(progress: float, config: CurriculumConfig) -> CurriculumStageReport:
    return CurriculumStageReport(
        stage=stage_for_progress(progress, config),
        progress=progress,
        weights=stage_weights(progress, config),
    )
