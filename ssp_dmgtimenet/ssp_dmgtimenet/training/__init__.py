"""Training, evaluation and logging utilities for scheme C."""

from .curriculum import (
    CurriculumStage,
    CurriculumStageReport,
    progress_for_epoch,
    stage_for_progress,
)
from .factory import build_model, build_total_loss
from .trainer import Trainer, TrainerConfig, training_step
from .evaluator import Evaluator, EvaluatorConfig, evaluate_model
from .logging import RichLogger, TensorboardLogger, MultiLogger

__all__ = [
    "CurriculumStage",
    "CurriculumStageReport",
    "progress_for_epoch",
    "stage_for_progress",
    "build_model",
    "build_total_loss",
    "Trainer",
    "TrainerConfig",
    "training_step",
    "Evaluator",
    "EvaluatorConfig",
    "evaluate_model",
    "RichLogger",
    "TensorboardLogger",
    "MultiLogger",
]
