"""Training loop for SSP-DMGTimeNet and the scheme-C baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..losses.total import TotalLoss, attention_blocks_of
from .curriculum import progress_for_epoch, report_curriculum
from .evaluator import Evaluator, EvaluatorConfig, EvaluationReport, report_to_flat_dict, write_report_markdown
from .logging import MultiLogger


@dataclass(slots=True, frozen=True)
class TrainerConfig:
    epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_epochs: int = 2
    eval_every: int = 1
    log_every: int = 50
    use_amp: bool = True
    early_stop_patience: int = 10
    early_stop_metric: str = "acc/v"
    early_stop_mode: str = "min"
    best_min_progress: float = 0.0
    checkpoint_dir: str | None = None
    save_best: bool = True
    save_last: bool = True
    seed: int = 42
    eval_after_stage: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _resolve_attention_blocks(model: nn.Module):
    if hasattr(model, "sp_daca_blocks"):
        return attention_blocks_of(model)
    return []


# SP-DACA delay-prior parameters: weight decay drags their logits towards 0,
# which IS the sigmoid mid-point (tau = (tau_min+tau_max)/2), silently erasing
# the learned delay structure (observed in v5: all 48 tau_logits collapsed to
# ~0). These must always live in a decay-free parameter group.
_NO_DECAY_SUFFIXES = ("tau_logits", "sigma_logits", "time_pe_scale")


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    no_decay: list[torch.nn.Parameter] = []
    decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(_NO_DECAY_SUFFIXES):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, Any]] = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def training_step(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    loss_fn: TotalLoss,
    optimizer: Optimizer,
    progress: float,
    grad_clip_norm: float,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> dict[str, torch.Tensor]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model(batch["history_raw"], batch["history_mask"])
            attention_blocks = _resolve_attention_blocks(model)
            loss, diag = loss_fn(output, batch, attention_blocks=attention_blocks, progress=progress)
        scaler.scale(loss).backward()
        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(batch["history_raw"], batch["history_mask"])
        attention_blocks = _resolve_attention_blocks(model)
        loss, diag = loss_fn(output, batch, attention_blocks=attention_blocks, progress=progress)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    diag["loss_total"] = loss.detach()
    return diag


class Trainer:
    """Drive the training loop with curriculum-aware loss scheduling."""

    def __init__(
        self,
        config: TrainerConfig,
        model: nn.Module,
        loss_fn: TotalLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        logger: MultiLogger,
        evaluator_config: EvaluatorConfig,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.evaluator = Evaluator(evaluator_config)

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, config.weight_decay),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.epochs * max(1, len(train_loader)) - config.warmup_epochs * max(1, len(train_loader))),
        )
        self.scaler: torch.cuda.amp.GradScaler | None = None
        if config.use_amp and device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        self._best_metric: float | None = None
        self._patience: int = 0
        self._global_step: int = 0
        self._epoch: int = 0
        self._checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
        if self._checkpoint_dir is not None:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_metric(self) -> float | None:
        return self._best_metric

    def fit(self) -> EvaluationReport:
        last_report: EvaluationReport | None = None
        for epoch in range(self.config.epochs):
            self._epoch = epoch
            progress = progress_for_epoch(epoch, self.config.epochs)
            curriculum = report_curriculum(progress, self.loss_fn.config.curriculum)
            self.logger.log_text(
                "curriculum",
                f"epoch={epoch} progress={progress:.3f} stage={curriculum.stage.name} weights={curriculum.weights}",
                self._global_step,
            )

            self._train_one_epoch(progress)

            if (epoch + 1) % self.config.eval_every == 0 or (epoch + 1) == self.config.epochs:
                report = self.evaluator.report(self.model, self.val_loader, device=self.device)
                last_report = report
                flat = report_to_flat_dict(report)
                self.logger.log_scalars("val", flat, self._global_step)
                self._maybe_checkpoint(report, flat, progress)
                if self._early_stop_triggered(flat):
                    self.logger.log_text(
                        "early_stop",
                        f"early stopping at epoch={epoch}, best_metric={self._best_metric}",
                        self._global_step,
                    )
                    break
        if last_report is None:
            raise RuntimeError("Training finished without producing an evaluation report")
        return last_report

    def _train_one_epoch(self, progress: float) -> None:
        for step, batch in enumerate(self.train_loader):
            batch = _move_batch(batch, self.device)
            diag = training_step(
                self.model,
                batch,
                self.loss_fn,
                self.optimizer,
                progress=progress,
                grad_clip_norm=self.config.grad_clip_norm,
                scaler=self.scaler,
            )
            warmup_complete = self._epoch >= self.config.warmup_epochs
            if warmup_complete:
                self.scheduler.step()
            else:
                self._linear_warmup(self._epoch, step)
            self._global_step += 1
            if self._global_step % self.config.log_every == 0:
                scalars = {k: float(v.item() if isinstance(v, torch.Tensor) else v) for k, v in diag.items()}
                scalars["lr"] = self.optimizer.param_groups[0]["lr"]
                self.logger.log_scalars("train", scalars, self._global_step)

    def _linear_warmup(self, epoch: int, step_in_epoch: int) -> None:
        steps_per_epoch = max(1, len(self.train_loader))
        total_warmup = max(1, self.config.warmup_epochs * steps_per_epoch)
        current = epoch * steps_per_epoch + step_in_epoch + 1
        scale = float(min(1.0, current / total_warmup))
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * scale

    def _maybe_checkpoint(self, report: EvaluationReport, flat: Mapping[str, float], progress: float = 1.0) -> None:
        if self._checkpoint_dir is None:
            return
        if self.config.save_last:
            self._save("last.pt", report)
        if progress < self.config.best_min_progress:
            # Best-checkpoint selection and early-stop patience only start once
            # the curriculum has reached the configured progress, so that the
            # selected model reflects the fully ramped loss.
            return
        metric_key = self.config.early_stop_metric
        if metric_key not in flat:
            return
        value = float(flat[metric_key])
        better = (
            self._best_metric is None
            or (self.config.early_stop_mode == "min" and value < self._best_metric)
            or (self.config.early_stop_mode == "max" and value > self._best_metric)
        )
        if better:
            self._best_metric = value
            self._patience = 0
            if self.config.save_best:
                self._save("best.pt", report)
            write_report_markdown(report, self._checkpoint_dir / "best_report.md")
        else:
            self._patience += 1

    def _save(self, name: str, report: EvaluationReport) -> Path:
        if self._checkpoint_dir is None:
            raise RuntimeError("checkpoint_dir is not configured")
        path = self._checkpoint_dir / name
        torch.save(
            {
                "epoch": self._epoch,
                "global_step": self._global_step,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loss_fn_threshold": float(self.loss_fn.excitation_gate.threshold),
                "best_metric": self._best_metric,
                "evaluation": report_to_flat_dict(report),
            },
            path,
        )
        return path

    def _early_stop_triggered(self, flat: Mapping[str, float]) -> bool:
        return self._patience >= self.config.early_stop_patience
