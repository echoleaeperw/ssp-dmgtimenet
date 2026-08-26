"""Train SSP-DMGTimeNet (or any registered baseline) from a YAML config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ..data.dataset import build_platoon_loaders
from ..training.evaluator import EvaluatorConfig, write_report_markdown
from ..training.factory import build_model, build_total_loss, initialise_normalisation_for_model
from ..training.logging import MultiLogger, RichLogger, TensorboardLogger
from ..training.trainer import Trainer, TrainerConfig
from ..utils.config import Config, load_config
from ..utils.seed import seed_everything


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSP-DMGTimeNet or a baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _resolve_paths(cfg: Config) -> dict[str, Path]:
    paths = cfg.get("paths", None)
    if paths is None:
        raise KeyError("Config is missing required 'paths' section")
    out = {
        "train": Path(paths.get("train")),
        "val": Path(paths.get("val")),
        "test": Path(paths.get("test")) if paths.get("test") else None,
        "checkpoint_dir": Path(paths.get("checkpoint_dir")) if paths.get("checkpoint_dir") else None,
        "tensorboard_dir": Path(paths.get("tensorboard_dir")) if paths.get("tensorboard_dir") else None,
        "report_dir": Path(paths.get("report_dir")) if paths.get("report_dir") else None,
    }
    return out


def _build_trainer_config(cfg: Config, checkpoint_dir: Path | None) -> TrainerConfig:
    section = cfg.get("trainer", {})
    return TrainerConfig(
        epochs=int(section.get("epochs", 60)),
        learning_rate=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
        grad_clip_norm=float(section.get("grad_clip_norm", 1.0)),
        warmup_epochs=int(section.get("warmup_epochs", 2)),
        eval_every=int(section.get("eval_every", 1)),
        log_every=int(section.get("log_every", 50)),
        use_amp=bool(section.get("use_amp", True)),
        early_stop_patience=int(section.get("early_stop_patience", 10)),
        early_stop_metric=str(section.get("early_stop_metric", "acc/mae_v")),
        early_stop_mode=str(section.get("early_stop_mode", "min")),
        best_min_progress=float(section.get("best_min_progress", 0.0)),
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        save_best=bool(section.get("save_best", True)),
        save_last=bool(section.get("save_last", True)),
        seed=int(section.get("seed", 42)),
        eval_after_stage=bool(section.get("eval_after_stage", True)),
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.train")

    cfg = load_config(args.config)
    paths = _resolve_paths(cfg)
    if args.checkpoint_dir is not None:
        paths["checkpoint_dir"] = args.checkpoint_dir
    seed = args.seed if args.seed is not None else int(cfg.get("trainer", {}).get("seed", 42))
    seed_everything(seed)

    loaders, normalisation, datasets = build_platoon_loaders(
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        batch_size=int(cfg.get("trainer", {}).get("batch_size", 64)),
        num_workers=args.num_workers,
        return_raw=True,
    )
    log.info(
        "Loaded train=%d val=%d test=%s",
        len(datasets["train"]),
        len(datasets["val"]),
        len(datasets["test"]) if "test" in datasets else "n/a",
    )

    model = build_model(cfg["model"])
    output_var_indices = []
    output_channels = cfg.get("loss", {}).get("prediction", {}).get("variables") or [
        "v",
        "s",
        "a",
        "x_rel_leader",
    ]
    for name in output_channels:
        if name not in normalisation.feature_names:
            raise KeyError(f"Output channel {name!r} missing from feature_names")
        output_var_indices.append(normalisation.feature_names.index(name))
    output_mean = torch.as_tensor(normalisation.mean[output_var_indices], dtype=torch.float32)
    output_std = torch.as_tensor(normalisation.std[output_var_indices], dtype=torch.float32)
    input_mean = torch.as_tensor(normalisation.mean, dtype=torch.float32)
    input_std = torch.as_tensor(normalisation.std, dtype=torch.float32)
    initialise_normalisation_for_model(model, input_mean, input_std, output_mean, output_std)

    loss_fn = build_total_loss(cfg.get("loss", {}))
    loss_fn.prediction.set_normalisation(output_mean, output_std)
    loss_fn.kinematics.set_normalisation(output_std)
    evaluator_cfg = EvaluatorConfig(
        target_hz=float(cfg.get("data", {}).get("target_hz", 10.0)),
        detrend_window_steps=int(cfg.get("loss", {}).get("stability", {}).get("detrend_window_steps", 8)),
        horizons_seconds=tuple(float(x) for x in cfg.get("evaluator", {}).get("horizons_seconds", (1.0, 2.0, 3.0))),
        fft_band_hz=tuple(
            float(x)
            for x in cfg.get("loss", {}).get("stability", {}).get("fft_band_hz", (0.05, 0.5))
        ),
        delta_unstable=float(cfg.get("evaluator", {}).get("delta_unstable", 0.0)),
        excitation_floor=float(
            cfg.get("evaluator", {}).get(
                "excitation_floor",
                cfg.get("loss", {}).get("stability", {}).get("excitation_floor", 0.05),
            )
        ),
        output_channels=tuple(output_channels),
    )

    trainer_cfg = _build_trainer_config(cfg, paths["checkpoint_dir"])

    loggers: list = [RichLogger(name=cfg.get("model", {}).get("name", "ssp"))]
    if paths["tensorboard_dir"] is not None:
        loggers.append(TensorboardLogger(paths["tensorboard_dir"]))
    multi_logger = MultiLogger(loggers)

    device = torch.device(args.device)
    trainer = Trainer(
        config=trainer_cfg,
        model=model,
        loss_fn=loss_fn,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        logger=multi_logger,
        evaluator_config=evaluator_cfg,
    )
    final_report = trainer.fit()
    log.info("Final validation report:\n%s", final_report.accuracy)
    if paths["report_dir"] is not None:
        write_report_markdown(final_report, paths["report_dir"] / "final_val.md")
    multi_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
