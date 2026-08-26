"""Evaluate a trained model on a held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..data.dataset import build_platoon_loaders
from ..training.evaluator import EvaluatorConfig, evaluate_model, report_to_flat_dict, write_report_markdown
from ..training.factory import build_model, initialise_normalisation_for_model
from ..utils.config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SSP-DMGTimeNet or a baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--test-path",
        type=Path,
        default=None,
        help=(
            "Override paths.test for cross-dataset zero-shot evaluation "
            "(e.g. NGSIM). Normalisation statistics still come from paths.train."
        ),
    )
    parser.add_argument("--out-markdown", type=Path, default=None)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write machine-readable metrics and evaluation provenance.",
    )
    parser.add_argument(
        "--delta-unstable",
        type=float,
        default=None,
        help="Override the instability margin delta (threshold = 1 + delta).",
    )
    parser.add_argument(
        "--excitation-floor",
        type=float,
        default=None,
        help="Override the detrended leader RMS excitation floor in m/s.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _file_fingerprint(path: Path) -> dict[str, str | int]:
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.eval")

    cfg = load_config(args.config)
    paths_section = cfg.get("paths", {})
    train_path = Path(paths_section["train"])
    val_path = Path(paths_section["val"])
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = Path(paths_section["test"]) if paths_section.get("test") else None

    loaders, normalisation, _ = build_platoon_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=int(cfg.get("trainer", {}).get("batch_size", 64)),
        num_workers=args.num_workers,
        return_raw=True,
    )
    if args.split not in loaders:
        raise KeyError(f"Split {args.split!r} not in loaders {list(loaders)}")
    loader = loaders[args.split]

    model = build_model(cfg["model"])
    state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)

    output_channels = cfg.get("loss", {}).get("prediction", {}).get("variables") or [
        "v",
        "s",
        "a",
        "x_rel_leader",
    ]
    output_var_indices = [normalisation.feature_names.index(name) for name in output_channels]
    output_mean = torch.as_tensor(normalisation.mean[output_var_indices], dtype=torch.float32)
    output_std = torch.as_tensor(normalisation.std[output_var_indices], dtype=torch.float32)
    input_mean = torch.as_tensor(normalisation.mean, dtype=torch.float32)
    input_std = torch.as_tensor(normalisation.std, dtype=torch.float32)
    normalisation_keys = {"input_mean", "input_std", "output_mean", "output_std"}
    checkpoint_has_normalisation = normalisation_keys.issubset(state_dict)
    if not checkpoint_has_normalisation:
        initialise_normalisation_for_model(model, input_mean, input_std, output_mean, output_std)
    effective_input_mean = getattr(model, "input_mean", input_mean)
    effective_input_std = getattr(model, "input_std", input_std)
    effective_output_mean = getattr(model, "output_mean", output_mean)
    effective_output_std = getattr(model, "output_std", output_std)

    device = torch.device(args.device)
    model = model.to(device)
    evaluator_cfg = EvaluatorConfig(
        target_hz=float(cfg.get("data", {}).get("target_hz", 10.0)),
        detrend_window_steps=int(cfg.get("loss", {}).get("stability", {}).get("detrend_window_steps", 8)),
        horizons_seconds=tuple(
            float(x)
            for x in cfg.get("evaluator", {}).get(
                "horizons_seconds", (1.0, 2.0, 3.0)
            )
        ),
        fft_band_hz=tuple(
            float(x)
            for x in cfg.get("loss", {}).get("stability", {}).get("fft_band_hz", (0.05, 0.5))
        ),
        delta_unstable=(
            float(args.delta_unstable)
            if args.delta_unstable is not None
            else float(cfg.get("evaluator", {}).get("delta_unstable", 0.0))
        ),
        excitation_floor=float(
            args.excitation_floor
            if args.excitation_floor is not None
            else cfg.get("evaluator", {}).get(
                "excitation_floor",
                cfg.get("loss", {}).get("stability", {}).get(
                    "excitation_floor", 0.05
                ),
            )
        ),
        output_channels=tuple(output_channels),
    )
    report = evaluate_model(model, loader, evaluator_cfg, device=device)
    flat = report_to_flat_dict(report)
    log.info("Evaluation results:")
    for k in sorted(flat):
        log.info("  %s = %.6f", k, flat[k])
    if args.out_markdown is not None:
        write_report_markdown(report, args.out_markdown)
        log.info("Markdown report written to %s", args.out_markdown)
    out_json = args.out_json
    if out_json is None and args.out_markdown is not None:
        out_json = args.out_markdown.with_suffix(".json")
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "stability_protocol_v3",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split,
            "config": _file_fingerprint(args.config),
            "checkpoint": _file_fingerprint(args.checkpoint),
            "evaluation_data": _file_fingerprint(
                test_path if args.split == "test" else Path(paths_section[args.split])
            ),
            "normalisation_source": (
                "checkpoint_buffers" if checkpoint_has_normalisation else "current_train_data"
            ),
            "normalisation_data": (
                _file_fingerprint(args.checkpoint)
                if checkpoint_has_normalisation
                else _file_fingerprint(train_path)
            ),
            "normalisation": {
                "feature_names": list(normalisation.feature_names),
                "input_mean": effective_input_mean.detach().cpu().numpy().tolist(),
                "input_std": effective_input_std.detach().cpu().numpy().tolist(),
                "output_channels": list(output_channels),
                "output_mean": effective_output_mean.detach().cpu().numpy().tolist(),
                "output_std": effective_output_std.detach().cpu().numpy().tolist(),
            },
            "evaluator": asdict(evaluator_cfg),
            "metrics": flat,
        }
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        log.info("JSON report written to %s", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
