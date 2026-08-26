"""Create a reproducible data/checkpoint manifest for stability evaluation v3."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ssp_dmgtimenet"))

from ssp_dmgtimenet.data.dataset import (
    PlatoonDataset,
    compute_history_normalisation,
)
from ssp_dmgtimenet.metrics.stability import leader_excitation_amplitude

DATA_DIR = ROOT / "artifacts" / "platoons" / "highd_N5_h5_p3"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
OUTPUT_PATH = ROOT / "artifacts" / "evaluation_v3" / "dataset_manifest.json"
SPLITS = ("train", "val", "test")
DETREND_WINDOW_STEPS = 8
EXCITATION_FLOOR = 0.05


def fingerprint(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def split_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as archive:
        history = archive["history"]
        future = archive["future"]
        feature_names = tuple(str(value) for value in archive["feature_names"].tolist())
        return {
            **fingerprint(path),
            "n_windows": int(history.shape[0]),
            "history_shape": list(history.shape),
            "future_shape": list(future.shape),
            "recording_ids": sorted(np.unique(archive["recording_ids"]).astype(int).tolist()),
            "feature_names": list(feature_names),
        }


def checkpoint_normalisation(path: Path) -> dict[str, object]:
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state)
    keys = ("input_mean", "input_std", "output_mean", "output_std")
    present = all(key in state_dict for key in keys)
    result: dict[str, object] = {
        **fingerprint(path),
        "has_normalisation_buffers": present,
    }
    if present:
        result["normalisation"] = {
            key: state_dict[key].detach().cpu().numpy().tolist() for key in keys
        }
    return result


def main() -> None:
    splits = {
        split: split_metadata(DATA_DIR / f"{split}.npz")
        for split in SPLITS
    }
    test_path = DATA_DIR / "test.npz"
    with np.load(test_path, allow_pickle=True) as archive:
        feature_names = tuple(str(value) for value in archive["feature_names"].tolist())
        velocity_idx = feature_names.index("v")
        future_velocity = archive["future"][..., velocity_idx]
        gt_amplitude = leader_excitation_amplitude(
            future_velocity,
            detrend_window_steps=DETREND_WINDOW_STEPS,
        )
    train_norm = compute_history_normalisation(PlatoonDataset(DATA_DIR / "train.npz"))

    checkpoints: dict[str, object] = {}
    for checkpoint_path in sorted(CHECKPOINT_DIR.glob("*/best.pt")):
        if checkpoint_path.parent.name.startswith("n_ext_"):
            continue
        checkpoints[checkpoint_path.parent.name] = checkpoint_normalisation(checkpoint_path)

    payload = {
        "schema_version": "stability_evaluation_v3_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": {
            "historical_generation": {"n_test": 1847, "n_gt_excited": 62},
            "v3_generation": {
                "n_test": len(gt_amplitude),
                "n_gt_excited": int((gt_amplitude >= EXCITATION_FLOOR).sum()),
            },
            "historical_npz_recovered": False,
            "reason": (
                "The historical NPZ files were overwritten and no hash-verifiable backup "
                "was found; checkpoint normalisation differs from the current train split."
            ),
        },
        "protocol": {
            "detrend_window_steps": DETREND_WINDOW_STEPS,
            "excitation_floor_mps": EXCITATION_FLOOR,
        },
        "splits": splits,
        "current_train_normalisation": {
            "feature_names": list(train_norm.feature_names),
            "mean": train_norm.mean.tolist(),
            "std": train_norm.std.tolist(),
        },
        "checkpoints": checkpoints,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
