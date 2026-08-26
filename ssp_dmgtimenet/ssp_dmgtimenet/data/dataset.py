"""PyTorch ``Dataset`` and ``DataLoader`` builders for SSP-DMGTimeNet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .windowing import FEATURE_NAMES


@dataclass(slots=True, frozen=True)
class FeatureNormalisation:
    """Per-channel z-score using only history-window statistics."""

    mean: np.ndarray  # shape (F,)
    std: np.ndarray   # shape (F,)
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def to_tensor(self, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.as_tensor(self.mean, dtype=torch.float32, device=device),
            torch.as_tensor(self.std, dtype=torch.float32, device=device),
        )


class PlatoonDataset(Dataset):
    """Memory-mapped dataset backed by a single ``.npz`` produced by build script."""

    def __init__(
        self,
        npz_path: str | Path,
        normalisation: FeatureNormalisation | None = None,
        return_raw: bool = False,
    ) -> None:
        super().__init__()
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"Platoon dataset not found: {path}")
        self._path = path
        self._archive = np.load(path, allow_pickle=True, mmap_mode="r")
        self._cached: dict[str, np.ndarray] = {}
        self.normalisation = normalisation
        self.return_raw = return_raw

        self.history = self._array("history")
        self.future = self._array("future")
        self.history_mask = self._array("history_mask")
        self.future_mask = self._array("future_mask")
        self.vehicle_lengths = self._array("vehicle_lengths")
        self.track_ids = self._array("track_ids")
        self.recording_ids = self._array("recording_ids")
        self.lane_ids = self._array("lane_ids")
        self.start_times = self._array("start_times")
        feature_names_arr = self._archive["feature_names"]
        self.feature_names = tuple(str(x) for x in feature_names_arr.tolist())

        if self.history.shape[1:] != self.history_mask.shape[1:]:
            raise ValueError("history and history_mask shape mismatch")
        if self.future.shape[1:] != self.future_mask.shape[1:]:
            raise ValueError("future and future_mask shape mismatch")
        if self.history.shape[0] != self.future.shape[0]:
            raise ValueError("history and future have different sample counts")

    def _array(self, key: str) -> np.ndarray:
        if key not in self._cached:
            self._cached[key] = np.asarray(self._archive[key])
        return self._cached[key]

    def __len__(self) -> int:
        return int(self.history.shape[0])

    @property
    def num_vehicles(self) -> int:
        return int(self.history.shape[2])

    @property
    def num_features(self) -> int:
        return int(self.history.shape[3])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        history = torch.from_numpy(np.ascontiguousarray(self.history[idx]))
        future = torch.from_numpy(np.ascontiguousarray(self.future[idx]))
        history_mask = torch.from_numpy(np.ascontiguousarray(self.history_mask[idx]))
        future_mask = torch.from_numpy(np.ascontiguousarray(self.future_mask[idx]))
        vehicle_lengths = torch.from_numpy(np.ascontiguousarray(self.vehicle_lengths[idx]))

        sample: dict[str, torch.Tensor] = {
            "history_raw": history,
            "future_raw": future,
            "history_mask": history_mask,
            "future_mask": future_mask,
            "vehicle_lengths": vehicle_lengths,
            "track_ids": torch.as_tensor(self.track_ids[idx], dtype=torch.long),
            "recording_id": torch.as_tensor(self.recording_ids[idx], dtype=torch.long),
            "lane_id": torch.as_tensor(self.lane_ids[idx], dtype=torch.long),
            "start_time": torch.as_tensor(self.start_times[idx], dtype=torch.float32),
        }
        if self.normalisation is not None:
            mean, std = self.normalisation.to_tensor(device=history.device)
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)
            history_n = (history - mean) / std
            future_n = (future - mean) / std
            sample["history"] = history_n
            sample["future"] = future_n
        else:
            sample["history"] = history
            sample["future"] = future
        if not self.return_raw:
            sample.pop("history_raw")
            sample.pop("future_raw")
        return sample


def compute_history_normalisation(dataset: PlatoonDataset) -> FeatureNormalisation:
    """Compute per-channel mean/std from the *history* portion of training samples.

    Future windows must not leak into normalisation statistics.
    """

    history = dataset.history  # shape (B, T, N, F)
    mask = dataset.history_mask
    if history.shape != mask.shape:
        raise ValueError("history and history_mask shape mismatch")

    F_ = history.shape[-1]
    sums = np.zeros(F_, dtype=np.float64)
    sqsums = np.zeros(F_, dtype=np.float64)
    counts = np.zeros(F_, dtype=np.float64)
    for f in range(F_):
        m = mask[..., f].astype(np.float64)
        v = history[..., f].astype(np.float64)
        v = np.where(m > 0, v, 0.0)
        sums[f] = float(v.sum())
        sqsums[f] = float((v * v).sum())
        counts[f] = float(m.sum())
    counts = np.maximum(counts, 1.0)
    mean = sums / counts
    var = np.maximum(sqsums / counts - mean ** 2, 1e-6)
    std = np.sqrt(var)
    std = np.where(std < 1e-3, 1.0, std)
    return FeatureNormalisation(mean=mean.astype(np.float32), std=std.astype(np.float32))


def collate_platoon(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        items = [b[k] for b in batch]
        out[k] = torch.stack(items, dim=0)
    return out


def build_platoon_loaders(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path | None,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_raw: bool = False,
) -> tuple[dict[str, DataLoader], FeatureNormalisation, dict[str, PlatoonDataset]]:
    """Construct the train / val / test loaders with shared normalisation."""

    train_ds = PlatoonDataset(train_path, normalisation=None, return_raw=return_raw)
    norm = compute_history_normalisation(train_ds)
    train_ds.normalisation = norm
    val_ds = PlatoonDataset(val_path, normalisation=norm, return_raw=return_raw)

    datasets: dict[str, PlatoonDataset] = {"train": train_ds, "val": val_ds}
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_platoon,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_platoon,
            drop_last=False,
        ),
    }
    if test_path is not None:
        test_ds = PlatoonDataset(test_path, normalisation=norm, return_raw=return_raw)
        datasets["test"] = test_ds
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_platoon,
            drop_last=False,
        )
    return loaders, norm, datasets
