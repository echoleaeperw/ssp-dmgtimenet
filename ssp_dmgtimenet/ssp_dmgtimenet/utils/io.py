"""IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p


def load_npz(path: str | Path, keys: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        names = list(data.files) if keys is None else list(keys)
        return {name: data[name] for name in names}
