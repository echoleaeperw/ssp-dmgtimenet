"""HighD raw CSV loader.

The official HighD release ships three CSVs per recording:

* ``XX_recordingMeta.csv``   metadata for the whole recording (e.g. frame
  rate, lane markings, speed limit).
* ``XX_tracksMeta.csv``      one row per ``trackId``: vehicle class, driving
  direction, life-cycle frame range, lane-change count, ...
* ``XX_tracks.csv``          per-frame state for every ``trackId``.

This module gives a thin, schema-checked loader so downstream code can rely
on column names, dtypes and units. We do not invent any data: missing columns
or empty CSVs raise immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


HIGHD_TRACKS_COLUMNS: tuple[str, ...] = (
    "frame",
    "id",
    "x",
    "y",
    "width",
    "height",
    "xVelocity",
    "yVelocity",
    "xAcceleration",
    "yAcceleration",
    "frontSightDistance",
    "backSightDistance",
    "dhw",
    "thw",
    "ttc",
    "precedingXVelocity",
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
    "laneId",
)

HIGHD_TRACKS_META_COLUMNS: tuple[str, ...] = (
    "id",
    "width",
    "height",
    "initialFrame",
    "finalFrame",
    "numFrames",
    "class",
    "drivingDirection",
    "traveledDistance",
    "minXVelocity",
    "maxXVelocity",
    "meanXVelocity",
    "minDHW",
    "minTHW",
    "minTTC",
    "numLaneChanges",
)

HIGHD_RECORDING_META_COLUMNS: tuple[str, ...] = (
    "id",
    "frameRate",
    "locationId",
    "speedLimit",
    "month",
    "weekDay",
    "startTime",
    "duration",
    "totalDrivenDistance",
    "totalDrivenTime",
    "numVehicles",
    "numCars",
    "numTrucks",
    "upperLaneMarkings",
    "lowerLaneMarkings",
)


def _check_columns(df: pd.DataFrame, expected: tuple[str, ...], path: Path) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")


@dataclass
class HighDRecording:
    """Container that lazily exposes the three HighD tables for one recording."""

    recording_id: int
    tracks_path: Path
    tracks_meta_path: Path
    recording_meta_path: Path

    @cached_property
    def recording_meta(self) -> pd.Series:
        df = pd.read_csv(self.recording_meta_path)
        _check_columns(df, HIGHD_RECORDING_META_COLUMNS, self.recording_meta_path)
        if len(df) != 1:
            raise ValueError(f"{self.recording_meta_path.name} should have exactly one row, got {len(df)}")
        return df.iloc[0]

    @cached_property
    def tracks_meta(self) -> pd.DataFrame:
        df = pd.read_csv(self.tracks_meta_path)
        _check_columns(df, HIGHD_TRACKS_META_COLUMNS, self.tracks_meta_path)
        df = df.copy()
        df["id"] = df["id"].astype(np.int64)
        df["initialFrame"] = df["initialFrame"].astype(np.int64)
        df["finalFrame"] = df["finalFrame"].astype(np.int64)
        df["numFrames"] = df["numFrames"].astype(np.int64)
        df["drivingDirection"] = df["drivingDirection"].astype(np.int64)
        df["numLaneChanges"] = df["numLaneChanges"].astype(np.int64)
        return df

    @cached_property
    def tracks(self) -> pd.DataFrame:
        df = pd.read_csv(self.tracks_path)
        _check_columns(df, HIGHD_TRACKS_COLUMNS, self.tracks_path)
        df = df.copy()
        int_cols = [
            "frame",
            "id",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftAlongsideId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightAlongsideId",
            "rightFollowingId",
            "laneId",
        ]
        for col in int_cols:
            df[col] = df[col].astype(np.int64)
        if df.empty:
            raise ValueError(f"{self.tracks_path.name} is empty")
        return df

    @property
    def frame_rate(self) -> float:
        return float(self.recording_meta["frameRate"])

    def driving_direction_lookup(self) -> dict[int, int]:
        return dict(zip(self.tracks_meta["id"].tolist(), self.tracks_meta["drivingDirection"].tolist()))

    def vehicle_length_lookup(self) -> dict[int, float]:
        return dict(zip(self.tracks_meta["id"].tolist(), self.tracks_meta["width"].tolist()))


def discover_highd_recordings(highd_root: str | Path) -> list[HighDRecording]:
    """Return one ``HighDRecording`` per ``XX_tracks.csv`` found under ``highd_root``."""

    root = Path(highd_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"HighD root not found: {root}")

    recordings: list[HighDRecording] = []
    for tracks_path in sorted(root.glob("*_tracks.csv")):
        rec_id_str = tracks_path.stem.split("_")[0]
        try:
            rec_id = int(rec_id_str)
        except ValueError as exc:
            raise ValueError(f"Cannot parse recording id from {tracks_path.name}") from exc
        meta_path = tracks_path.parent / f"{rec_id_str}_tracksMeta.csv"
        rec_meta_path = tracks_path.parent / f"{rec_id_str}_recordingMeta.csv"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing tracksMeta for recording {rec_id_str}: {meta_path}")
        if not rec_meta_path.is_file():
            raise FileNotFoundError(f"Missing recordingMeta for recording {rec_id_str}: {rec_meta_path}")
        recordings.append(
            HighDRecording(
                recording_id=rec_id,
                tracks_path=tracks_path,
                tracks_meta_path=meta_path,
                recording_meta_path=rec_meta_path,
            )
        )
    if not recordings:
        raise FileNotFoundError(f"No HighD recordings found under {root}. Check that *_tracks.csv files exist.")
    return recordings


def load_highd_recording(highd_root: str | Path, recording_id: int) -> HighDRecording:
    """Load a single HighD recording by integer id (e.g. 1..60)."""

    root = Path(highd_root).expanduser().resolve()
    rec_id_str = f"{recording_id:02d}"
    tracks_path = root / f"{rec_id_str}_tracks.csv"
    if not tracks_path.is_file():
        raise FileNotFoundError(f"HighD recording {rec_id_str} not found: {tracks_path}")
    return HighDRecording(
        recording_id=recording_id,
        tracks_path=tracks_path,
        tracks_meta_path=root / f"{rec_id_str}_tracksMeta.csv",
        recording_meta_path=root / f"{rec_id_str}_recordingMeta.csv",
    )


def iterate_recordings(highd_root: str | Path, recording_ids: list[int] | None = None) -> Iterator[HighDRecording]:
    if recording_ids is None:
        yield from discover_highd_recordings(highd_root)
        return
    for rec_id in recording_ids:
        yield load_highd_recording(highd_root, rec_id)
