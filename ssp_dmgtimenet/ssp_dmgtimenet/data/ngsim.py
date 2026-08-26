"""NGSIM raw trajectory loader exposing the :class:`HighDRecording` interface.

The NGSIM US-101 and I-80 freeway studies ship one CSV per 15-minute period
with per-frame vehicle state sampled at 10 Hz. The leading columns (in US
customary units: feet, ft/s, ft/s^2) are stable across every release::

    Vehicle_ID, Frame_ID, Total_Frames, Global_Time, Local_X, Local_Y,
    Global_X, Global_Y, v_Length, v_Width, v_Class, v_Vel, v_Acc, Lane_ID, ...

The trailing headway columns use inconsistent spellings between sites
(``Preceeding``/``Space_Hdwy`` on US-101 vs ``Preceding``/``Space_Headway`` on
I-80) and are not needed for platoon construction, so we ignore them.

To run the existing HighD platoon / windowing / evaluation pipeline unchanged
we map NGSIM onto the HighD schema:

* ``Local_Y`` (ft) -> ``x`` (m)            longitudinal, monotonically rising
* ``Local_X`` (ft) -> ``y`` (m)            lateral
* ``v_Vel``  (ft/s) -> ``xVelocity`` (m/s) speed, always non-negative
* ``v_Acc`` (ft/s^2) -> ``xAcceleration`` (m/s^2)
* ``v_Length`` (ft) -> ``width`` (m)       vehicle length (HighD ``width`` is along x)
* ``v_Width`` (ft) -> ``height`` (m)       vehicle lateral extent
* ``Lane_ID`` -> ``laneId``

All freeway vehicles travel toward increasing ``Local_Y``; this matches HighD's
``drivingDirection == 2`` convention (leader = largest ``x``), so the zero-shot
feature representation is identical to HighD direction-2 vehicles. The vehicle
class maps ``1/2/3 -> Motorcycle/Car/Truck`` and ``numLaneChanges`` is derived
from each vehicle's ``Lane_ID`` sequence (NGSIM ships no track-level metadata).
We never fabricate values: missing columns or empty CSVs raise immediately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

_log = logging.getLogger("ssp.ngsim")

FOOT_TO_METER: float = 0.3048

NGSIM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "Vehicle_ID",
    "Frame_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_Length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
)

NGSIM_CLASS_MAP: dict[int, str] = {1: "Motorcycle", 2: "Car", 3: "Truck"}

NGSIM_DRIVING_DIRECTION: int = 2

NGSIM_FRAME_RATE_HZ: float = 10.0

# (recording_id, site, period_dir) for the six scheme-C §6.2 freeway periods.
NGSIM_FREEWAY_PERIODS: tuple[tuple[int, str, str], ...] = (
    (101, "us101", "0750am-0805am"),
    (102, "us101", "0805am-0820am"),
    (103, "us101", "0820am-0835am"),
    (801, "i80", "0400pm-0415pm"),
    (802, "i80", "0500pm-0515pm"),
    (803, "i80", "0515pm-0530pm"),
)


def _check_columns(df: pd.DataFrame, expected: tuple[str, ...], path: Path) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")


@dataclass
class NGSIMRecording:
    """Lazily expose one NGSIM period CSV through the HighD recording API."""

    recording_id: int
    site: str
    period: str
    csv_path: Path
    frame_rate_hz: float = NGSIM_FRAME_RATE_HZ

    @cached_property
    def _raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        _check_columns(df, NGSIM_REQUIRED_COLUMNS, self.csv_path)
        df = df[list(NGSIM_REQUIRED_COLUMNS)].copy()
        for col in ("Vehicle_ID", "Frame_ID", "v_Class", "Lane_ID"):
            df[col] = df[col].astype(np.int64)
        n_before = len(df)
        df = df.drop_duplicates(subset=["Vehicle_ID", "Frame_ID"], keep="first")
        if len(df) != n_before:
            _log.warning(
                "%s: dropped %d duplicate (Vehicle_ID,Frame_ID) rows",
                self.csv_path.name,
                n_before - len(df),
            )
        df = df.sort_values(["Vehicle_ID", "Frame_ID"], kind="stable").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{self.csv_path.name} is empty")
        return df

    @property
    def frame_rate(self) -> float:
        return float(self.frame_rate_hz)

    @cached_property
    def tracks(self) -> pd.DataFrame:
        raw = self._raw
        return pd.DataFrame(
            {
                "frame": raw["Frame_ID"].to_numpy(np.int64),
                "id": raw["Vehicle_ID"].to_numpy(np.int64),
                "x": raw["Local_Y"].to_numpy(np.float64) * FOOT_TO_METER,
                "y": raw["Local_X"].to_numpy(np.float64) * FOOT_TO_METER,
                "width": raw["v_Length"].to_numpy(np.float64) * FOOT_TO_METER,
                "height": raw["v_Width"].to_numpy(np.float64) * FOOT_TO_METER,
                "xVelocity": raw["v_Vel"].to_numpy(np.float64) * FOOT_TO_METER,
                "xAcceleration": raw["v_Acc"].to_numpy(np.float64) * FOOT_TO_METER,
                "laneId": raw["Lane_ID"].to_numpy(np.int64),
                "drivingDirection": np.full(len(raw), NGSIM_DRIVING_DIRECTION, dtype=np.int64),
            }
        )

    @cached_property
    def tracks_meta(self) -> pd.DataFrame:
        raw = self._raw  # already sorted by (Vehicle_ID, Frame_ID)
        grouped = raw.groupby("Vehicle_ID", sort=True)
        v_class = grouped["v_Class"].first()
        unknown = sorted(set(v_class.unique()) - set(NGSIM_CLASS_MAP))
        if unknown:
            raise ValueError(f"{self.csv_path.name}: unknown v_Class values {unknown}")
        lane_changes = grouped["Lane_ID"].agg(
            lambda s: int(np.count_nonzero(np.diff(s.to_numpy()))) if len(s) > 1 else 0
        )
        meta = pd.DataFrame(
            {
                "id": v_class.index.to_numpy(np.int64),
                "width": grouped["v_Length"].first().to_numpy(np.float64) * FOOT_TO_METER,
                "height": grouped["v_Width"].first().to_numpy(np.float64) * FOOT_TO_METER,
                "initialFrame": grouped["Frame_ID"].min().to_numpy(np.int64),
                "finalFrame": grouped["Frame_ID"].max().to_numpy(np.int64),
                "numFrames": grouped["Frame_ID"].count().to_numpy(np.int64),
                "class": v_class.map(NGSIM_CLASS_MAP).to_numpy(object),
                "drivingDirection": np.full(len(v_class), NGSIM_DRIVING_DIRECTION, dtype=np.int64),
                "numLaneChanges": lane_changes.to_numpy(np.int64),
            }
        )
        return meta.reset_index(drop=True)

    def driving_direction_lookup(self) -> dict[int, int]:
        return {int(i): NGSIM_DRIVING_DIRECTION for i in self.tracks_meta["id"].tolist()}

    def vehicle_length_lookup(self) -> dict[int, float]:
        meta = self.tracks_meta
        return dict(zip(meta["id"].tolist(), meta["width"].tolist()))


def discover_ngsim_recordings(
    ngsim_root: str | Path,
    periods: tuple[tuple[int, str, str], ...] = NGSIM_FREEWAY_PERIODS,
) -> list[NGSIMRecording]:
    """Return one :class:`NGSIMRecording` per scheme-C §6.2 freeway period.

    ``ngsim_root`` is the ``vehicle-trajectory-data`` directory holding one
    sub-directory per period. Each period directory must contain exactly one
    ``trajectories-*.csv`` (RECONSTRUCTED variants are skipped here and loaded
    explicitly for the smoothing-sensitivity study).
    """

    root = Path(ngsim_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"NGSIM root not found: {root}")
    recordings: list[NGSIMRecording] = []
    for rec_id, site, period in periods:
        period_dir = root / period
        if not period_dir.is_dir():
            raise FileNotFoundError(f"NGSIM period directory missing: {period_dir}")
        matches = sorted(
            p for p in period_dir.glob("trajectories-*.csv") if not p.name.startswith("RECONSTRUCTED")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one trajectories-*.csv in {period_dir}, "
                f"found {[m.name for m in matches]}"
            )
        recordings.append(
            NGSIMRecording(recording_id=rec_id, site=site, period=period, csv_path=matches[0])
        )
    if not recordings:
        raise FileNotFoundError(f"No NGSIM recordings discovered under {root}")
    return recordings


def load_ngsim_recording(
    csv_path: str | Path,
    recording_id: int,
    site: str = "ngsim",
    period: str = "custom",
    frame_rate_hz: float = NGSIM_FRAME_RATE_HZ,
) -> NGSIMRecording:
    """Build a single :class:`NGSIMRecording` from an explicit CSV path."""

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NGSIM CSV not found: {path}")
    return NGSIMRecording(
        recording_id=recording_id,
        site=site,
        period=period,
        csv_path=path,
        frame_rate_hz=frame_rate_hz,
    )


# The Montanino-Punzo *reconstructed* release uses a different (10-column) schema
# than the raw NGSIM CSVs but keeps the same US customary units. Map the renamed
# kinematic columns onto the standard names so the rest of the loader is reused.
RECONSTRUCTED_COLUMN_MAP: dict[str, str] = {
    "Mean_Speed": "v_Vel",
    "Mean_Accel": "v_Acc",
    "Vehicle_Length": "v_Length",
    "Vehicle_Class_ID": "v_Class",
}


@dataclass
class ReconstructedNGSIMRecording(NGSIMRecording):
    """Montanino-Punzo *reconstructed* NGSIM CSV exposed through the HighD API.

    The reconstructed release (e.g. ``RECONSTRUCTED trajectories-400-0415_NO
    MOTORCYCLES.csv``) ships ten columns ``Vehicle_ID, Frame_ID, Lane_ID,
    Local_Y, Mean_Speed, Mean_Accel, Vehicle_Length, Vehicle_Class_ID,
    Follower_ID, Leader_ID`` in the *same* units (ft, ft/s, ft/s^2) as the raw
    release. It omits the lateral ``Local_X`` position, the ``v_Width`` extent and
    ``Global_Time``; the platoon / windowing pipeline never reads those channels
    (features use the longitudinal ``x``, speed, acceleration, vehicle length and
    lane only), so they are filled with ``NaN`` rather than fabricated. Every
    other step — unit conversion, leader ordering by ``Local_Y``, stationarity
    filtering — is identical to the raw recording, which is exactly what the §6.2
    smoothing-sensitivity study requires.
    """

    @cached_property
    def _raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df = df.rename(columns=RECONSTRUCTED_COLUMN_MAP)
        for absent in ("Local_X", "v_Width", "Global_Time"):
            if absent not in df.columns:
                df[absent] = np.nan
        _check_columns(df, NGSIM_REQUIRED_COLUMNS, self.csv_path)
        df = df[list(NGSIM_REQUIRED_COLUMNS)].copy()
        for col in ("Vehicle_ID", "Frame_ID", "v_Class", "Lane_ID"):
            df[col] = df[col].astype(np.int64)
        n_before = len(df)
        df = df.drop_duplicates(subset=["Vehicle_ID", "Frame_ID"], keep="first")
        if len(df) != n_before:
            _log.warning(
                "%s: dropped %d duplicate (Vehicle_ID,Frame_ID) rows",
                self.csv_path.name,
                n_before - len(df),
            )
        df = df.sort_values(["Vehicle_ID", "Frame_ID"], kind="stable").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"{self.csv_path.name} is empty")
        return df


def load_reconstructed_ngsim_recording(
    csv_path: str | Path,
    recording_id: int,
    site: str = "ngsim_reconstructed",
    period: str = "custom",
    frame_rate_hz: float = NGSIM_FRAME_RATE_HZ,
) -> ReconstructedNGSIMRecording:
    """Build a single :class:`ReconstructedNGSIMRecording` from an explicit CSV."""

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reconstructed NGSIM CSV not found: {path}")
    return ReconstructedNGSIMRecording(
        recording_id=recording_id,
        site=site,
        period=period,
        csv_path=path,
        frame_rate_hz=frame_rate_hz,
    )
