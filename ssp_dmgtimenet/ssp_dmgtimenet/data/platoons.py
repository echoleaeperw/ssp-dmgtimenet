"""Continuous platoon discovery and stationarity filtering on HighD.

The pipeline follows scheme C, §4.4:

1. Iterate over every (frame, laneId) pair and rank vehicles along the
   driving direction (HighD encodes the direction sign in ``xVelocity``;
   we cross-check with ``drivingDirection`` from ``tracksMeta``).
2. For each (laneId, ordered tuple of N vehicle ids) compute the maximal
   set of consecutive frames where the same composition is present.
3. Within each compositionally-stable segment, also check ordering, gap
   non-negativity and the absence of lane changes.
4. Compute leader-based stationarity features so windows can later be
   filtered by ``std(v_1)``, ``std(a_1)`` and the velocity-drop magnitude.

The discovery is implemented as a single pass over a Pandas DataFrame to
keep memory predictable for HighD-scale recordings (typical: ~500k rows).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .highd import HighDRecording

_log = logging.getLogger("ssp.platoons")


@dataclass(slots=True, frozen=True)
class PlatoonExtractionConfig:
    """Hyper-parameters that control which platoons survive the audit."""

    min_N: int = 3
    max_N: int = 7
    min_segment_seconds: float = 10.0
    require_no_lane_change: bool = True
    require_positive_gap: bool = True
    require_consistent_ordering: bool = True
    drop_truck_leaders: bool = False
    leader_classes: tuple[str, ...] | None = None


@dataclass(slots=True, frozen=True)
class PlatoonStationarityThresholds:
    """Per-recording thresholds on leader (C1) statistics for §4.4 step 5."""

    std_v_quantile: float = 0.5
    std_a_quantile: float = 0.5
    velocity_drop_quantile: float = 0.5

    def quantiles(self) -> dict[str, float]:
        return {
            "std_v": float(self.std_v_quantile),
            "std_a": float(self.std_a_quantile),
            "velocity_drop": float(self.velocity_drop_quantile),
        }


@dataclass(slots=True)
class PlatoonSegment:
    """A platoon held compositionally stable for a contiguous frame range."""

    recording_id: int
    lane_id: int
    driving_direction: int
    track_ids: tuple[int, ...]
    start_frame: int
    end_frame: int
    frame_rate: float
    leader_length: float
    follower_lengths: tuple[float, ...]

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.frame_rate

    @property
    def N(self) -> int:
        return len(self.track_ids)

    @property
    def vehicle_lengths(self) -> tuple[float, ...]:
        return (self.leader_length, *self.follower_lengths)


def _ordered_lane_groups(tracks: pd.DataFrame, dir_lookup: dict[int, int]) -> pd.DataFrame:
    """For each (frame, laneId), produce a per-row rank along driving direction.

    Upper-lane vehicles travel left (larger ``x`` is towards the rear); we sort
    by increasing ``x`` so the leader (smallest ``x``) gets rank 0. Lower-lane
    vehicles travel right; we sort by decreasing ``x`` (via ``sort_x = -x``)
    so the leader (largest ``x``) gets rank 0.
    """

    if "drivingDirection" not in tracks.columns:
        tracks = tracks.copy()
        tracks["drivingDirection"] = tracks["id"].map(dir_lookup)
        if tracks["drivingDirection"].isna().any():
            missing = tracks.loc[tracks["drivingDirection"].isna(), "id"].unique().tolist()
            raise ValueError(f"Missing drivingDirection for trackIds {missing[:5]} (truncated)")
    df = tracks.copy()
    # Rank 0 = platoon leader (front-most vehicle in the driving direction).
    # HighD direction 1: xVelocity < 0, travelling toward smaller x → leader has smallest x → sort ascending.
    # HighD direction 2: xVelocity > 0, travelling toward larger  x → leader has largest  x → sort descending (negate).
    df["sort_x"] = np.where(df["drivingDirection"] == 2, -df["x"], df["x"])
    df = df.sort_values(["frame", "laneId", "sort_x"], kind="stable").reset_index(drop=True)
    df["lane_rank"] = df.groupby(["frame", "laneId"], sort=False).cumcount()
    return df.drop(columns=["sort_x"])


def _consecutive_runs(frames: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, end_inclusive)`` for every run of consecutive integers."""

    if frames.size == 0:
        return []
    if not np.all(np.diff(frames) >= 0):
        frames = np.sort(frames)
    breaks = np.where(np.diff(frames) != 1)[0]
    starts_idx = np.concatenate(([0], breaks + 1))
    ends_idx = np.concatenate((breaks, [frames.size - 1]))
    return [(int(frames[s]), int(frames[e])) for s, e in zip(starts_idx, ends_idx)]


def extract_platoon_segments(
    recording: HighDRecording,
    config: PlatoonExtractionConfig,
) -> list[PlatoonSegment]:
    """Find all compositionally-stable continuous platoons in a recording."""

    if config.min_N < 2:
        raise ValueError("min_N must be >= 2 to form a platoon")
    if config.max_N < config.min_N:
        raise ValueError("max_N must be >= min_N")

    tracks = recording.tracks
    meta = recording.tracks_meta
    fr = recording.frame_rate
    min_frames = max(2, int(np.ceil(config.min_segment_seconds * fr)))

    dir_lookup = recording.driving_direction_lookup()
    length_lookup = recording.vehicle_length_lookup()
    class_lookup = dict(zip(meta["id"].tolist(), meta["class"].tolist()))
    lane_change_lookup = dict(zip(meta["id"].tolist(), meta["numLaneChanges"].tolist()))

    if config.require_no_lane_change:
        keep_ids = {tid for tid, n_lc in lane_change_lookup.items() if n_lc == 0}
        if not keep_ids:
            return []
        tracks = tracks[tracks["id"].isin(keep_ids)].copy()

    ordered = _ordered_lane_groups(tracks, dir_lookup)

    segments: list[PlatoonSegment] = []
    grouped = ordered.groupby(["frame", "laneId"], sort=True)

    for N in range(config.min_N, config.max_N + 1):
        composition_frames: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
        composition_dir: dict[tuple[int, int, tuple[int, ...]], int] = {}

        for (frame, lane_id), group in grouped:
            if len(group) < N:
                continue
            ids = group["id"].to_numpy(dtype=np.int64)
            directions = group["drivingDirection"].to_numpy(dtype=np.int64)
            for offset in range(0, len(ids) - N + 1):
                comp = tuple(int(x) for x in ids[offset : offset + N])
                if len(set(comp)) != N:
                    continue
                if config.require_consistent_ordering:
                    dirs = directions[offset : offset + N]
                    if not np.all(dirs == dirs[0]):
                        continue
                key = (int(lane_id), N, comp)
                composition_frames.setdefault(key, []).append(int(frame))
                composition_dir.setdefault(key, int(directions[offset]))

        for (lane_id, comp_N, comp), frames_list in composition_frames.items():
            frames_arr = np.asarray(sorted(set(frames_list)), dtype=np.int64)
            for start_frame, end_frame in _consecutive_runs(frames_arr):
                if end_frame - start_frame + 1 < min_frames:
                    continue
                if config.drop_truck_leaders:
                    leader_class = class_lookup.get(comp[0], "Car")
                    if isinstance(leader_class, str) and leader_class.lower() != "car":
                        continue
                if config.leader_classes is not None:
                    leader_class = class_lookup.get(comp[0], None)
                    if leader_class not in config.leader_classes:
                        continue
                if config.require_positive_gap and not _gaps_positive(
                    tracks, comp, start_frame, end_frame, length_lookup
                ):
                    continue
                segments.append(
                    PlatoonSegment(
                        recording_id=recording.recording_id,
                        lane_id=lane_id,
                        driving_direction=composition_dir[(lane_id, comp_N, comp)],
                        track_ids=comp,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        frame_rate=fr,
                        leader_length=float(length_lookup.get(comp[0], np.nan)),
                        follower_lengths=tuple(float(length_lookup.get(t, np.nan)) for t in comp[1:]),
                    )
                )
    return segments


def _gaps_positive(
    tracks: pd.DataFrame,
    composition: tuple[int, ...],
    start_frame: int,
    end_frame: int,
    length_lookup: dict[int, float],
) -> bool:
    """Verify s_i = x_{i-1} - x_i - len_{i-1} > 0 across the segment."""

    sub = tracks[
        tracks["id"].isin(composition)
        & (tracks["frame"] >= start_frame)
        & (tracks["frame"] <= end_frame)
    ]
    if sub.empty:
        return False
    pivot = sub.pivot(index="frame", columns="id", values="x").reindex(columns=list(composition))
    if pivot.isna().any().any():
        return False
    direction = int(np.sign(sub["xVelocity"].iloc[0])) or 1
    arr = pivot.to_numpy(dtype=np.float64)
    if direction > 0:
        gaps = -np.diff(arr, axis=1)
    else:
        gaps = np.diff(arr, axis=1)
    leader_lens = np.asarray([length_lookup[t] for t in composition[:-1]], dtype=np.float64)
    gaps = gaps - leader_lens
    return bool(np.all(gaps > 0))


def leader_stationarity_features(
    recording: HighDRecording,
    segment: PlatoonSegment,
) -> dict[str, float]:
    """Compute std(v_1), std(a_1), velocity drop and v range for the leader."""

    leader = segment.track_ids[0]
    tracks = recording.tracks
    sub = tracks[
        (tracks["id"] == leader)
        & (tracks["frame"] >= segment.start_frame)
        & (tracks["frame"] <= segment.end_frame)
    ].sort_values("frame")
    if sub.empty:
        raise ValueError(f"Leader {leader} has no rows in [{segment.start_frame}, {segment.end_frame}]")
    v = np.abs(sub["xVelocity"].to_numpy(dtype=np.float64))
    a = sub["xAcceleration"].to_numpy(dtype=np.float64)
    return {
        "std_v": float(np.std(v, ddof=0)),
        "std_a": float(np.std(a, ddof=0)),
        "v_range": float(np.max(v) - np.min(v)),
        "v_min": float(np.min(v)),
        "velocity_drop": float(np.maximum(0.0, np.max(v) - v[-1])),
    }


def filter_segments_by_stationarity(
    recording: HighDRecording,
    segments: Iterable[PlatoonSegment],
    thresholds: PlatoonStationarityThresholds,
) -> list[PlatoonSegment]:
    """Per-recording quantile filter as required by scheme C §4.4 step 5."""

    materialised = list(segments)
    if not materialised:
        return []
    feats = pd.DataFrame([leader_stationarity_features(recording, seg) for seg in materialised])
    cutoffs = {
        "std_v": float(np.quantile(feats["std_v"], 1 - thresholds.std_v_quantile)),
        "std_a": float(np.quantile(feats["std_a"], 1 - thresholds.std_a_quantile)),
        "velocity_drop": float(np.quantile(feats["velocity_drop"], 1 - thresholds.velocity_drop_quantile)),
    }
    keep_mask = (
        (feats["std_v"] >= cutoffs["std_v"])
        | (feats["std_a"] >= cutoffs["std_a"])
        | (feats["velocity_drop"] >= cutoffs["velocity_drop"])
    )
    return [seg for seg, keep in zip(materialised, keep_mask.tolist()) if bool(keep)]


@dataclass(slots=True)
class AuditRow:
    recording_id: int
    N: int
    threshold_quantile: float
    num_segments: int
    total_segment_seconds: float
    median_segment_seconds: float


def audit_platoon_counts(
    recordings: Iterable[HighDRecording],
    config: PlatoonExtractionConfig,
    quantiles: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25),
) -> pd.DataFrame:
    """Run the §4.4 audit across multiple N values and quantile thresholds.

    ``quantiles`` here are interpreted as "keep top-q fraction of segments by
    leader stationarity"; ``q=1.0`` disables the filter and counts everything.
    """

    recordings_list = list(recordings)
    rows: list[AuditRow] = []
    total_recs = len(recordings_list)
    for idx, rec in enumerate(recordings_list, start=1):
        _log.info(
            "Audit: recording %d/%d (HighD id=%02d), extracting platoons…",
            idx,
            total_recs,
            rec.recording_id,
        )
        all_segments = extract_platoon_segments(rec, config)
        _log.info(
            "Audit: recording %02d raw segments=%d",
            rec.recording_id,
            len(all_segments),
        )
        per_n: dict[int, list[PlatoonSegment]] = {}
        for seg in all_segments:
            per_n.setdefault(seg.N, []).append(seg)
        for N in range(config.min_N, config.max_N + 1):
            segs_N = per_n.get(N, [])
            if not segs_N:
                for q in quantiles:
                    rows.append(AuditRow(rec.recording_id, N, float(q), 0, 0.0, 0.0))
                continue
            feats = pd.DataFrame([leader_stationarity_features(rec, seg) for seg in segs_N])
            durations = np.asarray([s.duration_sec for s in segs_N], dtype=np.float64)
            for q in quantiles:
                if q >= 1.0:
                    keep = np.ones(len(segs_N), dtype=bool)
                else:
                    cutoffs = {
                        "std_v": float(np.quantile(feats["std_v"], 1 - q)),
                        "std_a": float(np.quantile(feats["std_a"], 1 - q)),
                        "velocity_drop": float(np.quantile(feats["velocity_drop"], 1 - q)),
                    }
                    keep = (
                        (feats["std_v"] >= cutoffs["std_v"])
                        | (feats["std_a"] >= cutoffs["std_a"])
                        | (feats["velocity_drop"] >= cutoffs["velocity_drop"])
                    ).to_numpy()
                kept_durations = durations[keep]
                rows.append(
                    AuditRow(
                        recording_id=rec.recording_id,
                        N=N,
                        threshold_quantile=float(q),
                        num_segments=int(keep.sum()),
                        total_segment_seconds=float(kept_durations.sum()),
                        median_segment_seconds=float(np.median(kept_durations)) if kept_durations.size else 0.0,
                    )
                )
    df = pd.DataFrame([asdict(row) for row in rows])
    return df.sort_values(["recording_id", "N", "threshold_quantile"]).reset_index(drop=True)


def split_recordings_for_training(
    recordings: list[HighDRecording],
    train_ids: Iterable[int] = range(1, 46),
    val_ids: Iterable[int] = range(46, 51),
    test_ids: Iterable[int] = range(51, 61),
) -> dict[str, list[HighDRecording]]:
    """Partition HighD recordings according to scheme C §4.4 step 6."""

    train_set = {int(i) for i in train_ids}
    val_set = {int(i) for i in val_ids}
    test_set = {int(i) for i in test_ids}
    out: dict[str, list[HighDRecording]] = {"train": [], "val": [], "test": []}
    for rec in recordings:
        if rec.recording_id in train_set:
            out["train"].append(rec)
        elif rec.recording_id in val_set:
            out["val"].append(rec)
        elif rec.recording_id in test_set:
            out["test"].append(rec)
    return out


def iter_recording_segments(
    recordings: Iterable[HighDRecording],
    config: PlatoonExtractionConfig,
    thresholds: PlatoonStationarityThresholds | None = None,
) -> Iterator[tuple[HighDRecording, list[PlatoonSegment]]]:
    for rec in recordings:
        segs = extract_platoon_segments(rec, config)
        if thresholds is not None:
            segs = filter_segments_by_stationarity(rec, segs, thresholds)
        yield rec, segs


def write_audit_report(audit_df: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    """Persist the audit table as CSV plus a per-(N, quantile) Markdown summary."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "platoon_audit.csv"
    audit_df.to_csv(csv_path, index=False)

    summary = (
        audit_df.groupby(["N", "threshold_quantile"], sort=True)
        .agg(
            recordings=("recording_id", "nunique"),
            num_segments=("num_segments", "sum"),
            total_segment_seconds=("total_segment_seconds", "sum"),
            median_segment_seconds=("median_segment_seconds", "median"),
        )
        .reset_index()
    )
    md_lines = [
        "# HighD Platoon Audit",
        "",
        "Auto-generated by `scripts/audit_highd_platoons.py`.",
        "",
        "| N | quantile | recordings | num_segments | total_seconds | median_segment_seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {int(row.N)} | {row.threshold_quantile:.2f} | {int(row.recordings)} | "
            f"{int(row.num_segments)} | {row.total_segment_seconds:.1f} | {row.median_segment_seconds:.2f} |"
        )
    md_path = out / "platoon_audit.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return csv_path, md_path
