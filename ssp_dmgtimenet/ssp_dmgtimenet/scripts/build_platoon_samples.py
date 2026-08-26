"""Materialise platoon samples from HighD recordings.

The script:

1. Discovers HighD recordings under ``--highd-root`` and partitions them
   into train/val/test according to the scheme-C §4.4 split.
2. Extracts compositionally-stable platoon segments of length ``--N``.
3. Optionally filters segments by a per-recording leader-stationarity
   quantile (``--nonstationary-quantile``).
4. Resamples each surviving segment to ``--target-hz``, low-pass filters
   speed/acceleration before downsampling, and slices history+predict
   windows.
5. Writes one ``.npz`` per split into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from ..data.highd import discover_highd_recordings, load_highd_recording
from ..data.platoons import (
    PlatoonExtractionConfig,
    PlatoonStationarityThresholds,
    extract_platoon_segments,
    filter_segments_by_stationarity,
    split_recordings_for_training,
)
from ..data.windowing import WindowingConfig, resample_segment_to_target_hz, slice_windows, stack_samples
from ..utils.io import save_npz


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HighD platoon samples (scheme C §4.4).")
    parser.add_argument("--highd-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--history-sec", type=float, default=5.0)
    parser.add_argument("--predict-sec", type=float, default=3.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--minimum-total-seconds", type=float, default=10.0)
    parser.add_argument("--nonstationary-quantile", type=float, default=0.5)
    parser.add_argument("--train-ids", type=int, nargs="+", default=list(range(1, 46)))
    parser.add_argument("--val-ids", type=int, nargs="+", default=list(range(46, 51)))
    parser.add_argument("--test-ids", type=int, nargs="+", default=list(range(51, 61)))
    parser.add_argument("--require-no-lane-change", action="store_true", default=True)
    parser.add_argument("--no-require-no-lane-change", action="store_true")
    parser.add_argument("--require-positive-gap", action="store_true", default=True)
    parser.add_argument("--no-require-positive-gap", action="store_true")
    parser.add_argument("--leader-classes", type=str, nargs="+", default=None)
    parser.add_argument("--max-recordings-per-split", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.build")

    extraction = PlatoonExtractionConfig(
        min_N=args.N,
        max_N=args.N,
        min_segment_seconds=args.minimum_total_seconds,
        require_no_lane_change=not args.no_require_no_lane_change,
        require_positive_gap=not args.no_require_positive_gap,
        leader_classes=tuple(args.leader_classes) if args.leader_classes else None,
    )
    thresholds = (
        PlatoonStationarityThresholds(
            std_v_quantile=args.nonstationary_quantile,
            std_a_quantile=args.nonstationary_quantile,
            velocity_drop_quantile=args.nonstationary_quantile,
        )
        if args.nonstationary_quantile < 1.0
        else None
    )
    windowing = WindowingConfig(
        target_hz=args.target_hz,
        history_seconds=args.history_sec,
        predict_seconds=args.predict_sec,
        stride_seconds=args.stride_sec,
        minimum_total_seconds=args.minimum_total_seconds,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    available = discover_highd_recordings(args.highd_root)
    available_ids = {rec.recording_id for rec in available}
    log.info("Discovered %d recordings under %s", len(available_ids), args.highd_root)
    splits = split_recordings_for_training(
        available,
        train_ids=args.train_ids,
        val_ids=args.val_ids,
        test_ids=args.test_ids,
    )
    summary: dict[str, int] = {}
    for split_name, recs in splits.items():
        if not recs:
            log.warning("Split %r has no available recordings; skipping", split_name)
            continue
        if args.max_recordings_per_split is not None:
            recs = recs[: args.max_recordings_per_split]
        samples_for_split: list = []
        for rec in recs:
            segments = extract_platoon_segments(rec, extraction)
            if thresholds is not None:
                segments = filter_segments_by_stationarity(rec, segments, thresholds)
            log.info(
                "Recording %02d → %d platoon segments (post-stationarity)", rec.recording_id, len(segments)
            )
            for segment in segments:
                features, vehicle_lengths = resample_segment_to_target_hz(rec, segment, windowing)
                samples = slice_windows(features, vehicle_lengths, segment, windowing)
                samples_for_split.extend(samples)
        if not samples_for_split:
            log.warning("Split %r produced no samples after filtering", split_name)
            summary[split_name] = 0
            continue
        stacked = stack_samples(samples_for_split)
        out_path = out_dir / f"{split_name}.npz"
        save_npz(out_path, **stacked)
        log.info("Wrote %s (%d samples)", out_path, stacked["history"].shape[0])
        summary[split_name] = stacked["history"].shape[0]
    log.info("Summary: %s", summary)
    np.save(out_dir / "summary.npy", np.asarray(summary, dtype=object))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
