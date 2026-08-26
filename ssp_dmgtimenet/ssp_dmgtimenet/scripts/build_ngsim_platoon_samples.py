"""Materialise NGSIM zero-shot platoon samples (scheme C §6.2 Step 2).

This mirrors :mod:`scripts.build_platoon_samples` but targets the NGSIM US-101 /
I-80 freeway periods returned by :func:`data.ngsim.discover_ngsim_recordings`.
For each site it:

1. Extracts compositionally-stable ``N``-vehicle platoons with the *same*
   protocol as the HighD main experiment (no lane change, positive gaps,
   consistent ordering) via :func:`extract_platoon_segments`.
2. Keeps the non-stationary subset at ``--nonstationary-quantile`` (default 0.5,
   matching the HighD main split) via :func:`filter_segments_by_stationarity`.
3. Resamples each surviving segment and slices history+predict windows. NGSIM is
   natively 10 Hz so the temporal grid is unchanged, but the Savitzky-Golay +
   low-pass smoothing of v/a still runs so the feature construction is identical
   to the HighD pipeline.
4. Deterministically down-samples the per-site window pool to
   ``--max-windows-per-site`` evenly across all periods/segments, then writes
   ``<out-dir>/<site>/test.npz`` (plus the full pool when the cap is disabled).

Normalisation is deliberately *not* applied here. Zero-shot evaluation reads the
HighD train-split statistics (see :func:`data.dataset.build_platoon_loaders`,
driven by ``scripts.evaluate`` with ``paths.train`` pointing at the HighD
``train.npz``), so the NGSIM tensors must stay in absolute physical units with a
schema identical to the HighD ``test.npz``.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.ngsim import (
    discover_ngsim_recordings,
    load_ngsim_recording,
    load_reconstructed_ngsim_recording,
)
from ..data.platoons import (
    PlatoonExtractionConfig,
    PlatoonStationarityThresholds,
    extract_platoon_segments,
    filter_segments_by_stationarity,
)
from ..data.windowing import (
    WindowingConfig,
    resample_segment_to_target_hz,
    slice_windows,
    stack_samples,
)
from ..utils.io import save_npz

_log = logging.getLogger("ssp.build_ngsim")


def _evenly_spaced_indices(n: int, k: int) -> np.ndarray:
    """Return up to ``k`` evenly spaced unique indices into ``range(n)``.

    Sampling spans the entire ``[0, n-1]`` range so the retained windows cover
    every period and segment proportionally to its window count. The selection
    is fully deterministic (no RNG) so the NGSIM test set is reproducible.
    """

    if k <= 0:
        raise ValueError(f"max-windows must be positive, got {k}")
    if k >= n:
        return np.arange(n, dtype=np.int64)
    idx = np.linspace(0, n - 1, num=k)
    return np.unique(np.rint(idx).astype(np.int64))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NGSIM zero-shot platoon samples (scheme C §6.2).")
    parser.add_argument("--ngsim-root", type=Path, default=None, help="vehicle-trajectory-data directory (discover mode)")
    parser.add_argument("--out-dir", type=Path, required=True, help="root dir; one <site>/test.npz is written per site")
    parser.add_argument(
        "--recording-csv",
        type=Path,
        default=None,
        help="single-file mode: build from this explicit CSV (e.g. a RECONSTRUCTED file)",
    )
    parser.add_argument("--recording-id", type=int, default=None, help="single-file mode: recording id to tag windows with")
    parser.add_argument("--site-name", type=str, default=None, help="single-file mode: output subdir name under out-dir")
    parser.add_argument("--period-name", type=str, default="custom", help="single-file mode: period label")
    parser.add_argument(
        "--reconstructed",
        action="store_true",
        help="single-file mode: use the Montanino-Punzo reconstructed schema loader",
    )
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--history-sec", type=float, default=5.0)
    parser.add_argument("--predict-sec", type=float, default=3.0)
    parser.add_argument("--stride-sec", type=float, default=1.0)
    parser.add_argument("--minimum-total-seconds", type=float, default=10.0)
    parser.add_argument("--nonstationary-quantile", type=float, default=0.5)
    parser.add_argument(
        "--max-windows-per-site",
        type=int,
        default=4000,
        help="evenly-spaced down-sampling cap per site; set <=0 to keep every window",
    )
    parser.add_argument("--sites", type=str, nargs="+", default=None, help="restrict to these sites (default: all)")
    parser.add_argument("--require-no-lane-change", action="store_true", default=True)
    parser.add_argument("--no-require-no-lane-change", action="store_true")
    parser.add_argument("--require-positive-gap", action="store_true", default=True)
    parser.add_argument("--no-require-positive-gap", action="store_true")
    parser.add_argument("--leader-classes", type=str, nargs="+", default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")

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

    if args.recording_csv is not None:
        if args.recording_id is None or args.site_name is None:
            raise ValueError("single-file mode requires --recording-id and --site-name")
        loader = load_reconstructed_ngsim_recording if args.reconstructed else load_ngsim_recording
        rec = loader(
            args.recording_csv,
            recording_id=args.recording_id,
            site=args.site_name,
            period=args.period_name,
        )
        by_site: dict[str, list] = {args.site_name: [rec]}
        _log.info(
            "Single-file mode: %s (id=%d, site=%s, reconstructed=%s)",
            args.recording_csv,
            args.recording_id,
            args.site_name,
            args.reconstructed,
        )
    else:
        if args.ngsim_root is None:
            raise ValueError("--ngsim-root is required unless --recording-csv is given")
        recordings = discover_ngsim_recordings(args.ngsim_root)
        by_site = {}
        for rec in recordings:
            by_site.setdefault(rec.site, []).append(rec)
        if args.sites is not None:
            wanted = set(args.sites)
            missing = wanted - set(by_site)
            if missing:
                raise ValueError(f"Requested sites {sorted(missing)} not in discovered sites {sorted(by_site)}")
            by_site = {s: by_site[s] for s in by_site if s in wanted}
        _log.info("Discovered NGSIM sites: %s", {s: [r.period for r in recs] for s, recs in by_site.items()})

    summary: dict[str, dict] = {}
    for site, recs in sorted(by_site.items()):
        samples_for_site: list = []
        raw_per_period: Counter[int] = Counter()
        for rec in recs:
            segments = extract_platoon_segments(rec, extraction)
            if thresholds is not None:
                segments = filter_segments_by_stationarity(rec, segments, thresholds)
            _log.info(
                "%s/%s (id=%d): %d platoon segments (post-stationarity q=%.2f)",
                site,
                rec.period,
                rec.recording_id,
                len(segments),
                args.nonstationary_quantile,
            )
            for segment in segments:
                features, vehicle_lengths = resample_segment_to_target_hz(rec, segment, windowing)
                windows = slice_windows(features, vehicle_lengths, segment, windowing)
                samples_for_site.extend(windows)
                raw_per_period[rec.recording_id] += len(windows)

        n_raw = len(samples_for_site)
        if n_raw == 0:
            raise ValueError(f"Site {site!r} produced no windows; check extraction/quantile settings")

        if args.max_windows_per_site > 0 and n_raw > args.max_windows_per_site:
            keep_idx = _evenly_spaced_indices(n_raw, args.max_windows_per_site)
            samples_for_site = [samples_for_site[i] for i in keep_idx]
            _log.info(
                "%s: down-sampling %d -> %d windows (evenly spaced)",
                site,
                n_raw,
                len(samples_for_site),
            )
        else:
            _log.info("%s: keeping all %d windows (no cap)", site, n_raw)

        stacked = stack_samples(samples_for_site)
        out_path = out_dir / site / "test.npz"
        save_npz(out_path, **stacked)

        kept_per_period = Counter(int(r) for r in stacked["recording_ids"].tolist())
        _log.info(
            "Wrote %s (%d samples; kept per period %s of raw %s)",
            out_path,
            stacked["history"].shape[0],
            dict(sorted(kept_per_period.items())),
            dict(sorted(raw_per_period.items())),
        )
        summary[site] = {
            "n_windows_raw": int(n_raw),
            "n_windows_kept": int(stacked["history"].shape[0]),
            "raw_per_period": {int(k): int(v) for k, v in sorted(raw_per_period.items())},
            "kept_per_period": {int(k): int(v) for k, v in sorted(kept_per_period.items())},
            "out_path": str(out_path),
        }

    np.save(out_dir / "summary.npy", np.asarray(summary, dtype=object))
    _log.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
