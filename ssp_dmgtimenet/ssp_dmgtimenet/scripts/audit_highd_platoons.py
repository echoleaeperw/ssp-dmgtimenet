"""Run the HighD platoon audit and write CSV/Markdown reports.

The script discovers every ``XX_tracks.csv`` under ``--highd-root`` and
computes how many continuous platoons of length ``N in [min_N, max_N]``
survive each stationarity threshold quantile. Output goes into
``--report-dir`` and is the basis for the scheme-C §4.4 Go/No-Go decision.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..data.highd import discover_highd_recordings
from ..data.platoons import (
    PlatoonExtractionConfig,
    audit_platoon_counts,
    write_audit_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HighD platoon audit (scheme C §4.4).")
    parser.add_argument("--highd-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--min-N", type=int, default=3)
    parser.add_argument("--max-N", type=int, default=7)
    parser.add_argument("--min-segment-seconds", type=float, default=10.0)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--no-require-no-lane-change", action="store_true")
    parser.add_argument("--require-positive-gap", action="store_true", default=True)
    parser.add_argument("--no-require-positive-gap", action="store_true")
    parser.add_argument("--leader-classes", type=str, nargs="+", default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(asctime)s][%(levelname)s] %(message)s")
    log = logging.getLogger("ssp.audit")

    extraction_config = PlatoonExtractionConfig(
        min_N=args.min_N,
        max_N=args.max_N,
        min_segment_seconds=args.min_segment_seconds,
        require_no_lane_change=not args.no_require_no_lane_change,
        require_positive_gap=not args.no_require_positive_gap,
        leader_classes=tuple(args.leader_classes) if args.leader_classes else None,
    )
    log.info("Discovering HighD recordings under %s", args.highd_root)
    recordings = discover_highd_recordings(args.highd_root)
    log.info("Found %d recordings", len(recordings))
    df = audit_platoon_counts(
        recordings,
        config=extraction_config,
        quantiles=tuple(args.quantiles),
    )
    csv_path, md_path = write_audit_report(df, args.report_dir)
    log.info("Wrote audit CSV: %s", csv_path)
    log.info("Wrote audit Markdown: %s", md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
