"""Audit and summarize NGSIM, sensitivity, and platoon-length v3 reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "artifacts" / "evaluation_v3" / "extensions"
OUTPUT_PATH = ROOT / "artifacts" / "evaluation_v3" / "extension_tables.md"
AUDIT_PATH = ROOT / "artifacts" / "evaluation_v3" / "extension_audit.json"

MODELS = [
    ("SSP-DMGTimeNet", "ssp_dmgtimenet_v6"),
    ("Int-LSTM", "interaction_lstm"),
    ("Transformer", "platoon_transformer"),
    ("Full-graph Attention", "full_graph_attention"),
    ("LSTM", "platoon_lstm"),
    ("CNN-Int-LSTM-IDM", "cnn_int_lstm_idm"),
    ("IDM cascade", "idm_cascade"),
    ("DMGTimeNet cascade", "dmg_cascade"),
    ("OVM cascade", "ovm_cascade"),
    ("FVDM cascade", "fvdm_cascade"),
]
N_MODELS = MODELS[:3]


def load(group: str, directory: str) -> dict[str, object]:
    path = EXT_DIR / group / directory / "test_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def value(report: dict[str, object], key: str) -> float:
    metrics = report["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be a mapping")
    return float(metrics[key])


def number(metric_value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(metric_value) else f"{metric_value:.{digits}f}"


def percent(metric_value: float) -> str:
    return "—" if not math.isfinite(metric_value) else f"{100.0 * metric_value:.2f}%"


def audit_group(group: str, models: list[tuple[str, str]]) -> dict[str, object]:
    errors: list[str] = []
    hashes: set[str] = set()
    totals: set[int] = set()
    gt_counts: set[int] = set()
    for _, directory in models:
        report = load(group, directory)
        data = report["evaluation_data"]
        if not isinstance(data, dict):
            errors.append(f"{group}/{directory}: missing evaluation_data")
            continue
        hashes.add(str(data["sha256"]))
        totals.add(round(value(report, "stab/detection_n_total")))
        gt_counts.add(round(value(report, "stab/detection_n_gt_excited")))
        if value(report, "stab/gt_ref_n_windows") != value(
            report, "stab/detection_n_gt_excited"
        ):
            errors.append(f"{group}/{directory}: external support differs from GT support")
        if value(report, "stab/conditional_internal_n_windows") != value(
            report, "stab/detection_tp"
        ):
            errors.append(f"{group}/{directory}: conditional support differs from TP")
    if len(hashes) != 1 or len(totals) != 1 or len(gt_counts) != 1:
        errors.append(f"{group}: models do not share data/support")
    return {
        "group": group,
        "n_total": next(iter(totals)) if len(totals) == 1 else sorted(totals),
        "n_gt_excited": next(iter(gt_counts)) if len(gt_counts) == 1 else sorted(gt_counts),
        "data_sha256": sorted(hashes),
        "passed": not errors,
        "errors": errors,
    }


def group_table(title: str, group: str, models: list[tuple[str, str]]) -> list[str]:
    first = load(group, models[0][1])
    n_total = round(value(first, "stab/detection_n_total"))
    n_gt = round(value(first, "stab/detection_n_gt_excited"))
    lines = [
        f"## {title}",
        "",
        f"> total={n_total}; GT-excited={n_gt}.",
        "",
        (
            "| Model | v-MAE | coverage | FPR | external unstable | external p95 | "
            "conditional n | conditional unstable |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, directory in models:
        report = load(group, directory)
        row = [
            name,
            number(value(report, "acc/v")),
            percent(value(report, "stab/detection_coverage")),
            percent(value(report, "stab/detection_fpr")),
            percent(value(report, "stab/gt_ref_unstable_window_ratio")),
            number(value(report, "stab/gt_ref_p95_gain")),
            str(round(value(report, "stab/conditional_internal_n_windows"))),
            percent(value(report, "stab/conditional_internal_unstable_window_ratio")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> None:
    groups = [
        ("NGSIM US-101 zero-shot", "ngsim_us101", MODELS),
        ("NGSIM I-80 zero-shot", "ngsim_i80", MODELS),
        ("I-80 original sensitivity", "sensitivity_i80_orig_0400", MODELS),
        ("I-80 reconstructed sensitivity", "sensitivity_i80_recon_0400", MODELS),
        ("N=3 extension", "n_ext_N3", N_MODELS),
        ("N=5 extension", "n_ext_N5", N_MODELS),
        ("N=6 extension", "n_ext_N6", N_MODELS),
        ("N=7 extension", "n_ext_N7", N_MODELS),
    ]
    audits = [audit_group(group, models) for _, group, models in groups]
    failures = [error for result in audits for error in result["errors"]]
    if failures:
        raise RuntimeError("\n".join(failures))

    lines = [
        "# Stability evaluation v3 extensions",
        "",
        (
            "> All groups use floor=0.05 m/s and instability threshold=1.05. "
            "Support is audited independently within each dataset/platoon length."
        ),
        "",
    ]
    for title, group, models in groups:
        lines.extend(group_table(title, group, models))
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    AUDIT_PATH.write_text(
        json.dumps(
            {"passed": True, "groups": audits},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"written: {OUTPUT_PATH}")
    print(f"audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
