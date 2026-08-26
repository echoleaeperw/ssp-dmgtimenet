"""Audit v3 JSON reports and build compact paper-facing tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "artifacts" / "evaluation_v3" / "reports"
OUTPUT_PATH = ROOT / "artifacts" / "evaluation_v3" / "tables.md"
AUDIT_PATH = ROOT / "artifacts" / "evaluation_v3" / "report_audit.json"

MAIN_MODELS = [
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

ABLATIONS = [
    ("Full model", "ssp_dmgtimenet_v6"),
    ("w/o delay bias", "ablation_wo_delay_bias"),
    ("w/o adjacent loss", "ablation_wo_adj"),
    ("w/o CFE", "ablation_wo_cfe"),
    ("Full-graph attention", "ablation_full_graph"),
    ("w/o sub-platoon loss", "ablation_wo_sub"),
    ("w/o HGF", "ablation_wo_hgf"),
    ("Fixed delay", "ablation_fixed_tau"),
    ("w/o FFT loss", "ablation_wo_fft"),
]


def load_report(directory: str) -> dict[str, object]:
    path = REPORTS_DIR / directory / "test_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict[str, object], key: str) -> float:
    metrics = report["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("report.metrics must be a mapping")
    return float(metrics[key])


def number(value: float, digits: int = 3) -> str:
    return "—" if not math.isfinite(value) else f"{value:.{digits}f}"


def percent(value: float) -> str:
    return "—" if not math.isfinite(value) else f"{100.0 * value:.2f}%"


def integer(value: float) -> str:
    return "—" if not math.isfinite(value) else str(round(value))


def accuracy_row(display_name: str, report: dict[str, object]) -> str:
    values = [
        display_name,
        number(metric(report, "acc/v")),
        number(metric(report, "acc/rmse_v")),
        number(metric(report, "acc/s")),
        number(metric(report, "acc/a")),
        number(metric(report, "tail/mae_v")),
    ]
    return "| " + " | ".join(values) + " |"


def stability_row(display_name: str, report: dict[str, object]) -> str:
    values = [
        display_name,
        percent(metric(report, "stab/detection_coverage")),
        percent(metric(report, "stab/detection_fpr")),
        percent(metric(report, "stab/gt_ref_unstable_window_ratio")),
        number(metric(report, "stab/gt_ref_p95_gain")),
        number(metric(report, "stab/gt_ref_max_gain")),
        integer(metric(report, "stab/conditional_internal_n_windows")),
        percent(metric(report, "stab/conditional_internal_unstable_window_ratio")),
        number(metric(report, "stab/conditional_internal_p95_gain")),
        number(metric(report, "stab/conditional_internal_max_gain")),
    ]
    return "| " + " | ".join(values) + " |"


def audit(reports: dict[str, dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    evaluation_hashes: set[str] = set()
    for directory, report in reports.items():
        if report.get("schema_version") != "stability_protocol_v3":
            errors.append(f"{directory}: wrong schema_version")
        data = report.get("evaluation_data")
        if not isinstance(data, dict):
            errors.append(f"{directory}: missing evaluation_data")
            continue
        evaluation_hashes.add(str(data.get("sha256")))
        expected = {
            "stab/detection_n_total": 2150.0,
            "stab/detection_n_gt_excited": 66.0,
            "stab/gt_ref_n_windows": 66.0,
            "stab/gt_ref_fft_n_windows": 66.0,
        }
        for key, target in expected.items():
            value = metric(report, key)
            if value != target:
                errors.append(f"{directory}: {key}={value}, expected {target}")
        conditional_n = metric(report, "stab/conditional_internal_n_windows")
        detection_tp = metric(report, "stab/detection_tp")
        if conditional_n != detection_tp:
            errors.append(
                f"{directory}: conditional_internal_n_windows={conditional_n} "
                f"!= detection_tp={detection_tp}"
            )
        evaluator = report.get("evaluator")
        if not isinstance(evaluator, dict):
            errors.append(f"{directory}: missing evaluator metadata")
        else:
            if float(evaluator.get("delta_unstable", -1.0)) != 0.05:
                errors.append(f"{directory}: delta_unstable is not 0.05")
            if float(evaluator.get("excitation_floor", -1.0)) != 0.05:
                errors.append(f"{directory}: excitation_floor is not 0.05")
    if len(evaluation_hashes) != 1:
        errors.append(f"evaluation-data hashes differ: {sorted(evaluation_hashes)}")
    return {
        "n_reports": len(reports),
        "evaluation_data_sha256": sorted(evaluation_hashes),
        "passed": not errors,
        "errors": errors,
    }


def main() -> None:
    directories = {directory for _, directory in MAIN_MODELS + ABLATIONS}
    reports = {directory: load_report(directory) for directory in sorted(directories)}
    audit_result = audit(reports)
    if not audit_result["passed"]:
        raise RuntimeError("\n".join(audit_result["errors"]))

    lines = [
        "# Stability evaluation v3 tables",
        "",
        (
            "> HighD N=5 test: 2150 windows; GT-excited support: 66 windows; "
            "detrended-RMS floor: 0.05 m/s; instability threshold: 1.05."
        ),
        "",
        "## Accuracy",
        "",
        "| Model | v-MAE | v-RMSE | s-MAE | a-MAE | tail v-MAE |",
        "|---|---:|---:|---:|---:|---:|",
        *[
            accuracy_row(name, reports[directory])
            for name, directory in MAIN_MODELS
        ],
        "",
        "## Three-layer stability protocol",
        "",
        (
            "| Model | coverage | FPR | external unstable | external p95 | external max | "
            "conditional n/66 | conditional unstable | conditional p95 | conditional max |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *[
            stability_row(name, reports[directory])
            for name, directory in MAIN_MODELS
        ],
        "",
        (
            "External metrics use all 66 GT-excited windows and the GT leader denominator. "
            "Conditional metrics use the predicted leader denominator only on "
            "`GT excited AND prediction excited`; `n` must accompany every conditional rate."
        ),
        "",
        "## Ablation",
        "",
        (
            "| Variant | v-MAE | coverage | FPR | external p95 | external max | "
            "conditional n/66 | conditional unstable | conditional max |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, directory in ABLATIONS:
        report = reports[directory]
        values = [
            name,
            number(metric(report, "acc/v")),
            percent(metric(report, "stab/detection_coverage")),
            percent(metric(report, "stab/detection_fpr")),
            number(metric(report, "stab/gt_ref_p95_gain")),
            number(metric(report, "stab/gt_ref_max_gain")),
            integer(metric(report, "stab/conditional_internal_n_windows")),
            percent(metric(report, "stab/conditional_internal_unstable_window_ratio")),
            number(metric(report, "stab/conditional_internal_max_gain")),
        ]
        lines.append("| " + " | ".join(values) + " |")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    AUDIT_PATH.write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"written: {OUTPUT_PATH}")
    print(f"audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
