#!/usr/bin/env bash
# Evaluate HighD test split with the paper v3 stability protocol.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

OUTPUT_DIR="$ROOT/artifacts/evaluation_v3/reports"
LOG_DIR="$ROOT/artifacts/evaluation_v3/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PACKAGE_DIR"

eval_one() {
  local name=$1
  local config=$2
  local ckpt="$ROOT/artifacts/checkpoints/$name/best.pt"
  if [[ ! -f "$ckpt" ]]; then
    echo "[eval] missing checkpoint, skip $name: $ckpt" >&2
    return 0
  fi
  local report_dir="$OUTPUT_DIR/$name"
  mkdir -p "$report_dir"
  echo "[eval-highd] $name"
  python -m ssp_dmgtimenet.scripts.evaluate \
    --config "configs/$config" \
    --checkpoint "$ckpt" \
    --split test \
    --device "$DEVICE" \
    --num-workers "$EVAL_WORKERS" \
    --delta-unstable "$DELTA_UNSTABLE" \
    --excitation-floor "$EXCITATION_FLOOR" \
    --out-markdown "$report_dir/test_report.md" \
    >"$LOG_DIR/${name}.log" 2>&1
}

for item in "${MAIN_MODELS[@]}" "${ABLATION_MODELS[@]}"; do
  IFS="|" read -r name config <<<"$item"
  eval_one "$name" "$config"
done

echo "HighD evaluations written to $OUTPUT_DIR"
