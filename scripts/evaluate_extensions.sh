#!/usr/bin/env bash
# Zero-shot NGSIM, I-80 smoothing sensitivity, and N-extension evaluations.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

OUTPUT_DIR="$ROOT/artifacts/evaluation_v3/extensions"
LOG_DIR="$ROOT/artifacts/evaluation_v3/logs/extensions"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PACKAGE_DIR"

run_eval() {
  local tag=$1
  local config=$2
  local checkpoint=$3
  local output=$4
  local test_path=${5:-}
  if [[ ! -f "$checkpoint" ]]; then
    echo "[eval] missing checkpoint, skip $tag: $checkpoint" >&2
    return 0
  fi
  mkdir -p "$output"
  local command=(
    python -m ssp_dmgtimenet.scripts.evaluate
    --config "$config"
    --checkpoint "$checkpoint"
    --split test
    --device "$DEVICE"
    --num-workers "$EVAL_WORKERS"
    --delta-unstable "$DELTA_UNSTABLE"
    --excitation-floor "$EXCITATION_FLOOR"
    --out-markdown "$output/test_report.md"
  )
  if [[ -n "$test_path" ]]; then
    if [[ ! -f "$test_path" ]]; then
      echo "[eval] missing test npz, skip $tag: $test_path" >&2
      return 0
    fi
    command+=(--test-path "$test_path")
  fi
  echo "[eval-ext] $tag"
  "${command[@]}" >"$LOG_DIR/${tag}.log" 2>&1
}

for site in us101 i80; do
  for item in "${MAIN_MODELS[@]}"; do
    IFS="|" read -r model config <<<"$item"
    run_eval \
      "ngsim_${site}_${model}" \
      "configs/$config" \
      "$ROOT/artifacts/checkpoints/$model/best.pt" \
      "$OUTPUT_DIR/ngsim_${site}/$model" \
      "$ROOT/artifacts/platoons/ngsim_N5_h5_p3/${site}/test.npz"
  done
done

for site in i80_orig_0400 i80_recon_0400; do
  for item in "${MAIN_MODELS[@]}"; do
    IFS="|" read -r model config <<<"$item"
    run_eval \
      "sensitivity_${site}_${model}" \
      "configs/$config" \
      "$ROOT/artifacts/checkpoints/$model/best.pt" \
      "$OUTPUT_DIR/sensitivity_${site}/$model" \
      "$ROOT/artifacts/platoons/ngsim_sensitivity/${site}/test.npz"
  done
done

for N in 3 6 7; do
  for item in "${N_EXT_MODELS[@]}"; do
    IFS="|" read -r model suffix <<<"$item"
    run_eval \
      "n_ext_N${N}_${model}" \
      "configs/n_ext_N${N}_${suffix}.yaml" \
      "$ROOT/artifacts/checkpoints/n_ext_N${N}/$model/best.pt" \
      "$OUTPUT_DIR/n_ext_N${N}/$model"
  done
done

for item in "${N_EXT_MODELS[@]}"; do
  IFS="|" read -r model suffix <<<"$item"
  case "$model" in
    ssp_dmgtimenet_v6) config="ssp_dmgtimenet_v6.yaml" ;;
    interaction_lstm) config="baseline_int_lstm.yaml" ;;
    platoon_transformer) config="baseline_transformer.yaml" ;;
  esac
  run_eval \
    "n_ext_N5_${model}" \
    "configs/$config" \
    "$ROOT/artifacts/checkpoints/$model/best.pt" \
    "$OUTPUT_DIR/n_ext_N5/$model"
done

echo "Extension evaluations written to $OUTPUT_DIR"
