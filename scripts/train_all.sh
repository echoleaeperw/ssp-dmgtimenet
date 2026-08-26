#!/usr/bin/env bash
# Train paper models. Usage:
#   ./scripts/train_all.sh                 # main + ablation + N-extension
#   ./scripts/train_all.sh main
#   ./scripts/train_all.sh ablation
#   ./scripts/train_all.sh n_ext
#   ./scripts/train_all.sh ssp_dmgtimenet_v6
# Resume: SKIP_EXISTING=1 ./scripts/train_all.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

GROUP="${1:-all}"
LOG_DIR="$ROOT/artifacts/logs/train"
mkdir -p "$LOG_DIR"
cd "$PACKAGE_DIR"

train_one() {
  local name=$1
  local config=$2
  local ckpt="$ROOT/artifacts/checkpoints/$name/best.pt"
  if skip_if_exists "$ckpt" "train $name"; then
    return 0
  fi
  echo "[train] $name  config=$config  device=$DEVICE"
  python -m ssp_dmgtimenet.scripts.train \
    --config "configs/$config" \
    --device "$DEVICE" \
    --num-workers "$TRAIN_WORKERS" \
    2>&1 | tee "$LOG_DIR/${name}.log"
}

train_n_ext_one() {
  local N=$1
  local model=$2
  local suffix=$3
  local config="n_ext_N${N}_${suffix}.yaml"
  local ckpt="$ROOT/artifacts/checkpoints/n_ext_N${N}/$model/best.pt"
  if skip_if_exists "$ckpt" "train n_ext N=$N $model"; then
    return 0
  fi
  echo "[train] N=$N $model  config=$config  device=$DEVICE"
  python -m ssp_dmgtimenet.scripts.train \
    --config "configs/$config" \
    --device "$DEVICE" \
    --num-workers "$TRAIN_WORKERS" \
    2>&1 | tee "$LOG_DIR/n_ext_N${N}_${model}.log"
}

run_main() {
  local item name config
  for item in "${MAIN_MODELS[@]}"; do
    IFS="|" read -r name config <<<"$item"
    train_one "$name" "$config"
  done
}

run_ablation() {
  local item name config
  for item in "${ABLATION_MODELS[@]}"; do
    IFS="|" read -r name config <<<"$item"
    train_one "$name" "$config"
  done
}

run_n_ext() {
  local N item model suffix
  for N in 3 6 7; do
    for item in "${N_EXT_MODELS[@]}"; do
      IFS="|" read -r model suffix <<<"$item"
      train_n_ext_one "$N" "$model" "$suffix"
    done
  done
}

case "$GROUP" in
  all)
    run_main
    run_ablation
    run_n_ext
    ;;
  main) run_main ;;
  ablation) run_ablation ;;
  n_ext) run_n_ext ;;
  *)
    found=0
    for item in "${MAIN_MODELS[@]}" "${ABLATION_MODELS[@]}"; do
      IFS="|" read -r name config <<<"$item"
      if [[ "$name" == "$GROUP" ]]; then
        train_one "$name" "$config"
        found=1
        break
      fi
    done
    if [[ "$found" -eq 0 ]]; then
      echo "Unknown group/model: $GROUP" >&2
      echo "Use: all | main | ablation | n_ext | <model_name>" >&2
      exit 2
    fi
    ;;
esac

echo "Training group '$GROUP' finished."
