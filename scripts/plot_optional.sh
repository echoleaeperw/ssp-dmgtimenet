#!/usr/bin/env bash
# Optional qualitative / latency extras after the main tables are done.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package
cd "$PACKAGE_DIR"

CKPT_SSP="$ROOT/artifacts/checkpoints/ssp_dmgtimenet_v6/best.pt"
CKPT_INT="$ROOT/artifacts/checkpoints/interaction_lstm/best.pt"
CKPT_TF="$ROOT/artifacts/checkpoints/platoon_transformer/best.pt"

if [[ -f "$CKPT_SSP" && -f "$CKPT_INT" && -f "$CKPT_TF" ]]; then
  echo "[plot] interpretability"
  python -m ssp_dmgtimenet.scripts.plot_interpretability \
    --reference-config configs/ssp_dmgtimenet_v6.yaml \
    --model "SSP-DMGTimeNet:configs/ssp_dmgtimenet_v6.yaml:$CKPT_SSP" \
    --model "Int-LSTM:configs/baseline_int_lstm.yaml:$CKPT_INT" \
    --model "Transformer:configs/baseline_transformer.yaml:$CKPT_TF" \
    --out-dir "$ROOT/artifacts/figures/interpretability"

  echo "[plot] trajectory cases"
  python -m ssp_dmgtimenet.scripts.plot_trajectory_cases \
    --reference-config configs/ssp_dmgtimenet_v6.yaml \
    --model "SSP-DMGTimeNet:configs/ssp_dmgtimenet_v6.yaml:$CKPT_SSP" \
    --model "Int-LSTM:configs/baseline_int_lstm.yaml:$CKPT_INT" \
    --model "Transformer:configs/baseline_transformer.yaml:$CKPT_TF" \
    --ours-name SSP-DMGTimeNet --contrast-name Int-LSTM \
    --out-dir "$ROOT/artifacts/figures/trajectory_cases"
else
  echo "[skip] interpretability / trajectory cases (need SSP, Int-LSTM, Transformer checkpoints)"
fi

need_latency=1
for item in "${MAIN_MODELS[@]}"; do
  IFS="|" read -r name _ <<<"$item"
  if [[ ! -f "$ROOT/artifacts/checkpoints/$name/best.pt" ]]; then
    need_latency=0
    break
  fi
done
if [[ "$need_latency" -eq 1 ]]; then
  echo "[plot] latency"
  python -m ssp_dmgtimenet.scripts.measure_latency \
    --out-json "$ROOT/artifacts/reports/latency/latency.json"
else
  echo "[skip] latency (need all 10 main-experiment checkpoints)"
fi

echo "Optional figures written under $ROOT/artifacts/figures/"
