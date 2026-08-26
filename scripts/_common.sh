#!/usr/bin/env bash
# Shared paths and helpers. Source this file from other scripts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT/ssp_dmgtimenet"
DEVICE="${DEVICE:-cuda}"
TRAIN_WORKERS="${TRAIN_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
DELTA_UNSTABLE="${DELTA_UNSTABLE:-0.05}"
EXCITATION_FLOOR="${EXCITATION_FLOOR:-0.05}"

MAIN_MODELS=(
  "ssp_dmgtimenet_v6|ssp_dmgtimenet_v6.yaml"
  "interaction_lstm|baseline_int_lstm.yaml"
  "platoon_transformer|baseline_transformer.yaml"
  "full_graph_attention|baseline_full_graph.yaml"
  "platoon_lstm|baseline_lstm.yaml"
  "cnn_int_lstm_idm|baseline_cnn_int_lstm_idm.yaml"
  "idm_cascade|baseline_idm.yaml"
  "dmg_cascade|baseline_dmg_cascade.yaml"
  "ovm_cascade|baseline_ovm.yaml"
  "fvdm_cascade|baseline_fvdm.yaml"
)

ABLATION_MODELS=(
  "ablation_wo_delay_bias|ablation_wo_delay_bias.yaml"
  "ablation_wo_adj|ablation_wo_adj.yaml"
  "ablation_wo_cfe|ablation_wo_cfe.yaml"
  "ablation_full_graph|ablation_full_graph.yaml"
  "ablation_wo_sub|ablation_wo_sub.yaml"
  "ablation_wo_hgf|ablation_wo_hgf.yaml"
  "ablation_fixed_tau|ablation_fixed_tau.yaml"
  "ablation_wo_fft|ablation_wo_fft.yaml"
)

N_EXT_MODELS=(
  "ssp_dmgtimenet_v6|ssp_v6"
  "interaction_lstm|int_lstm"
  "platoon_transformer|transformer"
)

require_package() {
  if [[ ! -f "$PACKAGE_DIR/pyproject.toml" ]]; then
    echo "Cannot find package at $PACKAGE_DIR" >&2
    exit 2
  fi
}

skip_if_exists() {
  local path=$1
  local label=$2
  if [[ "${SKIP_EXISTING:-0}" == "1" && -f "$path" ]]; then
    echo "[skip] $label already exists: $path"
    return 0
  fi
  return 1
}
