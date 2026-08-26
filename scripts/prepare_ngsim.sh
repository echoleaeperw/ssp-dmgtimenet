#!/usr/bin/env bash
# Build NGSIM zero-shot test sets and the I-80 16:00 smoothing-sensitivity pair.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

NGSIM_ROOT="${NGSIM_ROOT:-$ROOT/datasets/NGSIM/vehicle-trajectory-data}"
if [[ ! -d "$NGSIM_ROOT" ]]; then
  echo "NGSIM vehicle-trajectory-data not found: $NGSIM_ROOT" >&2
  echo "See datasets/README.md for the required layout." >&2
  exit 2
fi

cd "$PACKAGE_DIR"

echo "[1/3] audit NGSIM platoons"
python -m ssp_dmgtimenet.scripts.audit_ngsim_platoons \
  --ngsim-root "$NGSIM_ROOT" \
  --report-dir "$ROOT/artifacts/reports/ngsim_audit"

echo "[2/3] zero-shot test sets (US-101 / I-80, cap 4000 windows/site)"
out_dir="$ROOT/artifacts/platoons/ngsim_N5_h5_p3"
if ! skip_if_exists "$out_dir/us101/test.npz" "NGSIM zero-shot"; then
  python -m ssp_dmgtimenet.scripts.build_ngsim_platoon_samples \
    --ngsim-root "$NGSIM_ROOT" \
    --out-dir "$out_dir" \
    --N 5 --target-hz 10 --history-sec 5 --predict-sec 3 --stride-sec 1.0 \
    --nonstationary-quantile 0.5 \
    --max-windows-per-site 4000
fi

echo "[3/3] I-80 16:00-16:15 original vs reconstructed"
SENS_DIR="$ROOT/artifacts/platoons/ngsim_sensitivity"
ORIG_CSV="$NGSIM_ROOT/0400pm-0415pm/trajectories-0400-0415.csv"
RECON_CSV="$NGSIM_ROOT/0400pm-0415pm/RECONSTRUCTED trajectories-400-0415_NO MOTORCYCLES.csv"

if [[ ! -f "$ORIG_CSV" ]]; then
  echo "Missing original I-80 0400 CSV: $ORIG_CSV" >&2
  exit 2
fi

if ! skip_if_exists "$SENS_DIR/i80_orig_0400/test.npz" "I-80 original 0400"; then
  python -m ssp_dmgtimenet.scripts.build_ngsim_platoon_samples \
    --recording-csv "$ORIG_CSV" \
    --recording-id 801 \
    --site-name i80_orig_0400 \
    --period-name 0400pm-0415pm \
    --out-dir "$SENS_DIR" \
    --N 5 --target-hz 10 --history-sec 5 --predict-sec 3 --stride-sec 1.0 \
    --nonstationary-quantile 0.5 \
    --max-windows-per-site 4000
fi

if [[ -f "$RECON_CSV" ]]; then
  if ! skip_if_exists "$SENS_DIR/i80_recon_0400/test.npz" "I-80 reconstructed 0400"; then
    python -m ssp_dmgtimenet.scripts.build_ngsim_platoon_samples \
      --recording-csv "$RECON_CSV" \
      --recording-id 801 \
      --site-name i80_recon_0400 \
      --period-name 0400pm-0415pm \
      --reconstructed \
      --out-dir "$SENS_DIR" \
      --N 5 --target-hz 10 --history-sec 5 --predict-sec 3 --stride-sec 1.0 \
      --nonstationary-quantile 0.5 \
      --max-windows-per-site 4000
  fi
else
  echo "[warn] reconstructed CSV not found, skip sensitivity reconstructed split: $RECON_CSV"
fi

echo "NGSIM windows written under $ROOT/artifacts/platoons/"
