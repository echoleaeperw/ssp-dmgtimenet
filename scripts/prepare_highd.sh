#!/usr/bin/env bash
# Build HighD platoon windows for the main experiment (N=5) and N-extension (N=3,6,7).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

HIGHD_ROOT="${HIGHD_ROOT:-$ROOT/datasets/highD}"
if [[ ! -d "$HIGHD_ROOT" ]]; then
  echo "HighD directory not found: $HIGHD_ROOT" >&2
  echo "See datasets/README.md for the required layout." >&2
  exit 2
fi

cd "$PACKAGE_DIR"

echo "[1/2] audit HighD platoons"
python -m ssp_dmgtimenet.scripts.audit_highd_platoons \
  --highd-root "$HIGHD_ROOT" \
  --report-dir "$ROOT/artifacts/reports/highd_audit" \
  --max-N 7 --min-N 3

echo "[2/2] build train/val/test npz windows"
for N in 5 3 6 7; do
  out_dir="$ROOT/artifacts/platoons/highd_N${N}_h5_p3"
  if skip_if_exists "$out_dir/train.npz" "HighD N=$N"; then
    continue
  fi
  echo "  building N=$N -> $out_dir"
  python -m ssp_dmgtimenet.scripts.build_platoon_samples \
    --highd-root "$HIGHD_ROOT" \
    --out-dir "$out_dir" \
    --target-hz 10 \
    --N "$N" \
    --history-sec 5 \
    --predict-sec 3 \
    --stride-sec 1.0 \
    --nonstationary-quantile 0.5 \
    --train-ids $(seq 1 45) \
    --val-ids $(seq 46 50) \
    --test-ids $(seq 51 60)
done

echo "HighD windows written under $ROOT/artifacts/platoons/"
