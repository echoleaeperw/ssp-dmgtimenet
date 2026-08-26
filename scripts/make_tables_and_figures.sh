#!/usr/bin/env bash
# Build paper tables and summary figures from v3 JSON reports.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

python "$ROOT/scripts/build_evaluation_v3_tables.py"
python "$ROOT/scripts/build_extensions_v3_tables.py"
python "$ROOT/scripts/plot_evaluation_v3.py"

echo "Tables:"
echo "  $ROOT/artifacts/evaluation_v3/tables.md"
echo "  $ROOT/artifacts/evaluation_v3/extension_tables.md"
echo "Figures: $ROOT/artifacts/evaluation_v3/figures/"
