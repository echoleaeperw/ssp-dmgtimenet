#!/usr/bin/env bash
# Full paper reproduction pipeline. Each stage can also be run independently.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/smoke_test.sh"
"$ROOT/scripts/prepare_highd.sh"
"$ROOT/scripts/prepare_ngsim.sh"
"$ROOT/scripts/train_all.sh" all
"$ROOT/scripts/evaluate_highd.sh"
"$ROOT/scripts/evaluate_extensions.sh"
"$ROOT/scripts/make_tables_and_figures.sh"

echo
echo "Reproduction finished."
echo "Main tables: $ROOT/artifacts/evaluation_v3/tables.md"
echo "Extension tables: $ROOT/artifacts/evaluation_v3/extension_tables.md"
