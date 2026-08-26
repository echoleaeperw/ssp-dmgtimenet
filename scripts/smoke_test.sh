#!/usr/bin/env bash
# Installation + unit-test smoke check. Does not require HighD/NGSIM.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_package

cd "$PACKAGE_DIR"
python -c "import ssp_dmgtimenet, torch; print('ssp-dmgtimenet', ssp_dmgtimenet.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -m pytest -q tests
echo "Smoke test passed."
