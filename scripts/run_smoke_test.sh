#!/usr/bin/env bash
# Cheap end-to-end smoke pass on tiny subsets (writes only to *-smoke run dirs).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
"$PY" -m hinglish.prepare_data --verify
bash "$ROOT_DIR/scripts/run_spark.sh" e1 --smoke-test
bash "$ROOT_DIR/scripts/run_spark.sh" e2 --smoke-test
HF_HUB_OFFLINE=1 "$PY" -m hinglish.train --experiment-id E4 --model-name-or-path l3cube-pune/hing-roberta --smoke-test
echo "Smoke tests finished. Smoke artifacts live in artifacts/models/*/*-smoke and can be deleted."
