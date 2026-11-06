#!/usr/bin/env bash
# E4: l3cube-pune/hing-roberta. Delegates to the shared transformer trainer.
#   bash scripts/run_e4.sh            (full run)
#   bash scripts/run_e4.sh --smoke-test
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
exec "$PY" -m hinglish.train --experiment-id E4 --model-name-or-path l3cube-pune/hing-roberta "$@"
