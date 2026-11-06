#!/usr/bin/env bash
# E3: FacebookAI/xlm-roberta-base. Delegates to the shared transformer trainer.
#   bash scripts/run_e3.sh            (full run)
#   bash scripts/run_e3.sh --smoke-test
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
exec "$PY" -m hinglish.train --experiment-id E3 --model-name-or-path FacebookAI/xlm-roberta-base "$@"
