#!/usr/bin/env bash
# Train E3 then E4 SEQUENTIALLY (they cannot share a 4 GB GPU) through the one
# shared entry point. Only --experiment-id / --model-name-or-path differ.
#
#   bash scripts/train_transformers.sh            # both
#   bash scripts/train_transformers.sh E4         # just one
#
# Policy: one experiment failing does NOT mark the other as done; each run's
# exit status is logged to artifacts/logs/train_status.log and the script exits
# non-zero if any run failed.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/python"

if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then PY="$ROOT_DIR/.venv/Scripts/python.exe";
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then PY="$ROOT_DIR/.venv/bin/python";
else PY="python"; fi

export HINGLISH_PROJECT_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/python/src"
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"      # base checkpoints are already cached
# Keep temp files / crash dumps off a nearly-full system drive.
mkdir -p "$ROOT_DIR/.tmp" "$ROOT_DIR/artifacts/logs"
export TMP="$ROOT_DIR/.tmp" TEMP="$ROOT_DIR/.tmp"

declare -A CKPT=( [E3]="FacebookAI/xlm-roberta-base" [E4]="l3cube-pune/hing-roberta" )
EXPS=("${@:-E3 E4}"); EXPS=(${EXPS[@]})

"$PY" -m hinglish.prepare_data --verify || { echo "split verification failed"; exit 3; }

FAILED=0
for EXP in "${EXPS[@]}"; do
  LOG="$ROOT_DIR/artifacts/logs/train_${EXP,,}.log"
  echo "$(date -Iseconds) START $EXP (${CKPT[$EXP]})" | tee -a "$ROOT_DIR/artifacts/logs/train_status.log"
  "$PY" -X faulthandler -m hinglish.train --experiment-id "$EXP" --model-name-or-path "${CKPT[$EXP]}" > "$LOG" 2>&1
  RC=$?
  echo "$(date -Iseconds) END $EXP exit=$RC" | tee -a "$ROOT_DIR/artifacts/logs/train_status.log"
  [[ $RC -ne 0 ]] && FAILED=1
done
exit $FAILED
