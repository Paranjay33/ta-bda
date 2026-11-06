#!/usr/bin/env bash
# LOCKED test evaluation for E1-E4 + comparison report + error analysis.
# Run ONCE, after training, selection and calibration are frozen.
#   bash scripts/evaluate_all_locked.sh
# Every run is appended to artifacts/metrics/locked_test_log.jsonl, so repeated
# test evaluations are visible.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
FAILED=0
"$PY" -m hinglish.prepare_data --verify || exit 3
for e in e1 e2; do
  bash "$ROOT_DIR/scripts/run_spark.sh" "$e" --final-test-eval > "$ROOT_DIR/artifacts/logs/test_$e.log" 2>&1 || FAILED=1
  grep -E "LOCKED TEST" "$ROOT_DIR/artifacts/logs/test_$e.log" || true
done
HF_HUB_OFFLINE=1 "$PY" -m hinglish.evaluate_locked --experiment-id all || FAILED=1
"$PY" -m hinglish.report || FAILED=1
exit $FAILED
