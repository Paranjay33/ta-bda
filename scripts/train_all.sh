#!/usr/bin/env bash
# Full training pipeline (only when data, Java 17 + sbt, and a GPU are available).
#   bash scripts/train_all.sh
# Order: frozen split -> E1 -> E2 -> E2/E1 validation score handling -> E3 -> E4.
# Policy: continue past a failed experiment, record it, exit non-zero at the end.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
STATUS="$ROOT_DIR/artifacts/logs/train_status.log"
FAILED=0
step(){ local name="$1"; shift; echo "$(date -Iseconds) START $name" | tee -a "$STATUS"
  "$@"; local rc=$?; echo "$(date -Iseconds) END $name exit=$rc" | tee -a "$STATUS"; [[ $rc -ne 0 ]] && FAILED=1; return 0; }

step prepare_data "$PY" -m hinglish.prepare_data
step spark_compile bash "$ROOT_DIR/scripts/run_spark.sh" compile
step E1 bash -c "bash '$ROOT_DIR/scripts/run_spark.sh' e1 > '$ROOT_DIR/artifacts/logs/train_e1.log' 2>&1"
step E2 bash -c "bash '$ROOT_DIR/scripts/run_spark.sh' e2 > '$ROOT_DIR/artifacts/logs/train_e2.log' 2>&1"
step classical_calibration "$PY" -m hinglish.calibrate_classical
step E3_E4 bash "$ROOT_DIR/scripts/train_transformers.sh" E3 E4
exit $FAILED
