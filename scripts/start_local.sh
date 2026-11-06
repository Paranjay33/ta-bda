#!/usr/bin/env bash
# Start HinglishSense locally without Docker:
#   1. the persistent Spark inference service (E1/E2) on 127.0.0.1:8090
#   2. the API + frontend on http://127.0.0.1:${API_PORT:-8000}
#   bash scripts/start_local.sh          (Ctrl+C stops both)
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

bash "$ROOT_DIR/scripts/run_spark.sh" serve > "$ROOT_DIR/artifacts/logs/spark_serving.log" 2>&1 &
SPARK_PID=$!
trap 'kill $SPARK_PID 2>/dev/null' EXIT INT TERM
echo "Spark inference service starting (log: artifacts/logs/spark_serving.log)..."
for _ in $(seq 1 180); do
  grep -q "listening on" "$ROOT_DIR/artifacts/logs/spark_serving.log" 2>/dev/null && break
  kill -0 $SPARK_PID 2>/dev/null || { echo "Spark service exited; E1/E2 will be reported unavailable."; break; }
  sleep 1
done
echo "API + frontend on http://127.0.0.1:${API_PORT:-8000}  (docs: /docs)"
"$PY" -m uvicorn app.main:create_app --factory --app-dir "$ROOT_DIR/backend" \
  --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
