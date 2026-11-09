#!/usr/bin/env bash
# Start HinglishSense locally without Docker:
#   1. the persistent Spark inference service (E1/E2) on 127.0.0.1:8090
#   2. the API + frontend on http://127.0.0.1:${API_PORT:-8000}
#   bash scripts/start_local.sh          (Ctrl+C stops both)
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Supervised: if the Spark service ever exits (e.g. after laptop sleep), restart it.
: > "$ROOT_DIR/artifacts/logs/spark_serving.log"
(
  while true; do
    echo "[start_local] $(date -Iseconds) starting Spark inference service" >> "$ROOT_DIR/artifacts/logs/spark_serving.log"
    bash "$ROOT_DIR/scripts/run_spark.sh" serve >> "$ROOT_DIR/artifacts/logs/spark_serving.log" 2>&1
    echo "[start_local] $(date -Iseconds) Spark service exited ($?); restarting in 5 s" >> "$ROOT_DIR/artifacts/logs/spark_serving.log"
    sleep 5
  done
) &
SPARK_PID=$!
stop_spark() {
  kill $SPARK_PID 2>/dev/null
  if command -v powershell >/dev/null 2>&1; then   # Windows: the JVM is not a child signal target
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='java.exe'\" | Where-Object { \$_.CommandLine -match 'SparkInferenceServer' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
  else
    pkill -f SparkInferenceServer 2>/dev/null
  fi
}
trap stop_spark EXIT INT TERM
echo "Spark inference service starting (log: artifacts/logs/spark_serving.log)..."
for _ in $(seq 1 180); do
  grep -q "listening on" "$ROOT_DIR/artifacts/logs/spark_serving.log" 2>/dev/null && break
  kill -0 $SPARK_PID 2>/dev/null || { echo "Spark service exited; E1/E2 will be reported unavailable."; break; }
  sleep 1
done
echo "API + frontend on http://127.0.0.1:${API_PORT:-8000}  (docs: /docs)"
"$PY" -m uvicorn app.main:create_app --factory --app-dir "$ROOT_DIR/backend" \
  --host "${API_HOST:-127.0.0.1}" --port "${API_PORT:-8000}"
