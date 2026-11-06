#!/usr/bin/env bash
# BDA scalability benchmark (1x/5x/10x REPLICATED workload: synthetic copies for
# runtime measurement only, never extra observations). Local Spark mode -- not a cluster.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
echo "== Spark (Scala, local mode) =="
bash "$ROOT_DIR/scripts/run_spark.sh" scalability
echo "== Single-machine scikit-learn reference (same rows, same scales) =="
"$PY" -m hinglish.scalability_compare
