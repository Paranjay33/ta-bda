#!/usr/bin/env bash
# E2 (Spark MLlib). Kept for compatibility; delegates to run_spark.sh (Java 17 + exported classpath).
#   bash scripts/run_e2.sh [--smoke-test|--final-test-eval]
exec bash "$(cd "$(dirname "$0")" && pwd)/run_spark.sh" e2 "$@"
