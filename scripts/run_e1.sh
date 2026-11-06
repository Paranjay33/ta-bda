#!/usr/bin/env bash
# E1 (Spark MLlib). Kept for compatibility; delegates to run_spark.sh (Java 17 + exported classpath).
#   bash scripts/run_e1.sh [--smoke-test|--final-test-eval]
exec bash "$(cd "$(dirname "$0")" && pwd)/run_spark.sh" e1 "$@"
