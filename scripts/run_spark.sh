#!/usr/bin/env bash
# Run a Spark/Scala main class with the project's Java 17 + classpath setup.
#
#   bash scripts/run_spark.sh compile                 # sbt compile (+ export classpath)
#   bash scripts/run_spark.sh test                    # sbt test (ScalaTest)
#   bash scripts/run_spark.sh e1 [--smoke-test|--final-test-eval]
#   bash scripts/run_spark.sh e2 [--smoke-test|--final-test-eval]
#   bash scripts/run_spark.sh scalability
#   bash scripts/run_spark.sh serve                   # persistent inference service (E1/E2)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/spark_env.sh"
cd "$ROOT_DIR/spark"

CMD="${1:-}"; shift || true
case "$CMD" in
  compile) sbt -batch -no-colors compile; rm -f target/runtime-classpath.txt; spark_classpath >/dev/null; echo "classpath exported";;
  test)    sbt -batch -no-colors test;;
  e1)      exec java "${JAVA17_OPENS[@]}" -Xmx4g -cp "$(spark_classpath)" hinglish.experiments.E1LogisticRegression "$@";;
  e2)      exec java "${JAVA17_OPENS[@]}" -Xmx4g -cp "$(spark_classpath)" hinglish.experiments.E2LinearSVCOvR "$@";;
  scalability) exec java "${JAVA17_OPENS[@]}" -Xmx${SCALABILITY_HEAP:-3g} -cp "$(spark_classpath)" hinglish.scalability.ScalabilityBenchmark "$@";;
  serve)   exec java "${JAVA17_OPENS[@]}" -Xmx2g -cp "$(spark_classpath)" hinglish.serving.SparkInferenceServer "$@";;
  *) echo "usage: $0 {compile|test|e1|e2|scalability|serve} [args]"; exit 2;;
esac
