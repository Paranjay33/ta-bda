#!/usr/bin/env bash
# All automated tests. Exit code is non-zero if any suite fails.
#   bash scripts/test_all.sh            # repo + python + api + scala
#   RUN_UI=1 bash scripts/test_all.sh   # also browser tests (needs a running server, see docs/TESTING.md)
#   RUN_LIVE=1 bash scripts/test_all.sh # also real-model tests against a running stack
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
FAILED=0
run(){ echo; echo "== $1"; shift; "$@" || FAILED=1; }
run "Repository structure" "$PY" "$ROOT_DIR/tests/test_repo_structure.py"
run "Python research package" bash -c "cd '$ROOT_DIR/python' && '$PY' -m pytest -q -p no:cacheprovider tests"
run "API integration" bash -c "cd '$ROOT_DIR/backend' && '$PY' -m pytest -q -p no:cacheprovider tests --ignore=tests/test_ui.py --ignore=tests/test_live_models.py"
run "Scala / Spark (sbt test)" bash "$ROOT_DIR/scripts/run_spark.sh" test
if [[ "${RUN_UI:-0}" == "1" ]]; then
  run "Browser UI (Playwright, Edge/Chromium)" bash -c "cd '$ROOT_DIR/backend' && '$PY' -m pytest -q -p no:cacheprovider tests/test_ui.py"
fi
if [[ "${RUN_LIVE:-0}" == "1" ]]; then
  run "Live models (real artifacts)" bash -c "cd '$ROOT_DIR/backend' && '$PY' -m pytest -q -p no:cacheprovider tests/test_live_models.py"
fi
echo; [[ $FAILED -eq 0 ]] && echo "ALL SUITES PASSED" || echo "SOME SUITES FAILED"
exit $FAILED
