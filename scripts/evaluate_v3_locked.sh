#!/usr/bin/env bash
# One LOCKED test evaluation of the selected Kaggle E3/E4 run (v3 / v3b), then the
# comparison report (v3 + archived v2), significance test, error analysis and model cards.
# Run once, right after `python -m hinglish.select_transformer_run --apply`.
#   bash scripts/evaluate_v3_locked.sh
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
"$PY" -m hinglish.prepare_data --verify-files || exit 3
TAG=$("$PY" -c "from hinglish.config import load_config; print(load_config().get('transformer','run_tag'))")
if [[ "$TAG" == "v2-clean-split" ]]; then
  echo "transformer.run_tag is still v2-clean-split - run: python -m hinglish.select_transformer_run --apply"; exit 2
fi
for e in E3 E4; do
  if grep -q "\"model_version\": \"${e,,}-$TAG\"" "$ROOT_DIR/artifacts/metrics/locked_test_log.jsonl" 2>/dev/null; then
    echo "$e $TAG already has a locked test evaluation (see artifacts/metrics/locked_test_log.jsonl); not repeating."; continue
  fi
  HF_HUB_OFFLINE=1 "$PY" -m hinglish.evaluate_locked --experiment-id "$e" || exit 1
done
"$PY" -m hinglish.report
"$PY" -m hinglish.model_cards
echo "Done. Restart the app (bash scripts/start_local.sh) to serve $TAG."
