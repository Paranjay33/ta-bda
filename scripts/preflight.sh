#!/usr/bin/env bash
# Non-destructive environment + data preflight. Prints what can run and what is blocked.
#   bash scripts/preflight.sh
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
source "$ROOT_DIR/scripts/spark_env.sh"
ok(){ printf "  [ok]   %s\n" "$*"; }; bad(){ printf "  [MISS] %s\n" "$*"; }

echo "== Toolchain"
"$PY" --version >/dev/null 2>&1 && ok "python: $("$PY" --version 2>&1)" || bad "python"
"$PY" - <<'EOF'
import importlib
for m in ["torch","transformers","fastapi","sqlalchemy","nltk","sklearn","pandas","bitsandbytes"]:
    try:
        mod = importlib.import_module(m); print(f"  [ok]   {m} {getattr(mod,'__version__','')}")
    except Exception as e:
        print(f"  [MISS] {m}: {type(e).__name__}")
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  [ok]   CUDA: {p.name}, {p.total_memory/2**30:.1f} GB")
    else:
        print("  [MISS] CUDA GPU (E3/E4 training needs one; CPU inference works)")
except Exception:
    pass
EOF
command -v java >/dev/null && ok "java: $(java -version 2>&1 | head -1)" || bad "java 17 (set JAVA_HOME)"
command -v sbt >/dev/null && ok "sbt" || bad "sbt"
command -v docker >/dev/null && ok "docker: $(docker --version)" || bad "docker (compose deployment cannot be run on this machine)"

echo "== Data"
for f in data/raw/sentimix_supplied/FinalTrainingOnly.tsv data/raw/sentimix_supplied/ValidationOnly.tsv; do
  [[ -f "$ROOT_DIR/$f" ]] && ok "$f" || bad "$f"
done
"$PY" -m hinglish.prepare_data --verify >/dev/null 2>&1 && ok "frozen split verified (data/processed/split_manifest.json)" \
  || bad "frozen split (run: python -m hinglish.prepare_data)"

echo "== Base checkpoints in the Hugging Face cache"
"$PY" - <<'EOF'
try:
    from huggingface_hub import scan_cache_dir
    have = {r.repo_id for r in scan_cache_dir().repos}
except Exception:
    have = set()
for m in ["FacebookAI/xlm-roberta-base", "l3cube-pune/hing-roberta"]:
    print(("  [ok]   " if m in have else "  [MISS] ") + m)
EOF

echo "== Trained artifacts"
for e in e1 e2; do [[ -f "$ROOT_DIR/artifacts/models/$e/v2-clean-split/manifest.json" ]] && ok "$e PipelineModel" || bad "$e (bash scripts/run_spark.sh $e)"; done
for e in e3 e4; do [[ -f "$ROOT_DIR/artifacts/models/$e/v2-clean-split/manifest.json" ]] && ok "$e checkpoint" || bad "$e (bash scripts/train_transformers.sh ${e^^})"; done
echo "== Disk"
df -h "$ROOT_DIR" 2>/dev/null | tail -1
