#!/usr/bin/env bash
# Creates a venv ONLY if none exists (never replaces an existing one).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -d "$ROOT_DIR/.venv" ]]; then echo "Using existing $ROOT_DIR/.venv (not modified)."; exit 0; fi
python3 -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/python/requirements.txt" -r "$ROOT_DIR/backend/requirements.txt"
echo "Created $ROOT_DIR/.venv"
