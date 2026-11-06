#!/usr/bin/env bash
# Shared helpers (sourced). Resolves the project root and the Python interpreter
# (prefers the existing project venv, never creates or deletes one).
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export HINGLISH_PROJECT_ROOT="$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/python/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONIOENCODING=utf-8
if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then PY="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then PY="$ROOT_DIR/.venv/bin/python"
else PY="${PYTHON:-python3}"; fi
mkdir -p "$ROOT_DIR/.tmp" "$ROOT_DIR/artifacts/logs"
export TMP="$ROOT_DIR/.tmp" TEMP="$ROOT_DIR/.tmp"
# Load .env (KEY=VALUE lines) without overriding variables already set.
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" =~ ^# ]] && continue
    v="${v%%#*}"; v="${v%"${v##*[![:space:]]}"}"
    [[ -z "${!k:-}" ]] && export "$k=$v"
  done < "$ROOT_DIR/.env"
fi
