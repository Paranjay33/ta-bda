#!/usr/bin/env bash
# Kept for compatibility: runs every automated suite.
exec bash "$(cd "$(dirname "$0")" && pwd)/test_all.sh" "$@"
