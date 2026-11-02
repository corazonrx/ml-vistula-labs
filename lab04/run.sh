#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$LAB_DIR/.venv"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
  elif command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
  "$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10–3.12 (recommended 3.11); set PYTHON=/path/to/python3.11"'
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
if [[ ! -f "$ENV_DIR/requirements.txt" ]] || ! cmp -s "$LAB_DIR/requirements.txt" "$ENV_DIR/requirements.txt"; then
  "$ENV_DIR/bin/python" -m pip install --disable-pip-version-check -r "$LAB_DIR/requirements.txt"
  cp "$LAB_DIR/requirements.txt" "$ENV_DIR/requirements.txt"
fi
exec "$ENV_DIR/bin/python" "$LAB_DIR/run.py" "$@"
