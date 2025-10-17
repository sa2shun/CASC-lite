#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "[install] Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install
else
  echo "[install] pre-commit not found in PATH. Install it via pip to enable git hooks." >&2
fi

echo "[install] Environment ready. Activate via 'source $VENV_DIR/bin/activate'."
