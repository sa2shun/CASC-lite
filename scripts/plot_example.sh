#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

python -m src.casc_lite.cli.export_plots \
  --csv "$ROOT_DIR/results/aggregate.csv"
