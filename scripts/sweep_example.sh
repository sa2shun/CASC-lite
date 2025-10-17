#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

python -m src.casc_lite.cli.sweep_thresholds \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data "$ROOT_DIR/data/gsm8k_mini.jsonl" \
  --K 32 \
  --T 0.7 \
  --top_p 0.9 \
  --a_list 1.0,1.3,1.6 \
  --b_list 1.8,2.0,2.2
