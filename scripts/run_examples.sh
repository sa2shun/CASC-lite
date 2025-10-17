#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

python -m src.casc_lite.cli.run_once \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data "$ROOT_DIR/data/gsm8k_mini.jsonl" \
  --mode fixed \
  --n_fixed 3 \
  --K 32 \
  --T 0.7 \
  --top_p 0.9

python -m src.casc_lite.cli.run_once \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data "$ROOT_DIR/data/gsm8k_mini.jsonl" \
  --mode adaptive \
  --K 32 \
  --a 1.3 \
  --b 2.0 \
  --T 0.7 \
  --top_p 0.9
