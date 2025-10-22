#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  echo "[ERROR] .venv not found. Run scripts/install.sh first." >&2
  exit 1
fi
source .venv/bin/activate

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

COMMON_ARGS=(
  --config src/casc_lite/config/default.yaml
  --data data/gsm8k_full.jsonl
  --K 32 --a 0.34 --b 0.54
  --T 0.7 --top_p 0.9
  --max_new_tokens 128
  --output_dir results
  --device "${DEVICE_OVERRIDE:-cuda:0}"
)

run_fixed() {
  local n=$1
  echo "==> Running fixed baseline with n=${n}"
  python -m casc_lite.cli.run_once "${COMMON_ARGS[@]}" --mode fixed --n_fixed "$n"
}

echo "==> Running full GSM8K suite"
run_fixed 1
run_fixed 3
run_fixed 5

echo "==> Running adaptive configuration"
python -m casc_lite.cli.run_once "${COMMON_ARGS[@]}" --mode adaptive

echo "==> All experiments complete"
