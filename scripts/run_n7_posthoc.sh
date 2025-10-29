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

DATA_PATH="${DATA_PATH:-data/gsm8k_full.jsonl}"
DEVICE_VALUE="${DEVICE_OVERRIDE:-cuda:0}"
K_VALUE="${K_OVERRIDE:-50}"
POSTHOC_VALUES="${POSTHOC_VALUES:-1,3,5,7}"
SAVE_TOKENS="${SAVE_ENTROPY_TOKENS:-50}"
N_FIXED_VALUE="${N_FIXED_OVERRIDE:-7}"

SUITE_STAMP="$(date +%Y%m%d_%H%M%S)"
SUITE_BASE="${SUITE_NAME:-suite_${SUITE_STAMP}}"
RESULTS_DIR="results/${SUITE_BASE}"
if [[ -d "$RESULTS_DIR" ]]; then
  suffix=1
  while [[ -d "results/${SUITE_BASE}_${suffix}" ]]; do
    ((suffix++))
  done
  RESULTS_DIR="results/${SUITE_BASE}_${suffix}"
fi
mkdir -p "$RESULTS_DIR"
export CASC_SUITE_DIR="$RESULTS_DIR"

echo "==> Output directory: $RESULTS_DIR"

declare -a ARGS=(
  --config src/casc_lite/config/default.yaml
  --data "$DATA_PATH"
  --mode fixed
  --n_fixed "$N_FIXED_VALUE"
  --K "$K_VALUE"
  --T 0.7
  --top_p 0.9
  --max_new_tokens 128
  --output_dir "$RESULTS_DIR"
  --device "$DEVICE_VALUE"
  --posthoc_n_values "$POSTHOC_VALUES"
  --save_entropy_tokens "$SAVE_TOKENS"
)

if [[ "${SAVE_COMPLETIONS:-0}" == "1" ]]; then
  ARGS+=(--save_completions)
fi

echo "==> Running fixed n=${N_FIXED_VALUE} with post-hoc evaluations for {${POSTHOC_VALUES}} and K=${K_VALUE}"
python -m casc_lite.cli.run_once "${ARGS[@]}"

echo "==> Simulation-ready run complete"
