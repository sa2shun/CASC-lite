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

K_VALUE="${K_OVERRIDE:-8}"
DEVICE_VALUE="${DEVICE_OVERRIDE:-cuda:0}"

COMMON_ARGS=(
  --config src/casc_lite/config/default.yaml
  --data data/gsm8k_full.jsonl
  --K "$K_VALUE"
  --T 0.7 --top_p 0.9
  --max_new_tokens 128
  --output_dir results
  --device "$DEVICE_VALUE"
)

run_fixed() {
  local n=$1
  echo "==> Running fixed baseline with n=${n} (K=${K_VALUE})"
  python -m casc_lite.cli.run_once "${COMMON_ARGS[@]}" --mode fixed --n_fixed "$n"
}

derive_quartiles() {
  python - <<'PY'
import csv
import pathlib
import sys

results_dir = pathlib.Path("results")
agg_path = results_dir / "aggregate.csv"
if not agg_path.exists():
    sys.exit("aggregate.csv not found; run fixed baselines first")

latest = {}
with agg_path.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("mode") != "fixed":
            continue
        n_fixed = row.get("n_fixed", "")
        if not n_fixed:
            continue
        try:
            n = int(float(n_fixed))
        except ValueError:
            continue
        if n not in {1, 3, 5}:
            continue
        run_id = row.get("run_id", "")
        if not run_id:
            continue
        current = latest.get(n)
        if current is None or run_id > current["run_id"]:
            latest[n] = {"run_id": run_id}

missing = {n for n in (1, 3, 5) if n not in latest}
if missing:
    sys.exit(f"missing fixed baselines for n={sorted(missing)}")

entropies: list[float] = []
for info in latest.values():
    run_id = info["run_id"]
    path = results_dir / f"{run_id}_examples.csv"
    if not path.exists():
        sys.exit(f"per-example CSV missing for {run_id}")
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entropies.append(float(row["entropy"]))
            except (KeyError, TypeError, ValueError):
                continue

if len(entropies) < 4:
    sys.exit("insufficient entropy data to compute quartiles")

entropies.sort()

def percentile(p: float) -> float:
    pos = p * (len(entropies) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(entropies) - 1)
    weight = pos - lower
    return entropies[lower] * (1 - weight) + entropies[upper] * weight

q1 = percentile(0.25)
q3 = percentile(0.75)
print(f"{q1} {q3}")
PY
}

echo "==> Running full GSM8K suite"
run_fixed 1
run_fixed 3
run_fixed 5

thresholds=$(derive_quartiles)
if [[ -z "$thresholds" ]]; then
  echo "[ERROR] Failed to derive quartile thresholds" >&2
  exit 1
fi
read -r A_THRESHOLD B_THRESHOLD <<<"$thresholds"
echo "==> Derived adaptive entropy thresholds: a=${A_THRESHOLD}, b=${B_THRESHOLD}"

echo "==> Running adaptive configuration"
python -m casc_lite.cli.run_once "${COMMON_ARGS[@]}" --mode adaptive --a "$A_THRESHOLD" --b "$B_THRESHOLD"

echo "==> All experiments complete"
