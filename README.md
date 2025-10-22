# CASC-lite: Entropy-Adaptive Self-Consistency

CASC-lite reproduces the study on dynamically adjusting self-consistency sample counts for large language models using output entropy. This repository implements the Hugging Face Transformers backend, evaluation harness, configuration management, plotting utilities, and scripted workflows needed to replicate the experiments described in the accompanying proposal (kept out of version control).

## Key Ideas
- **Entropy-gated sampling:** Estimate uncertainty from the first *K* generated tokens and adapt the number of self-consistency samples (\(n \in \{1,3,5\}\)). The sampler now tops up to the next candidate size when the vote distribution lacks a majority, improving robustness without changing the core algorithm.
- **Backend abstraction:** Default HF Transformers implementation with an interface ready for future vLLM integration.
- **Deterministic runs:** Unified seeding across `random`, NumPy, and PyTorch with cuDNN determinism toggles.
- **Comprehensive logging:** CSV artifacts for per-example results and aggregate summaries, plus optional JSON dumps of raw completions.
- **Reproducible analysis:** Export accuracy–latency scatter plots and Pareto fronts directly from aggregated CSV outputs.

## Repository Layout
```
casc-lite-entropy-adaptive-sc/
├─ src/casc_lite/
│  ├─ backends/           # Backend interface & HF implementation
│  ├─ core/               # Entropy, sampling, evaluation, utilities
│  ├─ cli/                # Command-line entry points
│  └─ config/default.yaml # Default hyper-parameters
├─ tests/                 # Pytest-based unit tests
├─ data/gsm8k_mini.jsonl  # Five-sample GSM8K-compatible stub
├─ scripts/               # Setup and example shell pipelines
├─ results/               # Output directory (gitignored)
├─ Makefile               # Common workflows
└─ .github/workflows/ci.yml
```

> **Note**: `research_proposal.txt` is intentionally excluded via `.gitignore`, as instructed.

## Quick Start
```bash
# 1) Clone and enter the repo
git clone git@github.com:your-org/casc-lite-entropy-adaptive-sc.git
cd casc-lite-entropy-adaptive-sc

# 2) Bootstrap environment (creates .venv and installs deps)
bash scripts/install.sh

# 3) Run linting and tests to verify the setup
make lint
make test
```

To activate the virtual environment manually:
```bash
source .venv/bin/activate
```

## Running Experiments
### Single Configuration
```bash
python -m src.casc_lite.cli.run_once \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data data/gsm8k_mini.jsonl \
  --mode adaptive --K 32 --a 1.3 --b 2.0 \
  --T 0.7 --top_p 0.9 --save_completions
```

### Fixed-*n* Baselines
```bash
python -m src.casc_lite.cli.run_once \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data data/gsm8k_mini.jsonl \
  --mode fixed --n_fixed 3
```

### Threshold Sweeps
```bash
python -m src.casc_lite.cli.sweep_thresholds \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --data /path/to/gsm8k_full.jsonl \
  --a_list 1.0,1.3,1.6 --b_list 1.8,2.0,2.2
```

### Plotting
```bash
python -m src.casc_lite.cli.export_plots --csv results/aggregate.csv
```

All commands above mirror the template provided in `scripts/run_examples.sh`, `scripts/sweep_example.sh`, and `scripts/plot_example.sh`.

## Configuration
Hyper-parameters live in `src/casc_lite/config/default.yaml`. Every CLI flag overrides the YAML entry; alternatively pass `--n_candidates 1,3,5`, `--top_p 0.8`, etc. Key fields:
- `K`, `a`, `b`: entropy window and thresholds
- `n_candidates`: allowed sample counts
- `max_new_tokens`, `min_new_tokens`: generation bounds
- `output_dir`: where CSVs/plots accumulate (default `results/`)
- `retry_on_error`: automatic retries for transient backend failures

## Outputs
Running any experiment appends a row to `results/aggregate.csv` and writes detailed per-example records to `results/<run_id>_examples.csv`. Optional JSON dumps (`--save_completions`) capture raw completions with vote breakdowns. The plotting CLI consumes the aggregate CSV to produce:
- `accuracy_latency_scatter.png`
- `accuracy_latency_pareto.png`

## Evaluation Logic
- Numeric answers: final correctness determined via regex-based number extraction on both prediction and ground truth; strict and normalized scores coincide.
- Text answers: strict equality uses raw strings; normalized accuracy lowercases and collapses whitespace (reported alongside strict accuracy).
- Efficiency metric: normalized accuracy divided by average latency.

## Testing & Quality Gates
- **Unit tests:** `pytest -q`
- **Formatting & linting:** `make lint` (pre-commit hook integrates `black`, `isort`, `flake8`, `mypy`, `ruff`)
- **Continuous integration:** `.github/workflows/ci.yml` installs dependencies and runs lint + tests on every push/PR.

## Makefile Targets
- `make setup` – invoke `scripts/install.sh`
- `make lint` – run pre-commit across the codebase
- `make test` – run the PyTest suite
- `make run-fixed` / `make run-adaptive` – quick baselines on the mini dataset
- `make sweep` – example threshold sweep
- `make plot` – generate scatter + Pareto figures

## Future Work
- Implement `backends/vllm.py` by extending `BaseBackend`
- Mixed precision & batching support for large-scale sweeps
- Add distributed logging (e.g., WandB) for long-running experiments

## License
This project is released under the MIT License. See `LICENSE` for details.
