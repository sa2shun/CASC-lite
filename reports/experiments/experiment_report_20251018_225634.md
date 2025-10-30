# CASC-lite Experiment Report — 2025-10-18 22:56:34 Adaptive Run

## Context
- **Run ID:** `20251018_225634_adaptive`
- **Entry point:** `python -m src.casc_lite.cli.run_once`
- **Purpose:** Evaluate entropy-adaptive self-consistency on the full GSM8K dataset and compare against fixed-*n* baselines executed the same day.

## Configuration Snapshot
| Parameter | Value |
| --- | --- |
| Model | Meta-Llama/Meta-Llama-3-8B-Instruct |
| Backend | HF Transformers |
| Dataset | `data/gsm8k_full.jsonl` (8,792 items) |
| Mode | Adaptive entropy gating |
| Candidate `n` | {1, 3, 5} |
| Prefix window `K` | 32 tokens |
| Entropy thresholds | `a = 0.34`, `b = 0.54` |
| Sampling params | `temperature = 0.7`, `top_p = 0.9` |

Derived from `results/aggregate.csv` and the default configuration in `src/casc_lite/core/utils.py`.

## Aggregate Metrics (8,792 examples)
| Metric | Value |
| --- | --- |
| Strict / Normalized Accuracy | 18.56% |
| Average latency per question | 1.46 s |
| Average generated tokens per sample | 63.99 |
| Average samples per question | 3.99 |
| Efficiency (accuracy / latency) | 0.127 |
| Mean prefix entropy | 0.651 |
| Total completions generated | 35,040 |
| Approx. total tokens generated | 2.24 M |

## Sample-Count Allocation
| Samples used | Share | Avg. entropy | Accuracy |
| --- | --- | --- | --- |
| 1 | 10.7% (937) | 0.268 | 16.8% |
| 3 | 29.4% (2,586) | 0.445 | 18.3% |
| 5 | 58.9% (5,269) | 0.819 | 19.0% |

### Entropy Bucket View
- **Low (≤0.34):** 937 cases, accuracy 16.8%
- **Mid (0.34–0.54]:** 2,586 cases, accuracy 18.3%
- **High (>0.54):** 5,269 cases, accuracy 19.0%

High-entropy prompts dominate, so the controller escalates to five samples in nearly 60% of evaluations.

## Baseline Comparison (same dataset)
| Run ID | Mode | Accuracy | Avg latency (s) | Avg `n` |
| --- | --- | --- | --- | --- |
| 20251018_075509 | Fixed n=1 | 15.83% | 1.06 | 1.0 |
| 20251018_131047 | Fixed n=3 | 17.97% | 1.63 | 3.0 |
| 20251018_180446 | Fixed n=5 | 20.46% | 1.47 | 5.0 |
| **20251018_225634** | **Adaptive** | **18.56%** | **1.46** | **3.99** |

- Adaptive gating beats the n=3 baseline by +0.59 pts while staying 0.17 s faster.
- Compared to n=5, it saves ~1 sample per question on average, but gives up 1.9 pts of accuracy with almost identical latency. GPU/CPU scheduling or backend caching may mask the expected latency gain.

## Failure & Success Patterns
- The model occasionally echoes the prompt instead of computing the answer (e.g., index 0). This prevents numeric extraction and yields false negatives.
- Correct cases show coherent step-by-step reasoning (e.g., index 12 computes 60+25=85 correctly after voting on five samples).
- Numeric extraction dominates evaluation, so formatting noise rarely rescues a wrong numeric answer.

## Recommendations
1. **Retune thresholds:** Try higher `a`/`b` (e.g., 0.6 / 0.9) or broaden candidate set `{1, 3, 7}` to reduce the heavy reliance on five samples.
2. **Post-process generations:** Apply parsing that favors the final numeric string (e.g., regex on trailing sentence) to mitigate prompt-echo failures.
3. **Profile latency:** Inspect backend batching/logs to confirm why adaptive latency ≈ fixed n=5; ensure entropy and generation calls overlap efficiently.
4. **Targeted error audit:** Cluster high-entropy failures to see whether they stem from ambiguous prompts or arithmetic slips.

## Artifacts
- Per-example CSV: `results/20251018_225634_adaptive_examples.csv`
- Aggregate metrics: `results/aggregate.csv`

