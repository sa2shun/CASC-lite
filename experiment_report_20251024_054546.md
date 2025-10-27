# CASC-lite Experiment Report — 2025-10-24 05:45:46 Adaptive Run

## Context
- **Run ID:** `20251024_054546_adaptive`
- **Entry point:** `python -m src.casc_lite.cli.run_once`
- **Purpose:** Evaluate entropy-adaptive self-consistency with a shortened prefix window (`K = 8`) against the fixed-*n* GSM8K baselines produced on 2025-10-23.

## Configuration Snapshot
| Parameter | Value |
| --- | --- |
| Model | Meta-Llama/Meta-Llama-3-8B-Instruct |
| Backend | HF Transformers |
| Dataset | `data/gsm8k_full.jsonl` (8,792 items) |
| Mode | Adaptive entropy gating |
| Candidate `n` | {1, 3, 5} |
| Prefix window `K` | 8 tokens |
| Entropy thresholds | `a = 0.293`, `b = 0.815` |
| Sampling params | `temperature = 0.7`, `top_p = 0.9` |

Derived from `results/aggregate.csv` and the per-example trace in `results/20251024_054546_adaptive_examples.csv`.

## Aggregate Metrics (8,792 examples)
| Metric | Value |
| --- | --- |
| Strict / Normalized Accuracy | 43.22% |
| Average latency per question | 3.29 s |
| Average generated tokens per sample | 116.17 |
| Average samples per question | 3.49 |
| Efficiency (accuracy / latency) | 0.131 |
| Mean prefix entropy | 0.605 |
| Total completions generated | 30,724 |
| Approx. total tokens generated | 3.57 M |

## Sample-Count Allocation
| Samples used | Share | Avg. entropy | Accuracy |
| --- | --- | --- | --- |
| 1 | 25.0% (2,198) | 0.155 | 26.62% |
| 3 | 25.3% (2,222) | 0.584 | 70.43% |
| 5 | 49.7% (4,372) | 0.841 | 37.74% |

### Entropy Bucket View
- **Low (≤0.293):** 2,198 cases, accuracy 26.62%, avg `n` = 1.00, avg entropy 0.155.
- **Mid (0.293–0.815]:** 4,396 cases, accuracy 51.50%, avg `n` = 3.99, avg entropy 0.582.
- **High (>0.815):** 2,198 cases, accuracy 43.27%, avg `n` = 5.00, avg entropy 1.101.

Sampling splits mirror the bucket sizes almost exactly, indicating the controller relies on entropy thresholds alone without additional heuristics. The mid-entropy tranche is where most gains are realized: moving from 1 to 3 samples boosts accuracy to 70.4% while adding only ~1.7 s of latency versus low-entropy items.

## Baseline Comparison (same dataset, `K = 8`)
| Run ID | Mode | Accuracy | Avg latency (s) | Avg `n` |
| --- | --- | --- | --- | --- |
| 20251023_054120 | Fixed n=1 | 33.43% | 1.64 | 1.0 |
| 20251023_135130 | Fixed n=3 | 38.92% | 3.18 | 3.0 |
| 20251023_211929 | Fixed n=5 | 45.79% | 2.90 | 5.0 |
| **20251024_054546** | **Adaptive** | **43.22%** | **3.29** | **3.49** |

- Compared with the fixed n=3 baseline, the adaptive controller gains +4.30 pts of accuracy at a modest +0.11 s latency cost while using ~0.5 extra samples.
- It still trails the fixed n=5 run by −2.57 pts and is 0.40 s slower, suggesting the controller does not yet beat the always-5 policy under the shortened prefix window.
- Latency roughly doubled versus the earlier (K=32) adaptive run, reflecting larger token counts from shorter context priming and a heavier reliance on five samples.

## Failure & Success Patterns
- **Low-entropy failures:** Several low-entropy prompts (e.g., index 7) still produce arithmetic slips when restricted to a single sample; the model asserts the box weight is “0 pounds,” leading to the wrong final answer despite low uncertainty.
- **High-entropy failures:** Complex multi-step comparisons (index 5) exhaust five samples without consensus, often due to proportion or unit conversion mistakes that the vote cannot override.
- **Mid-entropy successes:** Representative cases like index 0 show the controller escalating to three samples, yielding consistent chain-of-thought and correct numeric extraction.
- **Latency tail:** The slowest errors correspond to verbose deliberations (>3× average tokens) where the model loops through justification phrases without converging on a numeric answer.

## Recommendations
1. **Raise the lower threshold** toward ~0.35 to reduce single-sample exposure on arithmetic tasks that still misfire despite low entropy.
2. **Revisit the upper threshold or add a 7-sample option** so genuinely high-entropy items can gather more diverse votes instead of plateauing at five inconsistent outputs.
3. **Trim verbose reasoning templates** (prompt tweaks or stop sequences) to pull average tokens per sample back toward the 60–80 range and recover latency headroom.
4. **Targeted error audit** on high-entropy failures to identify recurring proportion and unit-conversion patterns that might benefit from specialized post-processing.

## Artifacts
- Per-example CSV: `results/20251024_054546_adaptive_examples.csv`
- Aggregate metrics: `results/aggregate.csv`
