# CASC-lite Experiment Report — 2025-10-26 07:37:07 Adaptive Run

## Context
- **Run ID:** `20251026_073707_adaptive`
- **Entry point:** `python -m src.casc_lite.cli.run_once`
- **Purpose:** Re-evaluate entropy-adaptive self-consistency on GSM8K after the fixed-n sweep, checking whether prefix entropy alone can steer sampling.

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

Derived from `results/suite_20251024_094059/aggregate.csv`.

## Aggregate Metrics (8,792 examples)
| Metric | Value |
| --- | --- |
| Strict / Normalized Accuracy | 43.22% (3,800 / 8,792) |
| Average latency per question | 3.32 s |
| Average generated tokens per sample | 116.17 |
| Average samples per question | 3.49 |
| Efficiency (accuracy / latency) | 0.130 |
| Mean prefix entropy | 0.605 |
| Total completions generated | 30,724 |
| Approx. total tokens generated | 1.02 M |

## Sample-Count Allocation
| Samples used | Count | Share | Accuracy |
| --- | --- | --- | --- |
| 1 | 2,198 | 25.0% | 26.6% |
| 3 | 2,222 | 25.3% | 70.4% |
| 5 | 4,372 | 49.7% | 37.7% |

3-sample cases remain the most reliable despite being a quarter of the workload; five-sample escalations help but do not fully close errors.

## Entropy vs. Accuracy (adaptive run)
Shared 20% quantile bins across the full suite:

| Entropy bin | Accuracy | Count |
| --- | --- | --- |
| (0.051, 0.192] | 27.17% | 1,759 |
| (0.192, 0.514] | 43.63% | 1,758 |
| (0.514, 0.668] | **54.32%** | 1,758 |
| (0.668, 0.879] | 47.50% | 1,758 |
| (0.879, 2.857] | 43.49% | 1,759 |

The Pearson correlation between entropy and correctness is +0.099—weak and non-monotonic. Accuracy peaks in the middle bin, while the lowest and highest entropy bands both underperform, mirroring the fixed-n behavior.

## Baseline Comparison (same dataset)
| Run ID | Mode | Accuracy | Avg latency (s) | Avg `n` |
| --- | --- | --- | --- | --- |
| 20251024_140522_fixed | Fixed n=1 | 33.43% | 1.64 | 1.0 |
| 20251025_024204_fixed | Fixed n=3 | 38.36% | 5.00 | 3.0 |
| 20251025_230704_fixed | Fixed n=5 | 46.14% | 8.20 | 5.0 |
| **20251026_073707_adaptive** | **Adaptive** | **43.22%** | **3.32** | **3.49** |

- Adaptive control beats the n=3 baseline by +4.86 pts while cutting latency by 1.68 s.
- Compared with n=5, it surrenders 2.92 pts of accuracy but runs 4.88 s faster and saves ~1.5 samples per question on average.

## Discussion
- Prefix entropy alone fails as a monotonic proxy for correctness. Both low- and high-entropy inputs trigger more errors, so a simple "entropy high ⇒ increase n" rule misallocates budget.
- Despite the noisy signal, adaptive gating still captures a useful trade-off, outperforming n=3 at lower latency by leaning on three-sample decisions whenever entropy lands in the mid band.
- 3-sample branches are extremely accurate (70%), suggesting that the gate could afford to re-try or post-validate the high-entropy cases before escalating straight to five samples.

## Recommendations
1. Combine entropy with vote consistency or completion-level confidence to disambiguate the troublesome low/high entropy tails.
2. Revisit threshold tuning: increasing `a` toward the mid-entropy peak may bias the controller toward the successful 3-sample regime.
3. Add post-processing (numeric parsing, self-verification) for the high-entropy failures instead of always escalating to `n=5`.

## Artifacts
- Per-example CSV: `results/suite_20251024_094059/20251026_073707_adaptive_examples.csv`
- Aggregate metrics: `results/suite_20251024_094059/aggregate.csv`

