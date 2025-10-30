# CASC-lite Simulation Report — 2025-10-26 Offline Entropy Sweep

## Context
- **Objective:** Use the fixed-*n* GSM8K results from `results/suite_20251024_094059/` to simulate entropy-gated sampling policies without launching new inference.
- **Source runs:** `20251024_140522_fixed` (n=1), `20251025_024204_fixed` (n=3), `20251025_230704_fixed` (n=5).
- **Approach:** For each example we pick the completion, latency, and correctness from the fixed run whose sample count matches an entropy policy `n(\mathcal{H})` with thresholds `(a, b)`.
- **Goal:** Identify candidate `(a, b)` settings that balance accuracy and cost, and understand why observed adaptive latencies (~5 s when `n=5`) differ from the 8.2 s fixed baseline.

## Reference Metrics
| Run ID | Mode | Accuracy | Avg latency (s) | Avg `n` |
| --- | --- | --- | --- | --- |
| 20251024_140522_fixed | Fixed n=1 | 33.43% | 1.64 | 1.0 |
| 20251025_024204_fixed | Fixed n=3 | 38.36% | 5.00 | 3.0 |
| 20251025_230704_fixed | Fixed n=5 | 46.14% | 8.20 | 5.0 |
| 20251026_073707_adaptive | Adaptive (actual) | 43.22% | 3.32 | 3.49 |

The adaptive run achieves a favorable accuracy–latency trade-off compared with fixed baselines, but hits higher accuracy (43.22%) than any offline policy described below because its completions were freshly sampled and its latency benefited from runtime batching.

## Offline Policy Sweep
Policies follow: use `n=1` if `\mathcal{H} ≤ a`, `n=3` if `a < \mathcal{H} ≤ b`, otherwise `n=5`. Metrics reuse the fixed-run outputs, so no new completions are generated.

| Policy | `(a, b)` | Accuracy | Avg latency (s) | Avg `n` | Efficiency (acc/lat) | `n=1` share | `n=3` share | `n=5` share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| High-accuracy | (0.10, 0.60) | 42.46% | 6.58 | 3.93 | 0.065 | 2.3% | 48.7% | 49.0% |
| Mid band heavy | (0.293, 0.600) | 41.52% | 5.90 | 3.48 | 0.070 | 25.0% | 26.0% | 49.0% |
| Actual thresholds (sim) | (0.293, 0.815) | 38.67% | 5.09 | 3.00 | 0.076 | 25.0% | 50.0% | 25.0% |
| Balanced latency | (0.350, 0.815) | 38.33% | 4.97 | 2.92 | 0.077 | 28.8% | 46.2% | 25.0% |
| Latency-lean | (0.500, 1.200) | 36.52% | 4.05 | 2.38 | 0.090 | 38.6% | 54.1% | 7.3% |
| All-1 baseline | — | 33.43% | 1.64 | 1.00 | 0.203 | 100% | 0% | 0% |
| All-5 baseline | — | 46.14% | 8.20 | 5.00 | 0.056 | 0% | 0% | 100% |

Key takeaways:
- Lowering `b` to 0.60 pushes nearly half the questions to `n=5`, regaining ~4 pts of accuracy versus the `(0.293, 0.815)` setting, but keeps latency above 5.9 s.
- Raising both thresholds (e.g., `a=0.5`, `b=1.2`) yields the lowest simulated latency among adaptive policies (4.05 s) by rarely escalating to five samples (7%), but sacrifices ~6 pts of accuracy relative to the high-accuracy policy.
- The simulated reproduction of the actual thresholds lands at 38.67% accuracy and 5.09 s latency—worse than the real run because the offline reuse of fixed completions cannot exploit fresh sampling or execution parallelism.

## Why Simulation Undershoots Reality
1. **Fresh sampling matters:** The adaptive run drew new generations; several items flipped from incorrect to correct compared with the fixed history, lifting accuracy by ~4.6 pts over the offline reuse.
2. **Runtime batching:** The actual controller issues requests in parallel per question (three or five candidates in the same RPC), whereas fixed baselines measured sequential sampling. Reusing their per-question latencies therefore overestimates adaptive latency by ≈1.8 s.
3. **Token accounting:** Fixed runs log ~100 generated tokens regardless of `n`, so the simulation shows similar token costs even when `n` changes. The live adaptive job recorded 116 tokens on average because it sums across all sampled completions.

## Recommendations
1. **Two candidate policies for follow-up inference:**
   - `(a=0.350, b=0.815)` preserves the mid-entropy success region, trims `n=5` usage to 25%, and should close part of the latency gap while keeping accuracy in the high 30s.
   - `(a=0.500, b=1.200)` stresses latency savings; expect ~2.4 samples/question on average with a simulated accuracy of 36.5%. Worth testing if throughput is critical.
2. **Instrument live runs:** Log per-branch latency so we can replace sequential fixed-run timings with actual parallel execution costs in future simulations.
3. **Augment the policy:** Combine entropy with a secondary signal (vote variance, logit confidence) to better detect the hard high-entropy tail instead of escalating purely on entropy magnitude.

## Limitations
- No real inference was executed; metrics depend entirely on historical completions, so stochastic gains/losses from new sampling are absent.
- Latency figures inherit the fixed baselines' single-request measurements and likely overestimate adaptive wall-clock time by ~30–40%.
- Token usage is under-reported because the fixed CSVs average per-question tokens instead of summing over all samples.
- Threshold grid was coarse (nine `a` values × seven `b` values). Finer sweeps or Bayesian optimization could refine the frontier further if needed.

## Artifacts
- Fixed per-example logs: `results/suite_20251024_094059/20251024_140522_fixed_examples.csv`, `20251025_024204_fixed_examples.csv`, `20251025_230704_fixed_examples.csv`
- Adaptive run (actual reference): `results/suite_20251024_094059/20251026_073707_adaptive_examples.csv`
- Aggregate metrics: `results/suite_20251024_094059/aggregate.csv`
