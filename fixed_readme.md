# README Fix Memo

- Added note to the README explaining that the entropy-gated sampler now tops up to the next candidate size when votes do not reach a majority. This memo tracks that documentation tweak for future reference.
- Documented the change of default entropy window to `K=8` and clarified that adaptive thresholds are sourced from baseline quartiles via `scripts/run_full_suite.sh`.
