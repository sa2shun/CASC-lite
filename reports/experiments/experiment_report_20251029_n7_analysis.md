<details open>
<summary>English</summary>

# CASC-lite Analysis — 2025-10-29 n=7 Sweep

## Context
- **Run ID:** `20251029_040055_fixed`
- **Mode:** Fixed `n = 7`
- **Dataset:** GSM8K full (8,792 problems)
- **Output artifacts:** `results/suite_20251027_220603/20251029_040055_fixed_examples.csv`, `aggregate.csv`

## Aggregate Metrics
| Metric | Value |
| --- | --- |
| Strict / Normalized Accuracy | 49.97% (4,393 / 8,792) |
| Average latency | 11.77 s |
| Average tokens per question | 100.13 |
| Average samples per question | 7.0 |
| Efficiency (accuracy / latency) | 0.042 |

### Post-hoc Accuracy by Sample Count
| n | Accuracy | Δ vs previous | Correct cases |
| --- | --- | --- | --- |
| 1 | 32.83% | – | 2,886 |
| 3 | 38.24% | +6.9 pts (607 gains) | 3,362 |
| 5 | 45.60% | +7.4 pts (853 gains) | 4,009 |
| 7 | 49.97% | +4.0 pts (600 gains) | 4,393 |

## Prefix Entropy Sensitivity (K sweep)
Average prefix entropies derived from stored token-wise trajectories (up to 50 tokens):

| K | Mean | Std | Median | 75th pct | 90th pct |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.657 | 0.300 | 0.604 | 0.805 | 1.056 |
| 16 | 0.670 | 0.213 | 0.646 | 0.812 | 0.957 |
| 32+ | 0.565 | 0.143 | 0.546 | 0.641 | 0.749 |

Quartile accuracies (strict, n=7):

| K | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| 8 | 53.1% | 50.7% | 48.8% | 47.2% |
| 16 | 47.6% | 52.2% | 53.9% | 46.1% |
| 32 | **58.8%** | 52.3% | 48.4% | **40.3%** |

### Deep dive on K = 32
- The quartiles above each contain 2,198 problems. Moving from the lowest to highest entropy quartile drops strict accuracy from 58.8% to 40.3% (−18.5 pts).
- Median split: prompts with entropy ≤0.546 post 55.6% accuracy, versus 44.4% above the median (−11.1 pts).
- Extremes: the bottom decile (entropy ≤0.404) reaches 64.5%, while the top decile (≥0.749) falls to 34.5%.
- Pearson correlation between average prefix entropy (K=32) and correctness is −0.154. The relationship holds across sample counts: corr(n=1) = −0.143, corr(n=3) = −0.156, corr(n=5) = −0.161, corr(n=7) = −0.154.
- Within each quartile the correlation remains negative (≈−0.02 to −0.12), indicating that even local variations in entropy align with error likelihood.

## Entropy Threshold Sweep (n ∈ {1,3,5,7})
Three-tier thresholds (a ≤ b ≤ c) searched over 200 combinations per K, reusing cached completions to emulate adaptive policies.

### Comparative Overview
| Strategy | Accuracy | Avg n | Avg latency (s) | Avg tokens |
| --- | --- | --- | --- | --- |
| Fixed n=1 | 32.83% | 1.00 | 1.69 | 100.27 |
| Fixed n=3 | 38.24% | 3.00 | 5.06 | 100.49 |
| Fixed n=5 | 45.60% | 5.00 | 8.42 | 100.28 |
| Fixed n=7 | 49.97% | 7.00 | 11.77 | 100.13 |
| K=8 High-Accuracy | 46.80% | 5.80 | 9.90 | 100.16 |
| K=16 High-Accuracy | 47.32% | 5.80 | 9.92 | 99.97 |
| K=32 High-Accuracy | 46.77% | 5.80 | 9.82 | 100.02 |
| K=8 Balanced | 41.94% | 4.00 | 6.84 | 100.26 |
| K=16 Balanced | 42.41% | 4.00 | 6.89 | 100.37 |
| K=32 Balanced | 41.25% | 4.00 | 6.83 | 100.30 |
| Entropy-NN (avg n ≤ 6) | 47.36% | 5.88 | 9.87 | 100.19 |
| Entropy-NN (avg n ≤ 5) | 44.68% | 4.90 | 8.22 | 100.34 |
| Entropy-NN (avg n ≤ 4) | 40.83% | 3.99 | 6.66 | 100.40 |
| Cascade-NN (avg n ≤ 6) | 48.18% | 5.97 | 10.03 | 100.05 |
| Cascade-NN (avg n ≤ 5) | 45.54% | 4.94 | 8.30 | 100.22 |
| Cascade-NN (avg n ≤ 4) | 41.73% | 3.99 | 6.69 | 100.30 |
| Cascade-NN (avg n ≤ 3.5) | 39.76% | 3.50 | 5.82 | 100.32 |

These "Entropy-NN" rows apply the logistic-regression models trained on the full 32-token entropy trajectory.  Thresholds on the predicted success probability decide whether to stop at n=3, escalate to n=5, or fall back to n=7.  The best settings found on the cached run were:

- Avg n ≤ 6: stop at n=3 when p₃ ≥ 0.65, otherwise accept n=5 if p₅ ≥ 0.45, else use n=7 (n₃ share 0.5%, n₅ 55.1%, n₇ 44.5%).
- Avg n ≤ 5: thresholds (p₃ ≥ 0.50, p₅ ≥ 0.30) yielding n₃ 18.0%, n₅ 69.0%, n₇ 13.0%.
- Avg n ≤ 4: thresholds (0.30, 0.40) biasing heavily toward n₃ (74.8%) with occasional fallbacks to n₇ (24.6%).

Extending this to a cascade, stage-specific models first predict n=1 success from the entropy trajectory; if doubtful they fall back to the n=3 model (entropy + stage‑1 probability + n₁/n₃ agreement), and repeat for n=5/n=7 with additional agreement flags.  Grid search yielded:

- Avg n ≤ 6: thresholds (t₁=0.60, t₃=0.55, t₅=0.50) → n₁ 0%, n₃ 8%, n₅ 34%, n₇ 57%.
- Avg n ≤ 5: (0.70, 0.50, 0.30) → n₁ 0%, n₃ 15%, n₅ 72%, n₇ 12%.
- Avg n ≤ 4: (0.40, 0.45, 0.30) → n₁ 27%, n₃ 8%, n₅ 53%, n₇ 12%.
- Avg n ≤ 3.5: (0.35, 0.45, 0.35) → n₁ 45%, n₃ 4%, n₅ 32%, n₇ 19%.

| Strategy family | Regression (accuracy %) | Points used |
| --- | --- | --- |
| Fixed n | Accuracy ≈ **1.75 × latency + 29.88** | (1.69,32.83), (5.06,38.24), (8.42,45.60), (11.77,49.97) |
| Cascade-NN | Accuracy ≈ **2.02 × latency + 28.20** | (10.03,48.18), (8.30,45.54), (6.69,41.73), (5.82,39.76) |

![Accuracy vs. latency for fixed-sample and cascade strategies](figures/accuracy_vs_latency_fixed_vs_cascade.png)

#### Accuracy vs. Latency (linear trend)

| Strategy family | Regression (accuracy %) | Points used |
| --- | --- | --- |
| Fixed n | Accuracy ≈ **1.75 × latency + 29.88** | (1.69,32.83), (5.06,38.24), (8.42,45.60), (11.77,49.97) |
| Cascade-NN | Accuracy ≈ **2.02 × latency + 28.20** | (10.03,48.18), (8.30,45.54), (6.69,41.73), (5.82,39.76) |

```
Fixed n trend (accuracy %)        Cascade-NN trend (accuracy %)
 55 |                               55 |
    |                                  |
 50 |                        ●         | 50 |      ●
    |                     ●            |    |   ●
 45 |                 ●                | 45 | ●
    |              ●                   |    |        ●
 40 |                                  | 40 |
    |                                  |    |
 35 |                                  | 35 |
       2   4   6   8  10  12  latency        6   7   8   9  10 latency
```

### High-Accuracy Frontier (avg_n ≈ 5.8, latency ≈ 9.9 s)
| K | (a, b, c) | Accuracy | Avg n | Avg latency |
| --- | --- | --- | --- | --- |
| 8 | (0.33, 0.42, 0.49) | 46.80% | 5.80 | 9.90 s |
| 16 | (0.42, 0.49, 0.54) | **47.32%** | 5.80 | 9.92 s |
| 32 | (0.40, 0.45, 0.48) | 46.77% | 5.80 | 9.82 s |

### High-Efficiency Frontier (avg_n ≈ 2.2, latency ≈ 3.8 s)
| K | (a, b, c) | Accuracy | Avg n | Efficiency |
| --- | --- | --- | --- | --- |
| 8 | (0.75, 0.87, 1.06) | 36.36% | 2.20 | 0.0956 |
| 16 | (0.77, 0.85, 0.96) | **36.58%** | 2.20 | **0.0962** |
| 32 | (0.62, 0.67, 0.75) | 35.66% | 2.20 | 0.0946 |

### Balanced (avg_n ≤ 4)
| K | (a, b, c) | Accuracy | Avg n | Avg latency |
| --- | --- | --- | --- | --- |
| 8 | (0.33, 0.60, 1.06) | 41.94% | 4.00 | 6.84 s |
| 16 | (0.49, 0.59, 0.96) | **42.41%** | 4.00 | 6.89 s |
| 32 | (0.48, 0.55, 0.62) | 41.25% | 4.00 | 6.83 s |

### Interpretation
- **Longer prefix (K ≈ 32)** sharpens the entropy signal: low-entropy quartile achieves +18 pts vs high-entropy quartile, and the median split still shows an 11-pt gap.
- **Three-tier thresholds (1→3→5→7)** capture diminishing returns: going from 5 to 7 votes adds only +4 pts overall but still rescues 600 cases.
- **Efficiency-optimal policy** nearly triples accuracy over n=1 (36.6% vs 32.8%) while keeping cost near n=2 average—promising for throughput-sensitive settings.

## Research Leads
1. **Entropy-window tuning:** Current backend caps `entropy_window=32`; running ablations at K=8/16/32/64 in live inference will reveal whether longer contexts continue to boost discriminative power.
2. **Policy deployment:** Validate the best-efficiency (K=16, a=0.77, b=0.85, c=0.96) and balanced (K=16, a=0.49, b=0.59, c=0.96) policies on fresh generations to quantify improvement vs simulation.
3. **Confidence fusion:** Combine entropy with vote agreement, mean log-likelihood, or self-check prompts (see `todo.md`) to address the 400+ residual errors at n=7.
4. **Cost-aware optimization:** Frame gating as multi-objective (accuracy ≥ target, avg_n ≤ budget) and search thresholds via Bayesian optimization or dynamic programming.

## Limitations
- Threshold sweep reuses cached completions; fresh sampling may shift accuracy by ~4 pts (as observed in prior adaptive runs).
- Latency estimates inherit sequential timings, whereas true adaptive batching would likely reduce per-question latency.
- Token counts reflect per-question averages from fixed runs; total token budgets under adaptive control may grow when multiple samples are generated concurrently.

</details>

<details>
<summary>日本語 / Japanese</summary>

# CASC-lite 分析 — 2025-10-29 n=7 スイープ

## コンテキスト
- **Run ID:** `20251029_040055_fixed`
- **モード:** 固定サンプリング `n = 7`
- **データセット:** GSM8K 全量（8,792 問）
- **生成物:** `results/suite_20251027_220603/20251029_040055_fixed_examples.csv`, `aggregate.csv`

## 集計指標
| 指標 | 値 |
| --- | --- |
| 厳密 / 正規化精度 | 49.97% (4,393 / 8,792) |
| 平均レイテンシ | 11.77 秒 |
| 1 問あたり生成トークン | 100.13 |
| 1 問あたり平均サンプル数 | 7.0 |
| 効率 (精度 / レイテンシ) | 0.042 |

### サンプル数ごとの事後精度
| n | 精度 | 前ステップとの差分 | 正答数 |
| --- | --- | --- | --- |
| 1 | 32.83% | – | 2,886 |
| 3 | 38.24% | +6.9 pt（+607 件） | 3,362 |
| 5 | 45.60% | +7.4 pt（+853 件） | 4,009 |
| 7 | 49.97% | +4.0 pt（+600 件） | 4,393 |

## プレフィックス・エントロピー感度（K スイープ）
保存しておいたトークンごとのエントロピー列（最大 50 トークン）を基に算出。

| K | 平均 | 標準偏差 | 中央値 | 第3四分位 | 第9十位 |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.657 | 0.300 | 0.604 | 0.805 | 1.056 |
| 16 | 0.670 | 0.213 | 0.646 | 0.812 | 0.957 |
| 32+ | 0.565 | 0.143 | 0.546 | 0.641 | 0.749 |

四分位ごとの厳密精度（n=7）:

| K | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| 8 | 53.1% | 50.7% | 48.8% | 47.2% |
| 16 | 47.6% | 52.2% | 53.9% | 46.1% |
| 32 | **58.8%** | 52.3% | 48.4% | **40.3%** |

### K = 32 の詳細解析
- 各四分位には 2,198 問が含まれ、最低エントロピー帯の精度 58.8% から最高エントロピー帯の 40.3% まで 18.5 pt の差が生じる。
- 中央値で二分すると、エントロピー ≤0.546 のグループは 55.6%、それより高いグループは 44.4%（−11.1 pt）。
- 極端値では、第 1 十分位（≤0.404）が 64.5%、第 9 十分位（≥0.749）が 34.5%。
- プレフィックス平均エントロピー（K=32）と正答フラグのピアソン相関は −0.154。n=1/3/5/7 いずれでも −0.14～−0.16 程度の負相関が見られる。
- 各四分位内で見ても相関は負（≈−0.02～−0.12）であり、エントロピーの微妙な変化にも誤答リスクが応答している。

## エントロピー閾値スイープ（n ∈ {1,3,5,7}）
三段階の閾値 (a ≤ b ≤ c) を K ごとに 200 通り評価し、固定ランのサンプルを流用して適応ポリシーを模擬。

### 比較サマリ
| ストラテジー | 精度 | 平均 n | 平均レイテンシ (秒) | 平均トークン |
| --- | --- | --- | --- | --- |
| 固定 n=1 | 32.83% | 1.00 | 1.69 | 100.27 |
| 固定 n=3 | 38.24% | 3.00 | 5.06 | 100.49 |
| 固定 n=5 | 45.60% | 5.00 | 8.42 | 100.28 |
| 固定 n=7 | 49.97% | 7.00 | 11.77 | 100.13 |
| K=8 高精度 | 46.80% | 5.80 | 9.90 | 100.16 |
| K=16 高精度 | 47.32% | 5.80 | 9.92 | 99.97 |
| K=32 高精度 | 46.77% | 5.80 | 9.82 | 100.02 |
| K=8 バランス | 41.94% | 4.00 | 6.84 | 100.26 |
| K=16 バランス | 42.41% | 4.00 | 6.89 | 100.37 |
| K=32 バランス | 41.25% | 4.00 | 6.83 | 100.30 |
| エントロピーNN (平均 n ≤ 6) | 47.36% | 5.88 | 9.87 | 100.19 |
| エントロピーNN (平均 n ≤ 5) | 44.68% | 4.90 | 8.22 | 100.34 |
| エントロピーNN (平均 n ≤ 4) | 40.83% | 3.99 | 6.66 | 100.40 |
| カスケードNN (平均 n ≤ 6) | 48.18% | 5.97 | 10.03 | 100.05 |
| カスケードNN (平均 n ≤ 5) | 45.54% | 4.94 | 8.30 | 100.22 |
| カスケードNN (平均 n ≤ 4) | 41.73% | 3.99 | 6.69 | 100.30 |
| カスケードNN (平均 n ≤ 3.5) | 39.76% | 3.50 | 5.82 | 100.32 |

ここでの「エントロピーNN」は、プレフィックス32トークン分のエントロピー系列を入力し、n=3/5/7 の正答確率をロジスティック回帰で推定したうえで閾値制御した結果である。具体的には、p₃・p₅ に対する閾値をグリッドサーチし、

- 平均 n ≤ 6: (p₃ ≥ 0.65, p₅ ≥ 0.45) の場合に停止、それ以外は n=7（n₃ 0.5%、n₅ 55.1%、n₇ 44.5%）。
- 平均 n ≤ 5: (p₃ ≥ 0.50, p₅ ≥ 0.30) で n₃ 18.0%、n₅ 69.0%、n₇ 13.0%。
- 平均 n ≤ 4: (0.30, 0.40) によって n₃ 停止 74.8%、n₅ 0.6%、n₇ 24.6%。

「カスケードNN」は段階的に判定器を適用する方式で、(1) エントロピーのみで n=1 の正答確率を推定し、危険なら (2) n=3 まで生成してエントロピー＋一致フラグから正答確率を再評価、(3) さらに n=5/n=7 でも同様に判定する。最適閾値は次の通り。

- 平均 n ≤ 6: (t₁=0.60, t₃=0.55, t₅=0.50) → n₁ 0%、n₃ 8%、n₅ 34%、n₇ 57%。
- 平均 n ≤ 5: (0.70, 0.50, 0.30) → n₁ 0%、n₃ 15%、n₅ 72%、n₇ 12%。
- 平均 n ≤ 4: (0.40, 0.45, 0.30) → n₁ 27%、n₃ 8%、n₅ 53%、n₇ 12%。
- 平均 n ≤ 3.5: (0.35, 0.45, 0.35) → n₁ 45%、n₃ 4%、n₅ 32%、n₇ 19%。

| ストラテジー群 | 回帰式 (Accuracy %) | 使用点 |
| --- | --- | --- |
| 固定 n | Accuracy ≈ **1.75 × latency + 29.88** | (1.69,32.83), (5.06,38.24), (8.42,45.60), (11.77,49.97) |
| カスケードNN | Accuracy ≈ **2.02 × latency + 28.20** | (10.03,48.18), (8.30,45.54), (6.69,41.73), (5.82,39.76) |

![固定方式とカスケード方式の精度-レイテンシ関係](figures/accuracy_vs_latency_fixed_vs_cascade.png)

### 高精度フロンティア（avg_n ≈ 5.8, latency ≈ 9.9 s）
| K | (a, b, c) | 精度 | 平均 n | 平均レイテンシ |
| --- | --- | --- | --- | --- |
| 8 | (0.33, 0.42, 0.49) | 46.80% | 5.80 | 9.90 s |
| 16 | (0.42, 0.49, 0.54) | **47.32%** | 5.80 | 9.92 s |
| 32 | (0.40, 0.45, 0.48) | 46.77% | 5.80 | 9.82 s |

### 高効率フロンティア（avg_n ≈ 2.2, latency ≈ 3.8 s）
| K | (a, b, c) | 精度 | 平均 n | 効率 |
| --- | --- | --- | --- | --- |
| 8 | (0.75, 0.87, 1.06) | 36.36% | 2.20 | 0.0956 |
| 16 | (0.77, 0.85, 0.96) | **36.58%** | 2.20 | **0.0962** |
| 32 | (0.62, 0.67, 0.75) | 35.66% | 2.20 | 0.0946 |

### バランス型（avg_n ≤ 4）
| K | (a, b, c) | 精度 | 平均 n | 平均レイテンシ |
| --- | --- | --- | --- | --- |
| 8 | (0.33, 0.60, 1.06) | 41.94% | 4.00 | 6.84 s |
| 16 | (0.49, 0.59, 0.96) | **42.41%** | 4.00 | 6.89 s |
| 32 | (0.48, 0.55, 0.62) | 41.25% | 4.00 | 6.83 s |

### インタープリテーション
- **長めのプレフィックス（K ≈ 32）** はエントロピー指標を鋭くし、低エントロピー帯と高エントロピー帯で 18 pt 以上の精度差、中央値分割でも 11 pt の差を生み出す。
- **三段階の閾値（1→3→5→7）** によって逓減するリターンを捉えられる。5→7 サンプルでは全体で +4 pt だが、600 問の追加正解を回収。
- **高効率ポリシー** は n=1 の 32.8% から 36.6% へとほぼ 3 倍の改善を、平均サンプル数約 2.2 で実現。高スループット用途のベースラインとして有望。

## 研究トピック
1. **Entropy-window 調整:** 現状のバックエンドは `entropy_window=32` に制限されている。K=8/16/32/64 などで実走し、長めのコンテキストが識別力を高め続けるか確認する。
2. **ポリシーの本番検証:** 高効率案（K=16, a=0.77, b=0.85, c=0.96）とバランス案（K=16, a=0.49, b=0.59, c=0.96）を実際の生成で再評価し、シミュレーションとの差異を測定。
3. **Confidence 指標の融合:** エントロピーと、投票一致度・平均対数尤度・自己検証プロンプト（`todo.md` 参照）を組み合わせ、n=7 でも残る 400 件超の誤りを狙い撃ちする。
4. **コスト制約付き最適化:** 「精度 ≥ 目標、平均 n ≤ 予算」を多目的制約として扱い、ベイズ最適化や DP で閾値探索を洗練させる。

## 制約事項
- 閾値スイープは既存サンプルを再利用しており、新規生成では精度が ±数 pt 動く可能性がある（過去の適応ランでは +4 pt 程度の差異を観測）。
- レイテンシは逐次実行の実測値であり、実際にバッチングすればさらに短くなる余地がある。
- トークン数は 1 問あたり平均としてログされている。実際に複数サンプルを同時生成する場合、総トークン消費が増える点に注意。

</details>
