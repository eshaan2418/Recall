# M10 user study — analysis (SIMULATED)

> **SIMULATED DATA — pipeline validation only. NOT study results; do not report.**

Participants analysed: 12 · trials: 108

## Descriptives

| condition | median time-to-find (s) | success rate | trials |
|---|---|---|---|
| b2 | 42.9 | 0.97 | 36 |
| none | 58.9 | 0.94 | 36 |
| recall | 40.4 | 1.00 | 36 |

## Primary: log time-to-find (LMM)

Model: `log_t ~ condition + trial_c`, random effects crossed: (1|participant) + (1|item); n_obs = 108; trial slope = -0.010 log-s/trial. Not-found trials enter at the time cap.

| contrast | time ratio | 95% CI | p (z) | p (Holm, primary) |
|---|---|---|---|---|
| recall_vs_b2 | 0.76 | [0.59, 0.97] | 0.030 | 0.030 |
| recall_vs_none | 0.56 | [0.43, 0.72] | <0.001 | <0.001 |
| b2_vs_none | 0.73 | [0.57, 0.94] | 0.016 | n/a |

Sensitivity (found trials only):

- recall_vs_b2: ratio 0.79 [0.60, 1.03], p = 0.076
- recall_vs_none: ratio 0.59 [0.45, 0.77], p = <0.001
- b2_vs_none: ratio 0.75 [0.57, 0.98], p = 0.038

Robustness (paired t on per-participant mean log time):

- recall_vs_b2: ratio 0.76, p = 0.118 (n = 12)
- recall_vs_none: ratio 0.56, p = <0.001 (n = 12)
- b2_vs_none: ratio 0.73, p = 0.020 (n = 12)

## Secondary (Holm-corrected as one family)

Success: Wilcoxon signed-rank on per-participant success rates (separation/GEE failure fallback).

| test | effect | p | p (Holm) |
|---|---|---|---|
| success recall vs b2 | r_rb +1.00 | 1.000 | 1.000 |
| success recall vs none | r_rb +1.00 | 1.000 | 1.000 |
| tlx_recall_vs_b2 | median diff -4.6, r_rb -0.62 | 0.072 | 0.361 |
| tlx_recall_vs_none | median diff -11.7, r_rb -0.82 | 0.013 | 0.076 |
| tlx_b2_vs_none | median diff -1.7, r_rb -0.45 | 0.198 | 0.595 |
| sus_score_recall_vs_b2 | median diff +5.0, r_rb +0.85 | 0.010 | 0.068 |
| trust_recall_vs_b2 | median diff +0.5, r_rb +1.00 | 0.002 | 0.016 |
| useful_recall_vs_b2 | median diff +0.8, r_rb +0.58 | 0.092 | 0.367 |

NASA-TLX Friedman omnibus: chi2 = 5.64, p = 0.060, n = 12.

| measure | condition | median | mean | n |
|---|---|---|---|---|
| tlx_raw | b2 | 50.0 | 51.4 | 12 |
| tlx_raw | none | 53.8 | 55.9 | 12 |
| tlx_raw | recall | 45.0 | 46.4 | 12 |
| sus_score | b2 | 70.0 | 69.2 | 12 |
| sus_score | recall | 77.5 | 75.6 | 12 |
| trust | b2 | 4.8 | 4.6 | 12 |
| trust | recall | 5.7 | 5.4 | 12 |
| useful | b2 | 5.0 | 4.8 | 12 |
| useful | recall | 5.8 | 5.4 | 12 |

## Reliance (descriptive)

```
{
 "recall": {
  "assistant_accuracy": 0.8333333333333334,
  "n": 36,
  "followed_when_correct": 0.9,
  "followed_when_wrong": 0.3333333333333333
 },
 "b2": {
  "assistant_accuracy": 0.7222222222222222,
  "n": 36,
  "followed_when_correct": 0.7307692307692307,
  "followed_when_wrong": 0.8
 },
 "recall_confidence_auc": 0.9805555555555555
}
```

Notes: z-based p-values from MixedLM/GEE are anti-conservative at n ~ 12; the participant-level paired analysis is the small-sample check. Lower TLX = less workload; higher SUS/trust/usefulness = better.

**SIMULATED DATA — pipeline validation only. NOT study results; do not report.**
