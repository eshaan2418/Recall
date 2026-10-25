# M10 power / sensitivity — SIMULATION-BASED

> **SIMULATED DATA — pipeline validation only. NOT study results; do not report.**
> These are properties of the analysis under ASSUMED variance components, not findings about Recall.

Assumptions: participant SD 0.35, item SD 0.25, residual SD 0.55 (log-seconds); 3 items per condition; B2/none time ratio 0.80; SUS between-participant SD 10, within-participant noise SD 6; 300 replicates per cell; alpha = 0.05 two-sided, unadjusted.

## Time-to-find, Recall vs B2 (power to detect a planted time ratio)

| n | test | ratio 1.00 | ratio 0.90 | ratio 0.80 | ratio 0.75 | ratio 0.70 | ratio 0.65 | ratio 0.60 | ratio 0.50 | MDE (80%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | LMM | 0.05 | 0.09 | 0.36 | 0.51 | 0.67 | 0.89 | 0.95 | 1.00 | 0.65 |
| 10 | paired t | 0.04 | 0.05 | 0.21 | 0.32 | 0.49 | 0.68 | 0.82 | 0.98 | 0.60 |
| 12 | LMM | 0.04 | 0.16 | 0.48 | 0.64 | 0.77 | 0.91 | 0.97 | 1.00 | 0.65 |
| 12 | paired t | 0.03 | 0.10 | 0.32 | 0.44 | 0.58 | 0.79 | 0.89 | 1.00 | 0.60 |
| 15 | LMM | 0.05 | 0.15 | 0.45 | 0.63 | 0.90 | 0.95 | 0.99 | 1.00 | 0.70 |
| 15 | paired t | 0.03 | 0.09 | 0.29 | 0.47 | 0.73 | 0.86 | 0.98 | 1.00 | 0.65 |

Ratio 1.0 column = false-positive rate (should be about 0.05).

## SUS, Recall vs B2 (Wilcoxon)

| n | +0 pts | +4 pts | +8 pts | +12 pts | +16 pts | MDE (80%) |
|---|---|---|---|---|---|---|
| 10 | 0.05 | 0.14 | 0.44 | 0.77 | 0.96 | +16 |
| 12 | 0.05 | 0.17 | 0.61 | 0.89 | 0.99 | +12 |
| 15 | 0.01 | 0.24 | 0.65 | 0.96 | 0.99 | +12 |

Runtime 144.6 s.

**SIMULATED DATA — pipeline validation only. NOT study results; do not report.**
