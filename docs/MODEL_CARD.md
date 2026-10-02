# Model card — failure-risk classifier and RUL regressor

## Overview

| | |
|---|---|
| Task | Binary classification: will this machine fail within the next 20 operating cycles? Regression: remaining useful life (cycles, capped at 125). |
| Models | `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor` (scikit-learn); `IsolationForest` as an auxiliary anomaly score |
| Inputs | Per-cycle readings of 8 sensors plus the operating setting (`duty`) and load class, for one machine since commissioning |
| Outputs | Failure probability, alert flag (cost-based threshold), RUL estimate, anomaly z-score, top drifting sensors |
| Version | 0.1.0 |

## Intended use

Decision support for maintenance planners: prioritise which machines to inspect, and roughly when.
It is **not** a safety system and must not be the only safeguard against failures that endanger
people. Alerts should trigger an inspection by a person, not an automatic shutdown.

## Training data

Synthetic run-to-failure telemetry from `predmaint.simulate` (seed 7): 160 machines, 37,215
cycles, three load levels and four failure modes (bearing wear 30%, overheating 30%, seal leak 30%,
electrical fault 10%). The simulator adds operating-setting variation, per-machine baseline
offsets, measurement noise, transient vibration glitches (1% of cycles), sensor dropouts (0.5% of
readings) and two sensors unrelated to failure.

Because the data is simulated, the model's absolute performance says nothing about any specific
real machine. The value of the project is the pipeline and evaluation method.

## Evaluation

- **Split:** by machine (120 train / 40 test, `GroupShuffleSplit`). No machine appears in both.
- **Threshold:** chosen on out-of-fold predictions from 5-fold `GroupKFold` inside the training
  machines, minimising expected cost with a missed failure costing 10× a false alarm.
- **Test results** (from `reports/metrics.json`):

| Metric | Value |
|---|---:|
| ROC-AUC | 0.978 |
| PR-AUC | 0.898 (logistic-regression baseline: 0.887) |
| Precision / recall at threshold | 0.554 / 0.943 |
| Failures flagged ≥10 cycles early | 97.5% of 40 machines |
| Median warning time | 36 cycles |
| False-alarm rate on healthy cycles (>60 cycles before failure) | 0.65% |
| RUL MAE (all / final 30 cycles) | 12.0 / 10.6 cycles |
| Share of final-30-cycle RUL estimates that are optimistic | 58% |
| Anomaly score ROC-AUC | 0.871 |

**By failure mode** (machines flagged ≥10 cycles early): bearing wear 13/13, overheating 10/10,
seal leak 14/14, electrical fault 2/3.

## Limitations and risks

- **Abrupt failures.** Failure modes with a short precursor (electrical faults here) are the most
  likely to be missed. With only three such machines in the test set the miss rate is not well
  estimated.
- **Optimistic RUL near end of life.** 58% of RUL estimates in the final 30 cycles are later than
  the true failure; don't plan maintenance on the RUL estimate alone — use the alert.
- **Distribution shift.** A new machine type, sensor recalibration or a changed operating regime
  will shift feature distributions. Retrain and re-evaluate before relying on alerts.
- **Baseline assumption.** Drift features assume the first 20 cycles after commissioning (or after
  a repair) are healthy. Reset a machine's history after maintenance.
- **Uncalibrated probabilities.** Treat the probability as a ranking score; the cost-based
  threshold, not the raw value, defines an alert.

## How to reproduce

```bash
pip install -e .
predmaint train --report reports/metrics.json
```
