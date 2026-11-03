# Experiment brief: improving a small next-minute option-return forecaster

You are the research assistant in a controlled, budgeted experiment. You propose training and calibration settings for a fixed small neural network. A Python program validates each proposal, trains the network, and measures the results; you never see raw rows or labels, and you do not compute any metric yourself. Use only the information in this brief. Do not browse, run code, or draw on earlier conversations about this project.

## Prediction task

- Data: one-minute bars (open, high, low, close, volume) of Nifty index options. This search uses 700 trading dates from 2016-12-14 to 2019-10-17.
- Target: y = ln(close[t+1] / close[t]), the log return over the next minute, predicted after bar t is complete.
- Persistence predicts y = 0, i.e. an unchanged next close. It is the reference forecast.
- Each example needs six consecutive observed bars of one option within one date. There are 7,069,922 examples; rows per date range from 2,042 to 32,794 (median 8,747).
- Inputs (15):

| feature | definition |
|---|---|
| log_close | ln(close[t]) |
| log_strike | ln(strike) |
| log_volume | ln(1 + reported volume[t]); units unverified |
| log_high_low | ln(high[t]/low[t]); bar range, not bid-ask spread |
| log_close_open | ln(close[t]/open[t]) |
| return_1m | ln(close[t]/close[t-1]) |
| return_5m | ln(close[t]/close[t-5]) |
| return_std_5m | Population standard deviation of five one-minute returns from t-4 through t |
| log_mean_volume_6bars | ln(1 + mean(reported volume[t-5:t])) |
| minute_sin | sin(2*pi*source minute of day/1440) |
| minute_cos | cos(2*pi*source minute of day/1440) |
| is_call | 1 for CE, 0 for PE |
| zero_volume | 1 if reported volume[t] = 0 |
| close_at_most_1 | 1 if close[t] <= 1 |
| flat_ohlc | 1 if all OHLC[t] equal |

The first nine inputs are standardized with fit-window statistics; the last six are cyclic or binary.

## Fixed (cannot be changed)

- Network 15 -> 32 -> 16 -> 1 with ReLU hidden layers and a signed linear output. Output weights and bias start at zero, so training starts from the persistence forecast.
- The training target is y / std(y over the fit window), not centered or clipped. Predictions are converted back to log-return units.
- AdamW, batch size 4,096, at most 12 epochs. After each epoch the tune-window objective is computed: training stops after 3 epochs without an improvement of at least 1e-6, the learning rate halves after each epoch without improvement (minimum 1e-5), and the best epoch is restored. Gradient-norm clipping 1.0. Training seed 42.
- Inputs, data windows and evaluation code.

## Objective (lower is better)

Equal-date MAE: for each date, the mean absolute error of the predicted log return across that date's rows; then the plain average of the daily values, so each date counts equally regardless of its number of rows.

The search objective is the mean equal-date MAE over the score dates of three expanding forward folds:

| fold | fit dates (training) | tune dates | score dates |
|---|---|---|---|
| 1 | 2016-12-14 to 2018-08-21 (420 dates; 4,185,012 rows) | 2018-08-23 to 2018-10-23 (40 dates; 476,162 rows) | 2018-10-24 to 2019-02-18 (80 dates; 826,101 rows) |
| 2 | 2016-12-14 to 2018-12-21 (500 dates; 5,083,295 rows) | 2018-12-24 to 2019-02-18 (40 dates; 403,980 rows) | 2019-02-19 to 2019-06-19 (80 dates; 729,987 rows) |
| 3 | 2016-12-14 to 2019-04-18 (580 dates; 5,889,562 rows) | 2019-04-22 to 2019-06-19 (40 dates; 327,700 rows) | 2019-06-20 to 2019-10-17 (80 dates; 852,660 rows) |

The network trains only on fit dates. Tune dates drive early stopping, learning-rate reduction and any calibration. Score dates only produce the metrics you receive.

## Search space

Every proposal sets all eight values.

| parameter | allowed values | baseline | meaning |
|---|---|---|---|
| loss | "smooth_l1", "l1", "mse" | "smooth_l1" | Training loss on the scaled target y / std(y_fit) |
| smooth_l1_beta | 0.01 to 10 (log scale) | 1 | Smooth L1 transition point in scaled-target units; ignored unless loss is smooth_l1 |
| learning_rate | 0.0001 to 0.01 (log scale) | 0.001 | Initial AdamW learning rate |
| weight_decay | 1e-06 to 0.1 (log scale) | 0.0001 | AdamW decoupled weight decay |
| dropout | 0 to 0.5 | 0.1 | Dropout probability after each hidden ReLU layer |
| sample_weighting | "none", "date_balanced" | "none" | none weights fit rows equally; date_balanced gives every fit date equal total weight |
| low_price_weight | 0.01 to 1 (log scale) | 1 | Multiplies the training weight of rows with current close <= 1; weights are renormalized to mean 1 |
| calibration | "none", "global_scale", "price_group_scale" | "none" | Post-training prediction multiplier alpha in [0, 1] fitted on the tune window to minimize equal-date MAE; price_group_scale fits separate alphas for close <= 1 and close > 1 |

Continuous values are rounded to three significant digits. A configuration identical to the baseline or to an earlier configuration in this search uses up an evaluation without producing new information.

## Diagnostics

### Target distribution in the fold-1 fit dates (2016-12-14 to 2018-08-21; 4,185,012 rows)

- std(y) = 0.0978 log-return units (the target scaling unit); mean = -0.00028; exactly zero in 16.8% of rows.
- |y| quantiles: 50.0%: 0.0065 (0.066 std); 75.0%: 0.0143 (0.146 std); 90.0%: 0.0267 (0.273 std); 95.0%: 0.0397 (0.406 std); 99.0%: 0.1086 (1.111 std); 99.9%: 1.2770 (13.058 std); maximum 7.419 (75.9 std).
- close <= 1: 2.7% of rows; std(y) 0.3210; mean |y| 0.0917; exactly zero in 66.4%.
- close > 1: 97.3% of rows; std(y) 0.0835; mean |y| 0.0140; exactly zero in 15.4%.

### Baseline configuration on the three folds

Objective 0.0144484: +0.023% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.05% | -0.08% | -6.72% | +0.04% | +0.02% | 24/80 | 1 / 4 | - |
| 2 | -0.06% | -0.07% | -5.17% | -0.01% | -0.17% | 23/80 | 1 / 4 | - |
| 3 | +0.15% | +0.19% | -4.23% | +0.05% | +0.06% | 26/80 | 1 / 4 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -1.04%, RMSE -8.80%; close > 1: MAE +0.21%, RMSE +0.64%; calls: MAE -0.01%, RMSE -5.61%; puts: MAE +0.03%, RMSE -5.11%.
Prediction size: mean |prediction| / mean |target| = 0.084; correlation(prediction, target) = 0.327 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.29%, -0.12%, -0.15%, -0.16%]; training sample [0.01642, 0.01635, 0.01631, 0.01631]
- fold 2: tune [+0.17%, +0.18%, +0.30%, +0.29%]; training sample [0.01584, 0.01580, 0.01581, 0.01582]
- fold 3: tune [-0.06%, -0.01%, +0.03%, -0.02%]; training sample [0.01585, 0.01583, 0.01583, 0.01580]

### Earlier findings with the baseline configuration

Previously, the same configuration was trained on all 700 search dates and evaluated on 150 later dates (2019-10-18 to 2020-05-29), which this search does not use. Relative to persistence: equal-date MAE +0.21%, pooled MAE +0.30%, pooled RMSE -5.06%, price MAE +0.63%, price RMSE +1.19%. It beat persistence on 41 of 150 dates. Rows with close <= 1 (2.29% of rows) had MAE +0.07% and RMSE -8.02%, and accounted for 99.25% of the total reduction in squared error; rows with close > 1 had MAE +0.35% and RMSE -0.10%.

## Reply format

Reply with one JSON object and nothing else, for example:

{"proposals": [{"hypothesis": "...", "change": "...", "predicted_change_pct": -0.2, "expected_effect": "...", "rejection_criterion": "...", "config": {"loss": "smooth_l1", "smooth_l1_beta": 1.0, "learning_rate": 0.001, "weight_decay": 0.0001, "dropout": 0.1, "sample_weighting": "none", "low_price_weight": 1.0, "calibration": "none"}}]}

- hypothesis: what you expect and why, citing the diagnostics.
- change: which parameters differ from the configuration you build on, and how.
- predicted_change_pct: your predicted percentage change of the objective relative to the baseline configuration (negative means better).
- expected_effect: what should happen to the objective and the diagnostics.
- rejection_criterion: a measurable result that would reject the hypothesis.
- config: all eight parameters. The example values are the baseline configuration.

## Your task

Propose exactly 12 configurations now. All 12 will be trained and scored, and you will not see any results before proposing. Afterwards, the configuration with the lowest objective is selected. Every configuration must be complete, different from the others, and different from the baseline.
