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

## Results so far in this search

Evaluations used: 11 of 12.

### Trial 1
Hypothesis: The baseline smooth_l1 loss with beta = 1 scaled unit is quadratic for any error below 1 std, and 99% of fit rows have |y| < 1.111 std, so it is effectively an MSE fit of the conditional mean, with only about the largest 1% of moves in the linear region. That matches the diagnostics: pooled RMSE improves by 4-7% and correlation(prediction, target) is 0.327, yet the objective is +0.023% vs persistence, close > 1 rows (97.3% of rows and about 85% of the total absolute error) lose 0.21% MAE, only 23-26 of 80 score dates beat persistence, and tune equal-date MAE is best at epoch 1 in every fold and then drifts worse even though the training-sample MAE is flat to slightly lower, so the training loss and the metric pull apart. MAE is minimized by the conditional median, not the conditional mean. With 16.8% exact zeros overall (66.4% for close <= 1), a mean of -0.00028 and a median |y| of only 0.0065, the conditional median should be 0 in most states and non-zero only where a reversal is more likely than not, for example after large moves in low-price options. Training with L1 should remove the small mean-type predictions that are penalized on zero-return and roughly symmetric rows while keeping the reversal predictions that win, and so lower equal-date MAE.
Change: loss: smooth_l1 -> l1 (smooth_l1_beta is then unused and stays at 1). All other parameters stay at the baseline values, so the loss is the only change.
Predicted change: -0.15%
Configuration: loss=l1, learning_rate=0.001, weight_decay=0.0001, dropout=0.1, sample_weighting=none, low_price_weight=1, calibration=none
Objective 0.0143854: -0.436% vs the baseline configuration, -0.413% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.47% | -0.58% | -3.91% | -0.08% | +0.02% | 61/80 | 5 / 8 | - |
| 2 | -0.49% | -0.58% | -3.37% | -0.11% | -0.04% | 61/80 | 11 / 12 | - |
| 3 | -0.31% | -0.34% | -2.72% | -0.11% | +0.09% | 67/80 | 5 / 8 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -2.89%, RMSE -5.44%; close > 1: MAE -0.06%, RMSE +0.40%; calls: MAE -0.53%, RMSE -3.56%; puts: MAE -0.47%, RMSE -3.08%.
Prediction size: mean |prediction| / mean |target| = 0.048; correlation(prediction, target) = 0.266 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.40%, -0.45%, -0.54%, -0.56%, -0.61%, -0.50%, -0.51%, -0.57%]; training sample [0.01619, 0.01617, 0.01615, 0.01613, 0.01612, 0.01611, 0.01617, 0.01609]
- fold 2: tune [-0.19%, -0.25%, -0.25%, -0.30%, -0.26%, -0.31%, -0.31%, -0.32%, -0.33%, -0.33%, -0.35%, -0.33%]; training sample [0.01561, 0.01561, 0.01562, 0.01560, 0.01564, 0.01561, 0.01563, 0.01562, 0.01561, 0.01563, 0.01561, 0.01562]
- fold 3: tune [-0.44%, -0.51%, -0.52%, -0.51%, -0.55%, -0.52%, -0.51%, -0.53%]; training sample [0.01570, 0.01568, 0.01569, 0.01567, 0.01564, 0.01567, 0.01564, 0.01567]

### Trial 2
Hypothesis: Trial 1 aligned the loss with the metric. L1 fits the conditional median, and its fit loss equals the row-weighted (pooled) MAE of the fit window. The remaining mismatch is the weighting. The objective averages per-date MAE, so every date counts equally, but rows per date range from 2,042 to 32,794 (median 8,747). Under row weighting, the busiest fit dates therefore have up to 16 times the influence of the quietest ones. Trial 1 shows the result. In every fold the pooled MAE gain is larger than the equal-date gain (-0.58% vs -0.47%, -0.58% vs -0.49%, -0.34% vs -0.31%), so relative gains are largest on row-heavy dates, and 13 to 19 of 80 score dates per fold still do not beat persistence. With sample_weighting = date_balanced and L1, the fit loss becomes the equal-date MAE of the fit window (up to the constant target scale). That is the same functional the tune-window early stopping and the search objective use. The fixed network has only 1,057 parameters, so the loss weighting decides where its limited capacity goes. Equal date weights should move the fitted conditional median toward the behaviour of the quieter dates that now lag. They should cost little on the busiest dates, which the objective weights less. The loss of statistical efficiency is minor. The effective sample size is the number of fit dates times the harmonic mean of rows per date, at least 420 x 2,042 = 0.86 million rows in fold 1, which is far more than the parameter count. Mean rows per date are about 10,000 in every fit window (9,964; 10,167; 10,154), so the reweighting adds no time trend.
Change: Builds on trial 1, the best so far. sample_weighting changes from none to date_balanced. All other parameters keep their trial-1 values: loss l1, smooth_l1_beta 1 (unused), learning_rate 0.001, weight_decay 0.0001, dropout 0.1, low_price_weight 1, calibration none.
Predicted change: -0.47%
Configuration: loss=l1, learning_rate=0.001, weight_decay=0.0001, dropout=0.1, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143779: -0.488% vs the baseline configuration, -0.465% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.51% | -0.63% | -3.79% | -0.09% | +0.09% | 58/80 | 5 / 8 | - |
| 2 | -0.57% | -0.67% | -3.82% | -0.16% | -0.11% | 64/80 | 11 / 12 | - |
| 3 | -0.33% | -0.35% | -2.58% | -0.14% | +0.08% | 68/80 | 9 / 12 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.10%, RMSE -5.55%; close > 1: MAE -0.09%, RMSE +0.40%; calls: MAE -0.60%, RMSE -3.75%; puts: MAE -0.50%, RMSE -3.04%.
Prediction size: mean |prediction| / mean |target| = 0.056; correlation(prediction, target) = 0.274 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.41%, -0.58%, -0.51%, -0.62%, -0.64%, -0.53%, -0.50%, -0.58%]; training sample [0.01620, 0.01618, 0.01615, 0.01613, 0.01612, 0.01612, 0.01616, 0.01609]
- fold 2: tune [-0.22%, -0.29%, -0.31%, -0.34%, -0.33%, -0.35%, -0.36%, -0.35%, -0.38%, -0.38%, -0.40%, -0.41%]; training sample [0.01563, 0.01561, 0.01560, 0.01559, 0.01559, 0.01561, 0.01558, 0.01560, 0.01557, 0.01558, 0.01556, 0.01556]
- fold 3: tune [-0.48%, -0.56%, -0.56%, -0.58%, -0.59%, -0.59%, -0.58%, -0.60%, -0.64%, -0.64%, -0.59%, -0.62%]; training sample [0.01570, 0.01568, 0.01570, 0.01567, 0.01564, 0.01565, 0.01563, 0.01564, 0.01565, 0.01563, 0.01563, 0.01562]

### Trial 3
Hypothesis: Trial 2 matched the loss and the weighting to the objective. The next limit is likely dropout 0.1, a setting inherited from the MSE-like baseline that L1 makes more costly. (1) The network is under-fitted rather than over-fitted. It has 1,057 parameters for 4.2 to 5.9 million fit rows. In trial 2 the training-sample MAE falls only 0.4-0.7% over a run (fold 1 0.01620 -> 0.01609, fold 2 0.01563 -> 0.01556, fold 3 0.01570 -> 0.01562), and tune MAE in folds 2 and 3 still improves at epochs 9 to 12 (fold 2 selects epoch 11 of the 12-epoch cap). Fit and tune improve together; only fold 1 worsens on tune after epoch 5, and restoring the best epoch already handles that. Weight decay is negligible (lr x wd = 1e-7 per step, under 0.2% total shrinkage in at most about 17,000 steps), so dropout is the only active regularizer. (2) Under L1, dropout noise is penalized at first order. The target is exactly zero in 16.8% of fit rows (66.4% for close <= 1) and median |y| is only 0.066 std, so probability mass sits at the conditional median. Adding dropout noise e to a prediction there raises the expected loss by about P(y = median) x E|e|. Under the baseline smooth L1 with beta 1, which is quadratic for about 99% of rows, the same noise cost only about E[e^2]/2, a second-order amount. Because the noise scales with the hidden-unit contributions, the L1 penalty shrinks predictions toward zero beyond what the median requires and rules out large contributions that partly cancel. It should bite hardest where the gains are. Close <= 1 rows improve MAE by 3.10% against 0.09% for close > 1. They carry about 15% of fit-window absolute error (2.7% x 0.0917 against 97.3% x 0.0140), so they supply most of the mean pooled MAE gain (about 0.154 x 3.10% = 0.48% of the 0.55%). They need large reversal predictions in states where a reversal is only slightly more likely than not, which is where a first-order noise penalty can move the L1 optimum to zero. (3) With ReLU, the deterministic inference pass is not the average of the dropout passes (E[ReLU(z)] >= ReLU(E[z])), so inference outputs carry a state-dependent offset that the zero-heavy L1 metric charges linearly. Setting dropout to 0 removes both effects, so the fixed network should fit the conditional median more closely, mainly for close <= 1 reversals.
Change: Builds on trial 2, the best so far. dropout changes from 0.1 to 0. All other parameters keep their trial-2 values: loss l1, smooth_l1_beta 1 (unused), learning_rate 0.001, weight_decay 0.0001, sample_weighting date_balanced, low_price_weight 1, calibration none.
Predicted change: -0.56%
Configuration: loss=l1, learning_rate=0.001, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143748: -0.509% vs the baseline configuration, -0.486% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.58% | -0.71% | -2.92% | -0.12% | +0.13% | 57/80 | 4 / 7 | - |
| 2 | -0.55% | -0.64% | -3.41% | -0.11% | +0.09% | 60/80 | 8 / 11 | - |
| 3 | -0.35% | -0.36% | -2.27% | -0.13% | +0.19% | 69/80 | 5 / 8 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.25%, RMSE -4.73%; close > 1: MAE -0.08%, RMSE +0.40%; calls: MAE -0.65%, RMSE -3.23%; puts: MAE -0.50%, RMSE -2.50%.
Prediction size: mean |prediction| / mean |target| = 0.069; correlation(prediction, target) = 0.272 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.46%, -0.67%, -0.63%, -0.70%, -0.69%, -0.69%, -0.65%]; training sample [0.01616, 0.01613, 0.01611, 0.01613, 0.01608, 0.01610, 0.01611]
- fold 2: tune [-0.12%, -0.29%, -0.33%, -0.31%, -0.35%, -0.34%, -0.36%, -0.40%, -0.38%, -0.36%, -0.40%]; training sample [0.01565, 0.01558, 0.01557, 0.01555, 0.01556, 0.01557, 0.01555, 0.01554, 0.01554, 0.01557, 0.01553]
- fold 3: tune [-0.44%, -0.59%, -0.59%, -0.57%, -0.60%, -0.57%, -0.59%, -0.59%]; training sample [0.01570, 0.01565, 0.01566, 0.01563, 0.01562, 0.01563, 0.01562, 0.01562]

### Trial 4
Hypothesis: Most of the current gain comes from close <= 1 rows, while the close > 1 predictions look over-sized. Close > 1 rows are 97.3% of rows and about 85% of absolute error (0.973 x 0.0140 against 0.027 x 0.0917). In trials 2 and 3 their MAE was only 0.09% and 0.08% below persistence while their RMSE stayed 0.40% above it, so these predictions carry little signal relative to their size. Removing dropout in trial 3 raised mean |prediction| / mean |target| from 0.056 to 0.069 (+23%). Close <= 1 MAE improved from -3.10% to -3.25%, but close > 1 MAE did not improve, price RMSE (which weights higher-priced options most) worsened in every fold (+0.09/-0.11/+0.08% to +0.13/+0.09/+0.19%), and dates better than persistence fell from 190 to 186 of 240. The metric charges excess size at first order. Close > 1 targets are exactly zero in 15.4% of rows and the overall median |y| is only 0.066 std, so in most states the conditional median is 0, and the slope of E|y - p| at p = 0+ is P(y <= 0) - P(y > 0), which is positive whenever the zero atom outweighs the upward skew (symmetrically for p < 0). The fitted medians are also noisy: after the learning rate has halved, training-sample MAE still moves by 0.00001-0.00005 and tune MAE by up to 0.04 points between epochs. An L1 fit whose in-sample optimal scale is 1 is then over-sized on new dates, the usual case for shrinkage. price_group_scale fits alpha in [0, 1] on the tune window separately for close <= 1 and close > 1 rows. It can therefore shrink the weak close > 1 predictions while keeping the close <= 1 reversal predictions near full size; those supply about 0.50 of the 0.57-point mean pooled MAE gain (0.154 x 3.25%). A single global alpha would be a compromise held near 1 by the large close <= 1 predictions. Each tune window immediately precedes its score window and has 328,000 to 476,000 rows over 40 dates, enough to pin down two alphas, and the [0, 1] bound limits the worst case to moving part of the forecast back to persistence.
Change: Builds on trial 3, the best so far. calibration changes from none to price_group_scale. All other parameters keep their trial-3 values: loss l1, smooth_l1_beta 1 (unused), learning_rate 0.001, weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1. Because the tune curve is computed after calibration, epoch selection, learning-rate halving and stopping may differ slightly from trial 3.
Predicted change: -0.53%
Configuration: loss=l1, learning_rate=0.001, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=price_group_scale
Objective 0.0143769: -0.495% vs the baseline configuration, -0.472% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.60% | -0.74% | -3.22% | -0.12% | +0.04% | 64/80 | 4 / 7 | close_at_most_1 0.96, close_above_1 0.66 |
| 2 | -0.53% | -0.62% | -3.57% | -0.14% | -0.00% | 59/80 | 11 / 12 | close_at_most_1 0.94, close_above_1 0.91 |
| 3 | -0.31% | -0.32% | -2.09% | -0.12% | +0.16% | 67/80 | 2 / 5 | close_at_most_1 0.96, close_above_1 0.79 |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.23%, RMSE -4.83%; close > 1: MAE -0.08%, RMSE +0.35%; calls: MAE -0.63%, RMSE -3.31%; puts: MAE -0.50%, RMSE -2.60%.
Prediction size: mean |prediction| / mean |target| = 0.062; correlation(prediction, target) = 0.274 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.53%, -0.68%, -0.64%, -0.74%, -0.73%, -0.71%, -0.67%]; training sample [0.01616, 0.01613, 0.01611, 0.01613, 0.01608, 0.01610, 0.01611]
- fold 2: tune [-0.26%, -0.29%, -0.33%, -0.34%, -0.35%, -0.38%, -0.39%, -0.40%, -0.38%, -0.39%, -0.41%, -0.41%]; training sample [0.01565, 0.01558, 0.01557, 0.01555, 0.01556, 0.01557, 0.01555, 0.01554, 0.01554, 0.01557, 0.01553, 0.01551]
- fold 3: tune [-0.54%, -0.60%, -0.60%, -0.59%, -0.60%]; training sample [0.01570, 0.01565, 0.01566, 0.01563, 0.01562]

### Trial 5
Hypothesis: Trials 1-3 aligned the loss and the weighting with the objective and removed dropout noise; the next limit is likely optimizer noise set by the initial learning rate. Under L1 the network fits the conditional median, and for most rows that median sits on the zero atom: exact zeros are 16.8% of fit rows (15.4% for close > 1, 66.4% for close <= 1) and median |y| is only 0.066 std, so equal-date MAE charges any wobble of those predictions around zero at first order. AdamW moves each parameter by up to about the learning rate per step, largely independent of the gradient's size, and per-row L1 gradients keep the same size near the optimum, so at a fixed rate the fitted medians keep moving by an amount that grows with the rate. Trial 3's learning curves fit this. In every fold the full rate of 0.001 stopped improving the tune objective within three or four epochs (fold 1 epoch 3, fold 2 epoch 4, fold 3 by epoch 4), and every best epoch came only after the halving rule had cut the rate: fold 1 epoch 4 at 0.0005 (-0.70% against -0.67% at the full rate), fold 2 epoch 8 at 0.00025 (-0.40% against -0.33%), fold 3 epoch 5 at a halved rate (-0.60% against -0.59%). Trials 3 and 4 followed the same training path (identical training-sample MAE sequences), and in fold 3 the epoch-5 model selected in trial 3 scored -0.35% on the score dates against -0.31% for the full-rate epoch-2 model selected in trial 4 at the same tune level, although calibration also differed. Removing the other noise source, dropout, also helped (-0.488% to -0.509%) and brought the best epochs forward (5, 11, 9 to 4, 8, 5). Starting at 0.0005 should keep the whole run in the regime where the best epochs were found, while halvings can still take the rate to 0.00025 and below. Convergence should still fit in the 12-epoch cap: most of trial 3's tune gain arrived by epoch 2 in folds 1 and 3 (-0.67% and -0.59%), and fold 2 stopped on patience at epoch 11, not at the cap. 0.0005 is also about the lowest safe start: if fold 2's late gains were progress rather than noise reduction, matching its trial-3 best epoch (a summed learning rate of about 6.8 over 1,241 steps per epoch) would take about 11 epochs at a constant 0.0005 and about 18 at 0.0003.
Change: Builds on trial 3, the best so far. learning_rate changes from 0.001 to 0.0005. All other parameters keep their trial-3 values: loss l1, smooth_l1_beta 1 (unused), weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none. The decoupled weight-decay shrinkage per step (learning rate x weight decay) halves to 5e-8 and stays negligible.
Predicted change: -0.53%
Configuration: loss=l1, learning_rate=0.0005, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143775: -0.491% vs the baseline configuration, -0.468% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.57% | -0.68% | -3.61% | -0.11% | +0.15% | 56/80 | 5 / 8 | - |
| 2 | -0.53% | -0.61% | -2.82% | -0.16% | +0.00% | 58/80 | 11 / 12 | - |
| 3 | -0.33% | -0.34% | -1.96% | -0.12% | +0.21% | 67/80 | 5 / 8 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.21%, RMSE -4.74%; close > 1: MAE -0.06%, RMSE +0.67%; calls: MAE -0.61%, RMSE -3.14%; puts: MAE -0.48%, RMSE -2.44%.
Prediction size: mean |prediction| / mean |target| = 0.068; correlation(prediction, target) = 0.270 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.19%, -0.53%, -0.61%, -0.66%, -0.68%, -0.66%, -0.66%, -0.67%]; training sample [0.01645, 0.01615, 0.01613, 0.01613, 0.01611, 0.01612, 0.01613, 0.01610]
- fold 2: tune [-0.03%, -0.25%, -0.29%, -0.32%, -0.35%, -0.36%, -0.37%, -0.38%, -0.37%, -0.37%, -0.40%, -0.40%]; training sample [0.01597, 0.01563, 0.01558, 0.01556, 0.01558, 0.01558, 0.01559, 0.01559, 0.01559, 0.01560, 0.01559, 0.01558]
- fold 3: tune [-0.22%, -0.52%, -0.59%, -0.60%, -0.61%, -0.61%, -0.62%, -0.62%]; training sample [0.01579, 0.01568, 0.01566, 0.01565, 0.01564, 0.01565, 0.01564, 0.01564]

### Trial 6
Hypothesis: Trial 5 rejected the step-noise explanation, and its results point the other way. Halving the initial rate to 0.0005 left the best tune levels unchanged (fold 1 -0.68% vs -0.70% in trial 3, fold 2 -0.40% vs -0.40%, fold 3 -0.61% vs -0.60%). It delayed progress by about one epoch: epoch-1 tune was -0.19/-0.03/-0.22% vs -0.46/-0.12/-0.44%, and epoch-1 training-sample MAE was 0.01645/0.01597/0.01579 vs 0.01616/0.01565/0.01570. Score results were slightly worse in all three folds (-0.57/-0.53/-0.33% vs -0.58/-0.55/-0.35%), close > 1 RMSE rose from +0.40% to +0.67%, and close > 1 MAE weakened from -0.08% to -0.06%. The initial rate therefore seems to matter through how much progress a run makes before the halvings anneal it, not through per-step noise. Fold 2 shows this most clearly. It ran to or near the 12-epoch cap in every trial (best/run epochs 11/12, 11/12, 8/11, 11/12, 11/12), and its best tune epoch was always 8 or later. In trials 2-4 its training-sample MAE reached its lowest value in the final epochs (0.01565 -> 0.01553 in trial 3; 0.01551 at epoch 12 in trial 4). The plateau-halving rule already supplies the annealing: every trial-3 best epoch came after at least one halving. Starting at 0.002 adds one annealing stage within the same epoch budget (0.002 -> 0.001 -> 0.0005 -> 0.00025). The early epochs should then move faster toward the conditional-median fit, including the large close <= 1 reversal outputs. Close <= 1 targets have mean |y| 0.94 and std 3.3 in scaled units, and these rows supply about 0.50 of trial 3's 0.57-point mean pooled MAE gain (0.154 x 3.25%). Later halvings still bring the rate into the 0.0005-0.00025 range where the best epochs were found. Batch size 4,096, gradient clipping at 1.0 and the halving rule should keep a 0.002 start stable. Decoupled weight-decay shrinkage per step becomes 2e-7, still negligible.
Change: Builds on trial 3, the best so far. learning_rate changes from 0.001 to 0.002. All other parameters keep their trial-3 values: loss l1, smooth_l1_beta 1 (unused), weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none.
Predicted change: -0.52%
Configuration: loss=l1, learning_rate=0.002, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143675: -0.560% vs the baseline configuration, -0.537% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.59% | -0.70% | -3.59% | -0.13% | +0.13% | 57/80 | 4 / 7 | - |
| 2 | -0.60% | -0.69% | -3.38% | -0.16% | +0.05% | 58/80 | 8 / 11 | - |
| 3 | -0.45% | -0.48% | -3.01% | -0.14% | +0.26% | 64/80 | 10 / 12 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.44%, RMSE -5.38%; close > 1: MAE -0.11%, RMSE +0.28%; calls: MAE -0.68%, RMSE -3.66%; puts: MAE -0.57%, RMSE -3.00%.
Prediction size: mean |prediction| / mean |target| = 0.072; correlation(prediction, target) = 0.281 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.58%, -0.56%, -0.41%, -0.64%, -0.64%, -0.62%, -0.62%]; training sample [0.01615, 0.01611, 0.01611, 0.01611, 0.01608, 0.01609, 0.01610]
- fold 2: tune [-0.26%, -0.29%, -0.33%, -0.32%, -0.34%, -0.40%, -0.38%, -0.41%, -0.39%, -0.36%, -0.41%]; training sample [0.01564, 0.01560, 0.01559, 0.01554, 0.01554, 0.01554, 0.01553, 0.01552, 0.01551, 0.01554, 0.01550]
- fold 3: tune [-0.56%, -0.46%, -0.64%, -0.60%, -0.67%, -0.64%, -0.72%, -0.67%, -0.74%, -0.79%, -0.69%, -0.76%]; training sample [0.01566, 0.01567, 0.01565, 0.01565, 0.01565, 0.01563, 0.01561, 0.01564, 0.01561, 0.01559, 0.01561, 0.01561]

### Trial 7
Hypothesis: Trials 5, 3 and 6 form a one-factor learning-rate series with everything else fixed (l1, date_balanced, dropout 0, no calibration): 0.0005 -> -0.491%, 0.001 -> -0.509%, 0.002 -> -0.560% vs the baseline configuration, and the second doubling gained more than the first (0.051 vs 0.018 points). The step from 0.001 to 0.002 improved equal-date MAE in all three folds (-0.58/-0.55/-0.35% to -0.59/-0.60/-0.45%), so the optimum has probably not been reached. Where the gain arose, the runs are still limited by progress rather than by step noise. In fold 2 the tune objective improved in each of the first three full-rate epochs (-0.26, -0.29, -0.33%), and training-sample MAE reached the lowest value of the search (0.01550 vs 0.01553 in trial 3). Fold 3 reached 0.01559 (vs 0.01562), its tune objective set a new best after each of the first four halvings (-0.56% at 0.002, -0.64% at 0.001, -0.67% at 0.0005, -0.72% at 0.00025, -0.79% at 0.000125), and the run reached the 12-epoch cap. Fold 1 looks saturated: its training-sample minimum was 0.01608 in both trials 3 and 6 and its score barely moved across trials 5, 3 and 6 (-0.57/-0.58/-0.59%). The higher rate helped where the signal is: close <= 1 MAE -3.44% vs -3.25%, close <= 1 RMSE -5.38% vs -4.73%, fold 3 pooled RMSE -3.01% vs -2.27%, correlation 0.281 vs 0.272. These rows are only 2.7% of fit rows. Under L1 every row contributes a gradient of fixed size (its weight), so their reversal directions are a small signal in each batch against near-random signs from the many rows at the zero atom (16.8% exact zeros, median |y| 0.066 std). They are learned slowly, and with 1,057 parameters for 4.2 to 5.9 million rows the 12-epoch budget limits how far they get. AdamW moves each parameter by up to about the learning rate per step, so a 0.004 start doubles the early travel. The plateau-halving rule still anneals the added noise (0.004 -> 0.002 -> 0.001 -> ...) and the best epoch is restored; in trial 6 every best epoch already came after at least one halving (folds 1 and 2 at 0.0005, fold 3 at 0.000125). Two effects limit the expected gain. Fold 1 at 0.002 already had two consecutive non-improving epochs after epoch 1 (-0.56%, -0.41%), so at 0.004 a third could stop it with a high-rate epoch-1 model. Fold 3 used all 12 epochs, so an extra annealing stage may cut its final low-rate epochs. Weight-decay shrinkage per step becomes 4e-7, still negligible, and bounded L1 gradients with clipping at 1.0 should keep the higher rate stable.
Change: Builds on trial 6, the best so far. learning_rate changes from 0.002 to 0.004, one more doubling in the series 0.0005, 0.001, 0.002. All other parameters keep their trial-6 values: loss l1, smooth_l1_beta 1 (unused), weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none.
Predicted change: -0.57%
Configuration: loss=l1, learning_rate=0.004, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143661: -0.570% vs the baseline configuration, -0.547% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.53% | -0.68% | -4.01% | -0.02% | +0.07% | 51/80 | 10 / 12 | - |
| 2 | -0.70% | -0.82% | -3.91% | -0.14% | +0.07% | 60/80 | 12 / 12 | - |
| 3 | -0.43% | -0.46% | -3.09% | -0.13% | +0.22% | 68/80 | 7 / 10 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.78%, RMSE -5.84%; close > 1: MAE -0.09%, RMSE +0.14%; calls: MAE -0.73%, RMSE -4.08%; puts: MAE -0.58%, RMSE -3.27%.
Prediction size: mean |prediction| / mean |target| = 0.070; correlation(prediction, target) = 0.295 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.49%, -0.52%, -0.29%, -0.54%, -0.50%, -0.46%, -0.56%, -0.50%, -0.62%, -0.66%, -0.58%, -0.56%]; training sample [0.01615, 0.01615, 0.01612, 0.01614, 0.01610, 0.01610, 0.01606, 0.01604, 0.01604, 0.01604, 0.01603, 0.01606]
- fold 2: tune [-0.20%, -0.24%, -0.34%, -0.29%, -0.36%, -0.38%, -0.34%, -0.47%, -0.48%, -0.43%, -0.50%, -0.54%]; training sample [0.01565, 0.01562, 0.01557, 0.01553, 0.01557, 0.01550, 0.01550, 0.01549, 0.01549, 0.01550, 0.01544, 0.01546]
- fold 3: tune [-0.54%, -0.57%, -0.54%, -0.63%, -0.69%, -0.66%, -0.76%, -0.62%, -0.74%, -0.72%]; training sample [0.01570, 0.01567, 0.01566, 0.01562, 0.01562, 0.01564, 0.01562, 0.01566, 0.01560, 0.01557]

### Trial 8
Hypothesis: The learning-rate series is flattening. At lr 0.004 the remaining loss sits in fold-1 close > 1 predictions, which is the case where tune-fitted price_group_scale shrinkage has already helped. Trials 5, 3, 6 and 7 (lr 0.0005, 0.001, 0.002, 0.004) gave -0.491, -0.509, -0.560 and -0.570% vs the baseline. The last doubling gained 0.010 points against 0.051 for the one before. Only fold 2 improved (-0.60 to -0.70%), while fold 1 lost ground (-0.59 to -0.53%) and fold 3 was flat (-0.45 to -0.43%). The higher rate still improved the main gain source: close <= 1 MAE went from -3.44 to -3.78%, close <= 1 RMSE from -5.38 to -5.84%, and correlation from 0.281 to 0.295. Close > 1 MAE weakened, however (-0.11 to -0.09%). In fold 1, price MAE, which weights higher-priced options most, fell from -0.13 to -0.02%, and dates better than persistence fell from 57 to 51, although pooled RMSE improved (-3.59 to -4.01%). The selected fold-1 model (epoch 10) had a training-sample MAE of 0.01604, below every earlier fold-1 value (best 0.01608), and a better tune score than trial 6 (-0.66 vs -0.64%), yet it scored worse. So part of what it fitted for close > 1 rows does not carry over to new dates. The metric charges over-sized predictions at first order, because close > 1 targets are exactly zero in 15.4% of rows and the overall median |y| is only 0.066 std. price_group_scale fits separate alphas in [0, 1] for close <= 1 and close > 1 rows on the 40 tune dates just before the score dates. Trial 4 isolates its effect in fold 1, where the training path and the selected epoch (4 of 7) matched trial 3 exactly. There, calibration alone moved equal-date MAE from -0.58 to -0.60%, pooled MAE from -0.71 to -0.74%, price RMSE from +0.13 to +0.04% and dates better than persistence from 57 to 64, with alpha 0.66 for close > 1 and 0.96 for close <= 1. Close <= 1 alphas were 0.94-0.96 in every fold, so the reversal predictions that supply most of the gain are kept. Trial 4's net loss came in the two folds where calibration changed epoch selection. The flatter calibrated tune curve let patience stop fold 3 at epoch 5 with a full-rate epoch-2 model (-0.31 vs -0.35%), and fold 2 picked epoch 11 instead of 8 (-0.53 vs -0.55%). At lr 0.004 the later tune gains are larger and arrive after halvings (fold 2: -0.34% at epoch 3, -0.47% at 8, -0.54% at 12; fold 3: -0.57% at 2, -0.69% at 5, -0.76% at 7), so they should survive calibration. Calibration should also damp the epoch-to-epoch noise of the high-rate tune curves (fold 1 ranged from -0.29 to -0.66%), which now drives epoch selection.
Change: Builds on trial 7, the best so far. calibration changes from none to price_group_scale. All other parameters keep their trial-7 values: loss l1, smooth_l1_beta 1 (unused), learning_rate 0.004, weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1. Because the tune curve is computed after calibration, learning-rate halving, stopping and best-epoch selection may differ from trial 7.
Predicted change: -0.59%
Configuration: loss=l1, learning_rate=0.004, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=price_group_scale
Objective 0.0143734: -0.519% vs the baseline configuration, -0.496% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.48% | -0.63% | -4.18% | -0.00% | -0.00% | 14/80 | 1 / 4 | close_at_most_1 0.90, close_above_1 0.24 |
| 2 | -0.59% | -0.69% | -4.01% | -0.13% | +0.02% | 56/80 | 11 / 12 | close_at_most_1 0.95, close_above_1 0.83 |
| 3 | -0.44% | -0.48% | -3.14% | -0.14% | +0.15% | 70/80 | 7 / 10 | close_at_most_1 1.00, close_above_1 0.85 |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.50%, RMSE -5.95%; close > 1: MAE -0.07%, RMSE +0.06%; calls: MAE -0.65%, RMSE -4.11%; puts: MAE -0.54%, RMSE -3.45%.
Prediction size: mean |prediction| / mean |target| = 0.062; correlation(prediction, target) = 0.297 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.61%, -0.60%, -0.31%, -0.59%]; training sample [0.01615, 0.01615, 0.01612, 0.01610]
- fold 2: tune [-0.28%, -0.25%, -0.34%, -0.33%, -0.36%, -0.44%, -0.44%, -0.47%, -0.48%, -0.49%, -0.53%, -0.50%]; training sample [0.01565, 0.01562, 0.01557, 0.01553, 0.01557, 0.01550, 0.01550, 0.01549, 0.01549, 0.01550, 0.01548, 0.01547]
- fold 3: tune [-0.57%, -0.59%, -0.55%, -0.66%, -0.69%, -0.66%, -0.77%, -0.62%, -0.76%, -0.72%]; training sample [0.01570, 0.01567, 0.01566, 0.01562, 0.01562, 0.01564, 0.01562, 0.01566, 0.01560, 0.01557]

### Trial 9
Hypothesis: Raising the learning rate added optimizer noise that the run never removes, and decoupled weight decay is the only setting that can remove it. Going from lr 0.002 to 0.004 helped where the signal is (close <= 1 MAE -3.44% to -3.78%, correlation 0.281 to 0.295, fold 2 -0.60% to -0.70%) but hurt where predictions should stay near zero. In fold 1 the selected epoch-10 model had the lowest training-sample MAE of the search (0.01604; earlier fold-1 runs reached at best 0.01608) and a better tune score than trial 6 (-0.66% vs -0.64%), yet it scored -0.53% vs -0.59%, with price MAE, which weights higher-priced options most, at -0.02% vs -0.13% and 51 vs 57 dates better than persistence. Close > 1 RMSE has been above persistence in every L1 trial (+0.06% to +0.67%), and where the training path and selected epoch were unchanged, tune-fitted shrinkage of close > 1 outputs helped slightly (trial 4 fold 1: -0.58% to -0.60% with alpha 0.66; trial 8 fold 3: -0.43% to -0.44% with alpha 0.85), so these outputs carry some excess size on new dates. Calibration is a poor tool for this because it changes the tune curve and with it epoch selection (trial 8 fold 1 stopped at epoch 4 with its epoch-1 model and only 14/80 dates better). Mechanism: AdamW moves each parameter by up to about the learning rate per step, largely independent of the gradient's size, so components without consistent gradient support random-walk by about lr x sqrt(steps), 0.13-0.15 per full-rate epoch at 0.004 (1,022-1,438 steps), comparable to a 1/sqrt(fan-in) weight scale for fan-ins of 15 and 32 (0.26 and 0.18). In directions that are nearly flat on the fit loss nothing pulls these components back, and the halvings only freeze them. On score dates, where standardized inputs such as log_strike or log_volume can lie outside their fit-window range, they add noise to predictions whose conditional median is zero, which equal-date MAE charges at first order (15.4% exact zeros for close > 1, median |y| 0.066 std). At weight_decay 1e-4 the decay is inert: a trial-7 run sums lr x steps to about 19, 32 and 30 in folds 1-3, so total shrinkage is under 0.4%. At 0.1 the per-step decay is 4e-4 at the full rate, unsupported components shrink by 34-44% per full-rate epoch and to about 4-15% over a run, and their stationary spread falls to about sqrt(lr/(2 wd)) = 0.14 at lr 0.004 and 0.05 at 0.0005. Weights that carry signal hold once their Adam-normalized mean gradient reaches wd x |w|, about 0.03 for |w| near 0.3; with 4,096-row batches that needs a mean-to-RMS ratio of the per-row gradient of only about 0.03/64 = 0.0005, far below the reversal signal behind the close <= 1 gains, so the progress gained in trials 6 and 7 should be kept.
Change: Builds on trial 7, the best so far. weight_decay changes from 0.0001 to 0.1, the top of the allowed range; 0.01 would remove only about 18-28% of unsupported components over a trial-7-length run, too little to test the mechanism. All other parameters keep their trial-7 values: loss l1, smooth_l1_beta 1 (unused), learning_rate 0.004, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none. The decoupled decay per step is lr x wd, so it weakens with each halving and acts mostly in the full-rate epochs.
Predicted change: -0.58%
Configuration: loss=l1, learning_rate=0.004, weight_decay=0.1, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143805: -0.470% vs the baseline configuration, -0.447% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.48% | -0.63% | -4.03% | +0.00% | -0.00% | 15/80 | 2 / 5 | - |
| 2 | -0.49% | -0.57% | -4.04% | -0.12% | +0.02% | 54/80 | 6 / 9 | - |
| 3 | -0.39% | -0.41% | -3.42% | -0.15% | +0.10% | 71/80 | 3 / 6 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.15%, RMSE -6.18%; close > 1: MAE -0.06%, RMSE +0.27%; calls: MAE -0.60%, RMSE -4.15%; puts: MAE -0.47%, RMSE -3.50%.
Prediction size: mean |prediction| / mean |target| = 0.058; correlation(prediction, target) = 0.288 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.47%, -0.57%, -0.22%, -0.55%, -0.56%]; training sample [0.01617, 0.01611, 0.01615, 0.01614, 0.01609]
- fold 2: tune [-0.31%, -0.30%, -0.30%, -0.38%, -0.40%, -0.44%, -0.40%, -0.43%, -0.39%]; training sample [0.01558, 0.01563, 0.01570, 0.01558, 0.01561, 0.01559, 0.01565, 0.01561, 0.01561]
- fold 3: tune [-0.44%, -0.36%, -0.64%, -0.41%, -0.57%, -0.55%]; training sample [0.01569, 0.01570, 0.01564, 0.01567, 0.01565, 0.01564]

### Trial 10
Hypothesis: The fit is still limited by how far training gets on the close <= 1 reversal signal within the 12-epoch cap, and the initial learning rate is the one remaining setting that adds progress. In the one-factor series with l1, date_balanced, dropout 0 and no calibration (trials 5, 3, 6, 7 at lr 0.0005, 0.001, 0.002, 0.004), the signal diagnostics improved at every doubling: close <= 1 MAE -3.21, -3.25, -3.44, -3.78%; correlation 0.270, 0.272, 0.281, 0.295; fold-2 equal-date MAE -0.53, -0.55, -0.60, -0.70%; fold-2 training-sample minimum 0.01556, 0.01553, 0.01550, 0.01544. Close <= 1 rows carry most of the objective gain: fold-1 models with no gain on higher-priced options (trial 8 with close > 1 alpha 0.24 and price MAE -0.00%; trial 9 with price MAE +0.00%) still scored -0.48%, close to the -0.53 to -0.60% of the full models in trials 3-7. Higher rates have not cost close > 1 rows on average: their RMSE vs persistence fell from +0.67 to +0.40, +0.28 and +0.14%, and their MAE stayed between -0.06 and -0.11%. The small last step of the objective (-0.560 to -0.570%) came from fold 1, whose selected epoch-10 model had a lower training-sample MAE (0.01604 vs 0.01608) and a better tune score (-0.66 vs -0.64%) than trial 6 yet scored -0.53 vs -0.59%. That fold has scored between -0.47 and -0.60% in all nine L1 trials, so this looks like tune-to-score selection noise rather than a trend in the rate. Fold 2 is plainly progress-limited: it ran 11 or 12 epochs in trials 1-8, and at lr 0.004 its best tune score came at the last epoch (-0.54% at epoch 12, from -0.50% at epoch 11). Trial 9 agrees: weight decay 0.1 slowed fitting (fold-2 training minimum 0.01558, close <= 1 MAE -3.15%) and lost on every fold. Mechanism: under L1 every row contributes a gradient of fixed size, so the 2.7% close <= 1 rows are a small part of each batch and the large reversal outputs they need (mean |y| 0.94 std) build up slowly. AdamW moves each parameter by up to about the learning rate per step, so a 0.008 start doubles the early travel, while the plateau-halving rule still anneals (0.008 -> 0.004 -> 0.002 -> ...) and the best epoch is restored; bounded L1 gradients and clipping at 1.0 should keep training stable. Risks: the random travel of weights without consistent gradient support rises to about 0.26-0.30 per full-rate epoch (0.008 x sqrt of 1,022-1,438 steps), comparable to the 0.18-0.26 weight scale, so full-rate tune curves will be noisier and fold 1 could stop on patience with an early model, as in trials 8 and 9 (-0.48%); the extra annealing stage may leave fold 2 less annealed at the cap; and some ReLU units may die.
Change: Builds on trial 7, the best so far. learning_rate changes from 0.004 to 0.008, the next doubling in the series 0.0005, 0.001, 0.002, 0.004. All other parameters keep their trial-7 values: loss l1, smooth_l1_beta 1 (unused), weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none. Decoupled weight-decay shrinkage per step becomes 8e-7, still negligible.
Predicted change: -0.58%
Configuration: loss=l1, learning_rate=0.008, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0144105: -0.262% vs the baseline configuration, -0.239% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.08% | -0.06% | +0.01% | -0.07% | +0.02% | 60/80 | 8 / 11 | - |
| 2 | -0.41% | -0.52% | -3.36% | +0.03% | +0.01% | 8/80 | 3 / 6 | - |
| 3 | -0.23% | -0.29% | -3.18% | +0.01% | +0.07% | 20/80 | 5 / 8 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -1.97%, RMSE -3.60%; close > 1: MAE +0.00%, RMSE +0.06%; calls: MAE -0.27%, RMSE -2.27%; puts: MAE -0.31%, RMSE -2.10%.
Prediction size: mean |prediction| / mean |target| = 0.029; correlation(prediction, target) = 0.180 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [+0.06%, +0.01%, +0.11%, -0.01%, -0.06%, +0.05%, -0.06%, -0.08%, -0.02%, -0.08%, -0.07%]; training sample [0.01620, 0.01618, 0.01623, 0.01621, 0.01617, 0.01618, 0.01615, 0.01615, 0.01616, 0.01620, 0.01614]
- fold 2: tune [-0.18%, -0.18%, -0.24%, -0.18%, -0.23%, -0.24%]; training sample [0.01568, 0.01565, 0.01560, 0.01554, 0.01563, 0.01556]
- fold 3: tune [-0.44%, -0.24%, -0.41%, -0.51%, -0.54%, -0.53%, -0.54%, -0.53%]; training sample [0.01567, 0.01571, 0.01565, 0.01566, 0.01564, 0.01563, 0.01564, 0.01562]

### Trial 11
Hypothesis: The best initial learning rate lies between 0.002 and 0.004. A rate of 0.003 should keep most of the progress that 0.004 bought in fold 2 and avoid most of the damage it did in fold 1. In the one-factor series (l1, date_balanced, dropout 0, no calibration), lr 0.0005, 0.001, 0.002, 0.004 and 0.008 gave -0.491, -0.509, -0.560, -0.570 and -0.262% vs the baseline configuration. The gain flattens from 0.002 to 0.004 and collapses at 0.008. A quadratic in log2(lr) through 0.002, 0.004 and 0.008 has its minimum near 0.0029; one through 0.001, 0.002 and 0.004 has it near 0.0034. The folds disagree on the rate. Fold 1 scored -0.57, -0.58 and -0.59% at 0.0005, 0.001 and 0.002, then -0.53% at 0.004 and -0.08% at 0.008. Fold 2 improved up to 0.004 (-0.53, -0.55, -0.60, -0.70%, then -0.41% at 0.008). From 0.002 to 0.004 its best tune went from -0.41 to -0.54% and its training-sample minimum from 0.01550 to 0.01544. Fold 3 was flat (-0.45% at 0.002, -0.43% at 0.004). The fold-1 damage grows steadily with the starting rate, so it does not look random. A tune dip recurs at epoch 3, probably because the fixed seed repeats the batch order. The drop from epoch 2 to epoch 3 was absent at 0.0005 (epoch 3 improved, -0.53 -> -0.61%), 0.04 points at 0.001 (-0.67 -> -0.63%), 0.15 at 0.002 (-0.56 -> -0.41%) and 0.23 at 0.004 (-0.52 -> -0.29%). At 0.008, fold-1 tune never got below -0.08%. Mechanism: AdamW moves each parameter by up to about lr per step. More early travel builds the large close <= 1 reversal outputs faster: from 0.0005 to 0.004, close <= 1 MAE went -3.21, -3.25, -3.44, -3.78% and correlation 0.270, 0.272, 0.281, 0.295. It also enlarges the jumps that heavy-tailed rows cause in the shared hidden units (|y| up to 75.9 std; std(y) 0.321 for close <= 1 against 0.0835 for close > 1). Those jumps push close > 1 outputs off their zero median, and equal-date MAE charges that at first order (15.4% exact zeros for close > 1, median |y| 0.066 std). The fold-1 model selected at 0.004 fits this. Its training-sample MAE (0.01604) was lower than any other trial reached in fold 1, yet its price MAE was only -0.02% against -0.11 to -0.13% at 0.0005 to 0.002, and 51 of 80 dates beat persistence against 57 at 0.002. At 0.003 the per-step travel is 25% below 0.004 and 50% above 0.002. The halving ladder (0.0015, 0.00075, 0.000375, 0.00019) still passes through the 0.000125 to 0.001 range where all six best epochs of trials 6 and 7 were found. The main risk is that fold 2's gain at 0.004 depended on its particular high-rate path. Fold 2 would then fall back toward -0.60% and the objective would land near trial 6.
Change: Builds on trial 7, the best so far. learning_rate changes from 0.004 to 0.003, between the two best rates of the series (0.002 in trial 6 and 0.004 in trial 7). All other parameters keep their trial-7 values: loss l1, smooth_l1_beta 1 (unused), weight_decay 0.0001, dropout 0, sample_weighting date_balanced, low_price_weight 1, calibration none. Decoupled weight-decay shrinkage per step becomes 3e-7, still negligible.
Predicted change: -0.58%
Configuration: loss=l1, learning_rate=0.003, weight_decay=0.0001, dropout=0, sample_weighting=date_balanced, low_price_weight=1, calibration=none
Objective 0.0143620: -0.598% vs the baseline configuration, -0.575% vs persistence.

| fold | equal-date MAE | pooled MAE | pooled RMSE | price MAE | price RMSE | dates better than persistence | best / run epochs | calibration alpha |
|---|---|---|---|---|---|---|---|---|
| 1 | -0.56% | -0.67% | -3.62% | -0.08% | +0.16% | 54/80 | 2 / 5 | - |
| 2 | -0.64% | -0.75% | -3.70% | -0.14% | +0.01% | 61/80 | 11 / 12 | - |
| 3 | -0.53% | -0.58% | -2.84% | -0.16% | +0.19% | 68/80 | 12 / 12 | - |

Metric columns are changes relative to persistence on the same score rows (negative is better).
Subgroups vs persistence (mean over folds): close <= 1: MAE -3.72%, RMSE -5.39%; close > 1: MAE -0.12%, RMSE +0.16%; calls: MAE -0.72%, RMSE -3.66%; puts: MAE -0.63%, RMSE -3.16%.
Prediction size: mean |prediction| / mean |target| = 0.074; correlation(prediction, target) = 0.287 (means over folds).
Learning curves by epoch (tune equal-date MAE vs persistence, after any calibration; MAE on a fixed 65,536-row training sample):
- fold 1: tune [-0.60%, -0.66%, -0.42%, -0.61%, -0.66%]; training sample [0.01615, 0.01611, 0.01610, 0.01616, 0.01603]
- fold 2: tune [-0.26%, -0.28%, -0.30%, -0.34%, -0.35%, -0.25%, -0.38%, -0.40%, -0.40%, -0.40%, -0.45%, -0.45%]; training sample [0.01562, 0.01561, 0.01556, 0.01553, 0.01556, 0.01551, 0.01551, 0.01550, 0.01549, 0.01551, 0.01548, 0.01548]
- fold 3: tune [-0.51%, -0.47%, -0.64%, -0.60%, -0.67%, -0.62%, -0.69%, -0.72%, -0.75%, -0.72%, -0.65%, -0.83%]; training sample [0.01570, 0.01567, 0.01566, 0.01567, 0.01563, 0.01564, 0.01563, 0.01564, 0.01557, 0.01557, 0.01557, 0.01558]

Lowest objective so far: trial 11 (0.0143620, -0.598% vs the baseline configuration).

## Your task

Propose exactly 1 configuration for the next evaluation (1 remaining, including this one). When the budget is used up, the configuration with the lowest objective in this search is selected.
