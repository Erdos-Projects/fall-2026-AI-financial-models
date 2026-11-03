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

Evaluations used: 4 of 12.

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

Lowest objective so far: trial 3 (0.0143748, -0.509% vs the baseline configuration).

## Your task

Propose exactly 1 configuration for the next evaluation (8 remaining, including this one). When the budget is used up, the configuration with the lowest objective in this search is selected.
