# Baseline performance — short check-in answer

1. We applied the selected `15→32→16→1` MLP to 7,069,922 training examples and evaluated its frozen seed-42 checkpoint on 2,009,818 validation examples. The final test set remains reserved.
2. Validation return MAE was **0.01721**, RMSE **0.04157**, and equal-date MAE **0.0149292**; price MAE/RMSE were **1.717/4.120**. Warm CPU inference took approximately **49 µs per example**. Diagnostic figures show learning curves, daily errors, and subgroup comparisons.
3. Return RMSE improved **5.06%** over persistence, especially for low-price options (**8.02%** improvement). However, low-price options still had much higher absolute RMSE (**0.2122** versus **0.0267** above one price unit). Overall MAE and price errors were slightly worse, so the model has not demonstrated an overall forecasting advantage.
