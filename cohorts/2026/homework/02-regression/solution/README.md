# Homework 2 Solution: Machine Learning for Regression (ML Zoomcamp 2026)

Predicting `fuel_efficiency_mpg` on the pinned 2026 car fuel-efficiency dataset
(10,000 rows) using linear regression implemented from scratch with the normal
equation in NumPy. No scikit-learn.

## Answers

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| Q1 | Column with missing values | **`horsepower`** | 877 NaNs (8.77%), all other columns 0 |
| Q2 | Median of `horsepower` | **254** | `df.horsepower.median() = 254.0` |
| Q3 | Filling NAs | **With mean** | val RMSE: fill-0 = 2.205, fill-mean = 2.202 (mean = 254.46, train only) |
| Q4 | Best regularization `r` | **0** | 0→2.2053, 0.01→2.2058, 0.1→2.2241, 1→2.3492, 5→2.4094, 10→2.4195, 100→2.4292 |
| Q5 | Std of RMSE over seeds 0–9 | **0.029** | `np.std(scores) = 0.0290`, mean RMSE 2.214 |
| Q6 | Test RMSE (seed 9, train+val, r=0.001) | **2.236** | 2.2358 |

## Files

| File | Purpose |
|------|---------|
| [homework2_solution.ipynb](homework2_solution.ipynb) | Executed notebook with EDA, all questions, plots and commentary |
| [homework2_solution.py](homework2_solution.py) | The same pipeline as a standalone script |
| [images/](images/) | Target distribution and seed-stability plots |
| [deliverables/](deliverables/) | Learning-in-public posts, Medium article and FAQ proposal |

## Reproduce

```bash
# from this folder
uv run --no-project --with numpy --with pandas python homework2_solution.py
# or
pip install numpy pandas matplotlib jupyter && jupyter notebook homework2_solution.ipynb
```

The script reads `cohorts/2026/data/car_fuel_efficiency_2026.csv` if it exists
locally, otherwise it downloads the CSV from the course repo.

## Technical notes

**EDA.** The target is close to symmetric (skew 0.081, mean 29.998 and median
30.0 mpg, std 2.93). There's no long tail, so unlike the car-price lessons no
`log1p` transform is needed.

![target distribution](images/target_distribution.png)

**Q3, imputation.** Filling with 0 creates "0 hp" cars far outside the
observed range (171–334 hp). That distorts the horsepower coefficient,
which collapses to about −5e-6. Filling with the training-set mean keeps those
rows on-distribution. The mean is computed on `df_train` only, so no validation
or test information leaks into the features.

**Q4: why regularization doesn't help here.** The lecture code adds `r·I` to
the full diagonal of `XᵀX`, which also penalises the intercept. The features
are unscaled (`model_year ≈ 2000`, `vehicle_weight ≈ 4300`), so the
unregularised model needs a large negative bias (w0 ≈ −153.9) to offset
`0.102 × 2000 ≈ 204`. As `r` grows, w0 is pulled toward 0 (−32.3 at r=1, −0.4
at r=100) and the fit degrades. If the intercept is left unpenalised, or the
features are standardised, validation RMSE stays at 2.2053 for every `r`. With
4 features and 6,000 rows there's no overfitting for ridge to fix.
`cond(XᵀX) ≈ 6e11` comes from scale, not from collinearity.

**Q5: stability.** RMSE ranges from 2.163 to 2.270 across 10 splits, with
std 0.029 (≈1.3% of the mean). The model is stable.

![seed stability](images/seed_rmse.png)

**Q6: final model.** Retraining on train+val (8,000 rows) with r=0.001 gives a
test RMSE of 2.236 mpg, against a target std of 2.93 mpg. A linear model on
these 4 raw features explains about 42% of the variance (1 − 2.236² / 2.93²).
