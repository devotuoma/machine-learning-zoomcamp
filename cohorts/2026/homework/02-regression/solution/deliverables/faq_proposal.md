### Course

machine-learning-zoomcamp

### Question

Homework 2 (2026) Q4: Why does validation RMSE get worse as I increase the regularization parameter `r`, so that `r = 0` is the best?

### Answer

This is expected, and your code is probably correct. On the 2026 dataset the validation RMSE (NAs filled with 0, seed 42, 4 decimals) is:

| r | 0 | 0.01 | 0.1 | 1 | 5 | 10 | 100 |
|---|---|---|---|---|---|---|---|
| RMSE | 2.2053 | 2.2058 | 2.2241 | 2.3492 | 2.4094 | 2.4195 | 2.4292 |

So the best `r` is **0**.

**Why:** the lecture's `train_linear_regression_reg` adds `r * np.eye(...)` to the **whole** diagonal of `XᵀX`, including the position for the column of ones. That means the **intercept (bias) is penalized too**. The features are not scaled (`model_year ≈ 2000`, `vehicle_weight ≈ 4300`), so the best model needs a large negative intercept (w0 ≈ −153.9) to cancel out `0.102 × model_year ≈ 204`. As `r` grows, ridge shrinks w0 toward 0 (−32.3 at r=1, −0.4 at r=100), and the predictions drift away from the target.

You can check this by excluding the intercept from the penalty. RMSE then stays at 2.2053 for every `r`:

```python
X1 = np.column_stack([np.ones(len(X_train)), X_train])
P = r * np.eye(X1.shape[1])
P[0, 0] = 0  # don't regularize the bias
w = np.linalg.solve(X1.T @ X1 + P, X1.T @ y_train)
```

In the lecture's car-price example, regularization helped because `XᵀX` was near-singular: duplicated or one-hot columns caused the weights to explode. In this homework there are only 4 numeric features and 6,000 rows, so there's no overfitting or collinearity for ridge to fix.

**For the homework:** keep the lecture code unchanged and answer `0`. **In practice:** standardize features and don't penalize the intercept. scikit-learn's `Ridge(fit_intercept=True)` already handles the intercept this way.

Full worked example: https://github.com/devotuoma/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression/solution

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
