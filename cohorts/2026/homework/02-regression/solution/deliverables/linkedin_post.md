# LinkedIn post

Attach: `../images/seed_rmse.png` and `../images/target_distribution.png`

---

📉 ML Zoomcamp 2026, Module 2: linear regression from scratch, and a regularization result I didn't expect

This week in #mlzoomcamp by DataTalks.Club I built a fuel-efficiency (MPG) regression model on 10,000 cars. I used only NumPy and the normal equation, w = (XᵀX + rI)⁻¹Xᵀy, with no scikit-learn.

What I did:
🔹 EDA: the target is almost symmetric (skew 0.08, mean ≈ median ≈ 30 mpg), so it doesn't need a log transform
🔹 Missing data: 8.77% of horsepower is NaN. Mean imputation (training mean only, to avoid leakage) beat fill-with-0: RMSE 2.202 vs 2.205
🔹 Stability: across 10 random 60/20/20 splits, validation RMSE std = 0.029 (≈1.3% of the mean), so the model is stable
🔹 Final model: retrained on train+val, test RMSE = 2.236 mpg

The interesting part was regularization. r = 0 won, and RMSE got worse as r grew (2.205 → 2.429 at r=100).

The reason: the textbook ridge implementation adds r to the whole diagonal of XᵀX, including the bias term. With unscaled features like model_year ≈ 2000, the model needs an intercept of about −154. Ridge shrinks it toward 0 (−0.4 at r=100), so the fit breaks.

When I left the intercept out of the penalty, RMSE stayed at 2.2053 for every r. Takeaways:
1️⃣ Don't penalize the intercept
2️⃣ Standardize features before applying ridge
3️⃣ Regularization fixes variance and ill-conditioning. With 4 features and 6k rows there's little variance to fix.

Code, notebook and plots:
👉 https://github.com/devotuoma/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression/solution

Thanks to Alexey Grigorev and DataTalks.Club for a free course that makes you build things yourself.

#MachineLearning #DataScience #LinearRegression #Python #NumPy #LearningInPublic #mlzoomcamp #DataTalksClub
