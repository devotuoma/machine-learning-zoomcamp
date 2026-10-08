# X (Twitter) thread

Post tweet 1 with `../images/seed_rmse.png` attached, then reply with the rest.
Each tweet is under 280 characters.

---

**1/6**
Module 2 of #mlzoomcamp done ✅

I built a car MPG regression model from scratch with NumPy and the normal equation, no sklearn.

Test RMSE: 2.236 mpg on 10k cars.

The surprise was that ridge regularization made things worse. Thread 🧵👇 @DataTalksClub

**2/6**
EDA: the fuel_efficiency_mpg target is nearly symmetric (skew 0.08, mean ≈ median ≈ 30).

No long tail, so no log1p transform. That's different from the car-price lessons, where log-transforming the target was essential.

**3/6**
Missing values: 877 NaNs (8.77%) in horsepower.

Fill with 0 → val RMSE 2.205
Fill with train mean → 2.202 ✅

A "0 hp" car sits far outside the real 171–334 hp range. Compute the mean on train only, or you leak validation data.

**4/6**
Ridge: r=0 was best. RMSE rose from 2.205 to 2.429 at r=100.

Why? The lecture code adds r·I to the whole XᵀX diagonal, so the intercept is penalized too. With model_year ≈ 2000, w0 needs to be ≈ −154. Ridge drags it to −0.4.

**5/6**
Leaving the intercept unpenalized gives RMSE = 2.2053 for every r.

Lessons:
• don't penalize the bias
• standardize before ridge
• 4 features + 6k rows means there's no overfitting to fix

**6/6**
Seed stability: RMSE std = 0.029 across 10 splits, so the model is stable.

Code + notebook 👇
https://github.com/devotuoma/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression/solution

#MachineLearning #LearningInPublic #Python

---

## Single-post version (if you prefer one tweet)

Module 2 of #mlzoomcamp ✅ Linear regression from scratch in NumPy → test RMSE 2.236 mpg.

TIL: ridge made it worse because the lecture code penalizes the intercept, and unscaled model_year needs w0 ≈ −154.

https://github.com/devotuoma/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression/solution
