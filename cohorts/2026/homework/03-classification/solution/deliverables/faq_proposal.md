Proposed issue for https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml

**Title:** [FAQ] Homework 3 (2026): Why does removing `annual_income` increase validation accuracy?

### Course

machine-learning-zoomcamp

### Question

Homework 3 (2026), Q5: When I do feature elimination, removing `annual_income` *increases* validation accuracy from 0.645 to 0.724. Is income a useless feature? Should I drop it?

### Answer

No. `annual_income` does carry signal: converters earn about 54.6k on average and non-converters about 45.0k. The problem is **scale**, not the feature.

`annual_income` is around 54,000, while every other column is between 0 and about 10 (one-hot columns, `lead_score`, counts). `LogisticRegression` with an L2 penalty (the default) is **not scale invariant**. The huge raw column dominates the liblinear optimisation and the fit ends up badly conditioned. You can see it in the fitted model:

```python
w = dict(zip(dv.get_feature_names_out(), model.coef_[0]))
print(w['annual_income'])           # ≈ -4.6e-06, negative even though converters earn more
print(model.predict(X_val).mean())  # ≈ 0.87, predicts "converted" for 87% of leads (true rate ≈ 0.60)
```

Standardize the numerical columns (fit the scaler on the training set only) and income helps instead of hurting:

```python
from sklearn.preprocessing import StandardScaler

num_idx = [i for i, n in enumerate(dv.get_feature_names_out()) if n in numerical]
scaler = StandardScaler().fit(X_train[:, num_idx])
X_train[:, num_idx] = scaler.transform(X_train[:, num_idx])
X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])
# validation accuracy: 0.645 -> 0.728 (C=1), and stays > 0.71 for every C in Q6
```

`np.log1p(annual_income)` instead of scaling gives 0.722.

**For the homework:** don't scale, and keep the setup exactly as written. The answer options (Q4 = 0.65, Q5 = `number_of_courses_viewed`, Q6 = 0.001) are calculated from the unscaled features. **In practice:** always scale numeric features before regularized logistic regression, and before you conclude from feature elimination that a feature is useless.

Full worked example: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
