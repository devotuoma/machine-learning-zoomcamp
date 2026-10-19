# The Feature That "Hurt" My Classifier Was Just Unscaled: Logistic Regression on 5,000 Leads

*Subtitle: ML Zoomcamp 2026, Module 3. Mutual information, one-hot encoding, feature elimination, regularization, and why you should scale before you drop a feature.*

*Suggested Medium tags: Machine Learning, Logistic Regression, Data Science, Python, Mlzoomcamp*

---

Module 3 of [DataTalks.Club's ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) covers binary classification with logistic regression. The 2026 homework uses a lead-scoring dataset. Each row is a person who showed interest in an online course, and the target `converted` says whether they signed up.

The homework questions themselves are straightforward. One side result wasn't. **Removing `annual_income` from the model raised validation accuracy by eight points.** Working out why taught me more than the homework did.

All code is on GitHub: [devotuoma/machine-learning-zoomcamp · 03-classification/solution](https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution)

---

## 1. The data

There are 5,000 leads and 9 columns:

```python
categorical = ['lead_source', 'industry', 'employment_status', 'location']
numerical = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']
target = 'converted'
```

| feature | mean | missing |
|---|---|---|
| annual_income | 54,533 | 7.4% |
| number_of_courses_viewed | 2.09 | 0% |
| interaction_count | 4.47 | 0% |
| lead_score | 0.49 | 0.7% |

Note the scales: one column is in the tens of thousands and the rest are single digits. That turns out to matter.

Following the homework, missing categoricals become the string `'NA'` and missing numericals become `0.0`:

```python
df[categorical] = df[categorical].fillna('NA')
df[numerical] = df[numerical].fillna(0.0)
```

`'NA'` is not thrown away. `DictVectorizer` treats it as its own category, so the model can learn whether "didn't tell us their industry" predicts anything.

**Class balance.** 57.7% of leads convert overall, and 59.7% in the validation split. **A model that always says "yes" already scores about 0.60.** Keep that number in mind.

## 2. Q1 and Q2: mode and correlation

The most common `industry` is **technology** (1,173 rows).

![Correlation matrix](../images/correlation_matrix.png)

Of the pairs the homework asks about, **`interaction_count` and `lead_score`** correlate most strongly (r = 0.916). `lead_score` looks like a score derived mostly from engagement, so the two features largely duplicate each other.

## 3. Q3: mutual information

The data is split 60/20/20 with `train_test_split(..., random_state=42)` twice. Mutual information is then computed on the training set only:

```python
from sklearn.metrics import mutual_info_score

mi = {c: round(mutual_info_score(y_train, df_train[c]), 2) for c in categorical}
# {'lead_source': 0.03, 'employment_status': 0.02, 'industry': 0.0, 'location': 0.0}
```

![Mutual information](../images/mutual_information.png)

**`lead_source` wins.** How someone found the course says more about whether they'll sign up than where they live or what industry they work in.

## 4. Q4: logistic regression

One-hot encode with `DictVectorizer` and fit with the parameters the homework fixes:

```python
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train[features].to_dict(orient='records'))
X_val = dv.transform(df_val[features].to_dict(orient='records'))

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)
accuracy_score(y_val, model.predict(X_val))   # 0.645
```

Validation accuracy is **0.645, which rounds to 0.65**. That's only about 5 points above the "always yes" baseline.

## 5. Q5: feature elimination, and the twist

Drop each feature in turn, retrain, and measure how much accuracy falls:

| feature removed | accuracy | difference |
|---|---|---|
| interaction_count | 0.601 | +0.044 |
| lead_source | 0.642 | +0.003 |
| number_of_courses_viewed | 0.643 | **+0.002** |
| lead_score | 0.644 | +0.001 |
| industry / location / employment_status | 0.645 | 0.000 |
| **annual_income** | **0.724** | **−0.079** |

![Feature elimination](../images/feature_elimination.png)

Among the three options the homework offers, **`number_of_courses_viewed`** has the smallest difference. Its information is largely covered by `interaction_count` and `lead_score`.

The bottom row is the interesting one. **Without `annual_income` the model gets 8 points better.** A naive reading says "income is noise, drop it". The data disagrees: converters earn 54.6k on average and non-converters 45.0k, so income clearly carries signal.

### What is actually happening

Logistic regression with an L2 penalty is **not scale invariant**. `annual_income` is about 10,000 times larger than every other column. Under liblinear's optimisation with the default penalty, that column dominates and the problem is badly conditioned. The symptoms are easy to spot once you look:

```python
w = dict(zip(dv.get_feature_names_out(), model.coef_[0]))
w['annual_income']           # -4.6e-06  (negative, even though converters earn more)
model.predict(X_val).mean()  # 0.874     (predicts "converted" for 87% of leads; true rate is 60%)
```

The model ends up as a slightly better "always yes" classifier.

## 6. Q6: regularization, and the fix

In scikit-learn, `C` is the *inverse* regularization strength, so smaller `C` means stronger shrinkage:

| C | 1e-6 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| accuracy | 0.598 | 0.598 | 0.613 | **0.645** |

**Best: C = 0.001.** At 1e-6 and 1e-5 the weights are crushed to zero and the model predicts the majority class. 0.598 is essentially the baseline.

Now standardize the four numerical columns. Fit the scaler on train only to avoid leakage, then repeat the sweep:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train[:, num_idx])
X_train[:, num_idx] = scaler.transform(X_train[:, num_idx])
X_val[:, num_idx] = scaler.transform(X_val[:, num_idx])
```

![C sweep, raw vs scaled](../images/c_sweep_scaling.png)

| C | 1e-6 | 1e-4 | 1e-3 | 1e-1 | 1 |
|---|---|---|---|---|---|
| raw | 0.598 | 0.613 | 0.645 | 0.645 | 0.645 |
| **scaled** | **0.712** | **0.713** | **0.717** | **0.727** | **0.728** |

With scaling, accuracy rises to **0.728**, better than dropping income (0.724), and it stays above 0.71 even with very strong regularization. A `log1p` transform of income alone also works (0.722).

## 7. Takeaways

1. **Scale numeric features before regularized logistic regression.** The penalty treats all weights equally, so it only makes sense when the features are on comparable scales.
2. **Feature elimination can mislead you.** A feature that "hurts" may be fine and badly preprocessed. Check its scale and its relationship with the target before you delete it.
3. **Always report the baseline.** 0.645 sounds fine until you learn that "always yes" gets 0.60.
4. **Correlated features make each other look unimportant.** `number_of_courses_viewed` barely matters once `interaction_count` and `lead_score` are in the model. That doesn't make it useless on its own.

For the homework, keep the setup exactly as specified. The answer options are calculated from it. In a real project, the scaler is a one-line change worth 8 accuracy points.

## Answers summary

| Q | Answer |
|---|---|
| 1 | technology |
| 2 | interaction_count and lead_score |
| 3 | lead_source |
| 4 | 0.65 |
| 5 | number_of_courses_viewed |
| 6 | 0.001 |

---

*Thank you to **Alexey Grigorev** and the whole **DataTalks.Club** team for building ML Zoomcamp and keeping it free. The course makes you work through every step yourself, and that's how I ran into this week's lesson. If you're learning ML, join the next cohort: [github.com/DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).*
