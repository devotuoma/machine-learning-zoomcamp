# My Model Lost to One of Its Own Features

*ML Zoomcamp week 4: evaluating a classifier with ROC AUC, precision, recall, F1 and cross-validation.*

**Code + notebook:** https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

---

The task: predict whether a lead `converted` from 4 categorical and 4 numerical features (5,000 rows, 60/20/20 split, `random_state=1`).

## 1. AUC works on raw features

ROC AUC only measures ranking, so any numeric column can act as a score:

```python
auc = roc_auc_score(y_train, df_train[c])
if auc < 0.5:
    auc = roc_auc_score(y_train, -df_train[c])   # flip negative relationships
```

`lead_score` 0.788 · `interaction_count` 0.765 · `number_of_courses_viewed` 0.723 · `annual_income` 0.608

![Single-feature AUC](../images/feature_auc.png)

## 2. The model that lost

`DictVectorizer` + `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)` gives a validation AUC of **0.732**. On the same validation set, `lead_score` alone scores **0.783**.

The cause: `annual_income` is around 50,000 while the other features are between 0 and 10. The L2 penalty isn't scale invariant, so income dominates the fit. The model flags 82% of leads as converting when the true rate is 59%.

```python
scaler = StandardScaler().fit(df_train[numerical])   # fit on train only
```

Scaled: **0.788**.

![ROC curves](../images/roc_curves.png)

## 3. Thresholds: P = R is not max F1

![Precision and recall](../images/precision_recall.png)

Precision and recall cross at **0.63**. F1 peaks at **0.41** (0.758): with 59% positives, a low threshold keeps recall at 0.98 while precision only falls to 0.62. Choose the threshold based on what errors cost you.

## 4. Cross-validation

5-fold AUC: **0.732 ± 0.007**, so the single split was representative.

| C | 1e-6 | **0.001** | 1 |
|---|---|---|---|
| mean AUC | 0.617 | **0.749** | 0.732 |

C=0.001 wins only because strong regularization limits the unscaled income column. With scaled features, every C scores 0.792.

## Takeaways

1. Your best single feature is a free baseline. Beat it.
2. Scale before regularized linear models.
3. Choose thresholds based on cost.
4. Report CV mean ± std.

**Answers:** lead_score · 0.732 · 0.63 · 0.41 · 0.007 · 0.001

**Full code:** https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

*Thanks to Alexey Grigorev and DataTalks.Club for ML Zoomcamp.*

*Tags: Machine Learning, Data Science, Model Evaluation, Python, Scikit Learn*
