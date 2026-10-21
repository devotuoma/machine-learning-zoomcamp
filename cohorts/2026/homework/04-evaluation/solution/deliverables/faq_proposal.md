Proposed issue for https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml

**Title:** [FAQ] Homework 4 (2026) Q2/Q6: Why does my model's AUC (0.732) lose to `lead_score` alone, and why does C=0.001 beat C=1?

### Course

machine-learning-zoomcamp

### Question

Homework 4 (2026): In Q1 `lead_score` alone has a training AUC of 0.788, but in Q2 the full logistic regression model only gets 0.732 on validation, even though `lead_score` is one of its features. In Q6 the heavily regularized `C=0.001` (0.749) also beats `C=1` (0.732). Is my model broken?

### Answer

Your model isn't broken, and **0.732 and C=0.001 are still the correct homework answers**. The low score comes from feature scale, not from the evaluation code.

`annual_income` is around 50,000, while every other feature is between 0 and 10 (one-hot columns are 0/1). Logistic regression with an L2 penalty (the sklearn default) is **not scale invariant**. With `C=1.0` the fit leans on income, and its weight comes out around −1e-5. On validation the model predicts `converted` for 82% of leads when the real rate is 59%. The useful features get drowned out, so the model's ranking is worse than ranking by `lead_score` alone (validation AUC 0.783).

That also explains Q6. A smaller `C` shrinks the weights, which limits how much the unscaled income column can distort the fit. That's why 0.001 wins. With `C=1e-6` every weight is close to zero and the model underfits.

Scale the numeric columns (fit on the training data only) and the effect goes away:

```python
from sklearn.preprocessing import StandardScaler

numerical = ["number_of_courses_viewed", "annual_income", "interaction_count", "lead_score"]
scaler = StandardScaler().fit(df_train[numerical])
df_train[numerical] = scaler.transform(df_train[numerical])
df_val[numerical] = scaler.transform(df_val[numerical])
# then DictVectorizer + LogisticRegression(solver='liblinear', C=1.0, max_iter=1000) as usual
```

| setup | validation AUC | 5-fold mean ± std, C=1e-6 / 0.001 / 1 |
|---|---|---|
| homework (unscaled) | 0.732 | 0.617±0.019 / 0.749±0.010 / 0.732±0.007 |
| `lead_score` alone | 0.783 | — |
| numeric features scaled | **0.788** | **0.792±0.013** for all three C |

After scaling, C has no effect on AUC. Even `C=1e-6` ranks leads just as well, because AUC only depends on the order of the scores, not their size. Q6's ranking of C values is caused by the unscaled income column.

Takeaways:
- Always compare your model against the best single feature (Q1 hands you that baseline). If the model can't beat it, check preprocessing.
- Scale numeric features before regularized linear models.
- For the homework, submit the unscaled values the questions ask for (0.732 and C=0.001).

Full worked example with plots: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
