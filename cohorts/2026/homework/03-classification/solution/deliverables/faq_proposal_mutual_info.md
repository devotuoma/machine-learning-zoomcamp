Proposed issue for https://github.com/DataTalksClub/faq/issues/new?template=faq-proposal.yml

**Title:** [FAQ] Homework 3 (2026) Q3: Why do `mutual_info_classif` scores differ from `mutual_info_score`, or change on every run?

### Course

machine-learning-zoomcamp

### Question

Homework 3 (2026), Q3: I computed mutual information with `sklearn.feature_selection.mutual_info_classif` instead of `mutual_info_score`. My scores are different, they change with `random_state`, and sometimes `lead_source` and `employment_status` tie at 0.03. Which function should I use?

### Answer

Use `sklearn.metrics.mutual_info_score`, as in the module 3 lesson. It computes the exact MI between two **discrete** variables from their contingency table, so the result is deterministic:

```python
from sklearn.metrics import mutual_info_score

mi = {c: round(mutual_info_score(y_train, df_train[c]), 2) for c in categorical}
# lead_source 0.03 (0.0343), employment_status 0.02 (0.0190), industry 0.0, location 0.0
```

`mutual_info_classif` is a different tool, designed for feature selection. There are two common traps on this dataset:

**1. Label-encoded categories are treated as continuous.** With the default `discrete_features='auto'` and a dense `X`, the integer codes are treated as continuous. sklearn then uses a k-nearest-neighbours *estimator* that adds random noise, so the result is approximate and seed-dependent. On the training split:

| call | lead_source | employment_status |
|---|---|---|
| `mutual_info_score` (exact) | 0.0343 | 0.0190 |
| `mutual_info_classif(X, y, random_state=0)` | 0.0342 | 0.0194 |
| `mutual_info_classif(X, y, random_state=1)` | 0.0178 | 0.0056 |
| `mutual_info_classif(X, y, random_state=42)` | 0.0319 | 0.0300 ← both round to 0.03 |

Passing `discrete_features=True` makes it exact and identical to `mutual_info_score`:

```python
from sklearn.feature_selection import mutual_info_classif

X = df_train[categorical].apply(lambda s: s.astype('category').cat.codes)
mutual_info_classif(X, y_train, discrete_features=True)   # 0.0343, 0.0020, 0.0190, 0.0010
```

**2. One-hot columns split a feature's information.** If you run MI on `pd.get_dummies` or `DictVectorizer` output, every category becomes its own binary column. The top column is then `employment_status=employed` (0.016), ahead of any single `lead_source` column (0.014 at most), which points you to the wrong answer. Q3 asks about the *variables*, so compute MI on the original categorical columns.

Also note: `mutual_info_score` returns nats (natural log), not bits, so values are small. The argument order doesn't matter, because MI is symmetric. Use the training set only.

Full worked example: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution

### Checklist

- [x] I have searched existing FAQs and this question is not already answered
- [x] The answer provides accurate, helpful information
- [x] I have included any relevant code examples or links
