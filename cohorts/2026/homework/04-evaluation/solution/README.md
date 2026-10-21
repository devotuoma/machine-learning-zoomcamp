# Homework 4 — Evaluation Metrics for Classification (2026)

Solution for [Homework 4](../homework.md) of ML Zoomcamp 2026. A logistic regression model predicts `converted` (did the lead sign up?) on the pinned [2026 lead-scoring release](../../../data/course_lead_scoring_2026.csv). We evaluate it with ROC AUC, precision/recall, F1 and K-fold cross-validation.

## Files

| File | What it is |
|---|---|
| [homework4_solution.py](homework4_solution.py) | Reproducible script that answers all six questions |
| [homework4_solution.ipynb](homework4_solution.ipynb) | Executed notebook with plots and commentary |
| [images/](images/) | Plots used below |
| [deliverables/](deliverables/) | Learning-in-public posts and FAQ proposal |

```bash
pip install pandas scikit-learn matplotlib
python homework4_solution.py          # uses the local CSV, falls back to GitHub
```

## Answers

| # | Question | Result | Answer |
|---|---|---|---|
| 1 | Numerical feature with highest AUC | lead_score 0.788, interaction_count 0.765, number_of_courses_viewed 0.723, annual_income 0.608 | **`lead_score`** |
| 2 | Validation AUC, `C=1.0` | 0.7318 | **0.732** |
| 3 | Precision/recall intersection | P=0.733, R=0.725 at t=0.63 | **0.63** |
| 4 | Threshold with max F1 | F1=0.758 at t=0.41 | **0.41** |
| 5 | Std of 5-fold CV AUC | folds 0.731, 0.730, 0.736, 0.741, 0.721 → std 0.0069 | **0.007** |
| 6 | Best `C` | 1e-6: 0.617±0.019 · 0.001: 0.749±0.010 · 1: 0.732±0.007 | **0.001** |

All six values match `cohorts/2026/data/validate_homework.py`.

## Notes

**Q1: AUC as feature importance.** All four features had AUC > 0.5 on the training set, so none needed inverting. ROC AUC only depends on the ranking of the scores, so it works on raw features with any units.

![Single-feature AUC](images/feature_auc.png)

**Q2: the model loses to one of its own features.** The full model scores 0.732 on validation, but `lead_score` alone scores 0.783. The reason is `annual_income`: values around 50,000 sit next to features between 0 and 10. L2-regularized logistic regression is not scale invariant, so income takes over the fit. Its weight is about −1e-5, and the model predicts "converted" for 82% of validation leads when the real rate is 59%. Scaling the numeric columns with `StandardScaler` (fit on train only) raises the AUC to **0.788**. The homework answer is still 0.732, because the question specifies the unscaled pipeline.

![ROC curves](images/roc_curves.png)

**Q3: precision vs recall.** Precision starts at the base rate (0.586) when every lead is predicted positive and rises with the threshold, while recall falls. They cross at 0.63. Above about 0.97 no lead is predicted positive and both metrics drop to zero. The question says to skip those thresholds.

![Precision and recall](images/precision_recall.png)

**Q4: F1 peaks below the crossover.** F1 is maximal at 0.41, not at 0.63. With 59% positives, a low threshold keeps recall near 0.98 while precision only falls to about 0.62, and the harmonic mean rewards that. The best F1 threshold is not where P = R.

![F1 vs threshold](images/f1_threshold.png)

**Q5 & Q6: cross-validation.** The five fold AUCs sit between 0.721 and 0.741 (std 0.007), so the 0.732 validation score is not a lucky split. `C=0.001` gives the best mean AUC. Strong regularization limits how much the unscaled income column can distort the fit. `C=1e-6` shrinks every weight close to zero and underfits.

![CV by C](images/cv_by_c.png)

## Takeaways

1. Check a strong single feature as a baseline. If the model can't beat it, look at the preprocessing.
2. Pick the threshold for the metric you care about: P = R and max F1 give different thresholds.
3. Report CV mean ± std, not just one validation number.
