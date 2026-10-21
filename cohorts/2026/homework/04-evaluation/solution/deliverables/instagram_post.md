# Instagram carousel — Homework 4

**Format:** 6 slides, 1080×1350, dark background, one big number per slide. Plots come from [../images/](../images/).

| Slide | Visual | On-slide text |
|---|---|---|
| 1 | Big "0.732 < 0.783" in red | **My model lost to ONE feature** · ML Zoomcamp week 4 |
| 2 | `feature_auc.png` | **AUC = how well a score ranks yes above no.** lead_score alone: 0.788 |
| 3 | `roc_curves.png` | **Full model: 0.732.** Same data, 8 features, worse ranking 🤔 |
| 4 | "50,000 vs 0–10" with a scale/balance icon | **Culprit: unscaled income.** StandardScaler → 0.788 ✅ |
| 5 | `precision_recall.png` + `f1_threshold.png` stacked | **P = R at 0.63. Max F1 at 0.41.** Choose the threshold for the metric you care about |
| 6 | Checklist | ✔ Beat your best single feature ✔ Scale before regularizing ✔ Report CV mean ± std (0.732 ± 0.007) · Link in bio |

## Caption

My model scored worse than one of its own inputs. 📉

Week 4 of #mlzoomcamp was all about evaluation: ROC AUC, precision, recall, F1 and cross-validation.

lead_score on its own ranked leads better (AUC 0.783) than my 8-feature logistic regression (0.732). The reason: annual_income is about 50,000 while everything else is between 0 and 10, and regularized logistic regression isn't scale invariant. One StandardScaler → 0.788.

Lesson: your best single feature is a free baseline. Beat it before you tune anything. 💡

Swipe for the plots 👉 Full code at the link in bio.

Thank you @datatalksclub & Alexey Grigorev 🙏

#machinelearning #datascience #python #scikitlearn #learninpublic #mlzoomcamp #datatalksclub #aiengineer #codinglife #techcommunity
