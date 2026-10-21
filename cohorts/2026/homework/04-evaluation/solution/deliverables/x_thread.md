# X — Homework 4

## Thread (6 posts, each under 280 characters)

**1/**
My logistic regression lost to one of its own input features. 🧵

#mlzoomcamp week 4: evaluation metrics for classification. ROC AUC, precision/recall, F1, K-fold CV. Here's what I learned 👇

**2/**
Q1: use each numeric feature directly as a score and compute AUC.

lead_score 0.788
interaction_count 0.765
courses_viewed 0.723
annual_income 0.608

AUC only cares about ranking, so raw features work fine as scores.

**3/**
Q2: the full model (8 features, lead_score included) scored validation AUC 0.732.

lead_score alone: 0.783.

More features, worse ranking. 🤔

**4/**
The cause: annual_income ≈ 50,000 next to features between 0 and 10.

L2-regularized logistic regression isn't scale invariant. Income took over the fit.

Add StandardScaler → AUC 0.788, and every C from 1e-6 to 1 gives 0.792 in 5-fold CV.

**5/**
Thresholds:
• precision = recall at 0.63
• max F1 at 0.41

With 59% positives, a low threshold keeps recall near 0.98 while precision barely drops. The "balanced" point isn't the best F1.

**6/**
Takeaways:
1. Your best single feature is a free baseline. Beat it.
2. Scale before regularizing.
3. Report CV mean ± std (mine: 0.732 ± 0.007).

Code + plots: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

Thanks @Al_Grigor @DataTalksClub 🙏

## Single post version

My model (AUC 0.732) lost to one of its own features, lead_score alone (0.783). 📉

Culprit: unscaled annual_income dominating an L2-regularized logistic regression. StandardScaler → 0.788.

Beat your best single feature first.

#mlzoomcamp #MachineLearning #LearningInPublic

## X Spaces talking points (10-minute slot)

**Title:** "My model lost to one feature: lessons from evaluating a classifier"

1. **Hook (1 min):** "Who has trained a model that scored worse than one of its own inputs?" Share 0.732 vs 0.783.
2. **AUC as feature importance (2 min):** AUC is a ranking metric, so any numeric column can be a score. Flip the sign if AUC < 0.5.
3. **The scaling trap (3 min):** income ≈ 50,000 vs features between 0 and 10. The L2 penalty isn't scale invariant. StandardScaler fixes it, and C stops mattering (0.792 for every C).
4. **Thresholds (2 min):** P = R at 0.63, max F1 at 0.41. Ask listeners: what does a false positive cost in your use case?
5. **CV (1 min):** 5 folds, std 0.007. One split is an anecdote, five are evidence.
6. **Close (1 min):** "Beat your best single feature first." Point to the repo and invite questions.
