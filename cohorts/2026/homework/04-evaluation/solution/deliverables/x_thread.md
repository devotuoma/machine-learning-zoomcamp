# X — Homework 4

## Thread (4 posts, each under 280 characters)

**1/**
My model lost to one of its own features. 🧵

#mlzoomcamp week 4: ROC AUC, precision/recall, F1, K-fold CV.

lead_score alone: AUC 0.783
8-feature logistic regression: 0.732

**2/**
Cause: annual_income (~50,000) next to features between 0 and 10. L2-regularized logistic regression isn't scale invariant.

StandardScaler → 0.788. Every C then scores 0.792 in 5-fold CV.

**3/**
Thresholds:
• precision = recall at 0.63
• max F1 at 0.41

Choose the threshold for the metric you care about.

5-fold CV: 0.732 ± 0.007

**4/**
Lesson: your best single feature is a free baseline. Beat it.

Code + plots: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

Thanks @Al_Grigor @DataTalksClub 🙏

## Single post version

My model (AUC 0.732) lost to one of its own features, lead_score (0.783). 📉 Unscaled income was the cause. StandardScaler → 0.788.

https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

#mlzoomcamp #LearningInPublic

## X Spaces talking points (5 minutes)

**Title:** "My model lost to one feature"

1. **Hook:** 0.732 (model) vs 0.783 (lead_score alone).
2. **Cause:** unscaled income + L2 penalty. StandardScaler → 0.788, and C stops mattering.
3. **Thresholds:** P = R at 0.63, max F1 at 0.41. Ask listeners what a false positive costs them.
4. **CV:** 0.732 ± 0.007. One split is an anecdote.
5. **Close:** "Beat your best single feature first." Share the repo: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution
