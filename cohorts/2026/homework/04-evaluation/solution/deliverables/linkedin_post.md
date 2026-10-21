# LinkedIn post — Homework 4

My model lost to one of its own features. 📉

#mlzoomcamp week 4: evaluating a lead-conversion classifier.

🔹 lead_score alone: AUC 0.783
🔹 Logistic regression with 8 features: AUC 0.732

The cause: annual_income (~50,000) next to features between 0 and 10. L2-regularized logistic regression isn't scale invariant.
✅ StandardScaler: 0.732 → 0.788

Also learned:
1️⃣ Precision = recall at threshold 0.63, but F1 peaks at 0.41
2️⃣ 5-fold CV: 0.732 ± 0.007
3️⃣ Your best single feature is a free baseline. Beat it before you tune.

Code + plots 👉 https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

Thanks @Alexey Grigorev and @DataTalks.Club 🙏

#MachineLearning #DataScience #ModelEvaluation #ScikitLearn #LearningInPublic #mlzoomcamp
