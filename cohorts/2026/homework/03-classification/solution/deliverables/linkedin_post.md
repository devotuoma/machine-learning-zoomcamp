# LinkedIn post

Attach: `../images/c_sweep_scaling.png` and `../images/feature_elimination.png`.
When you type "@Alexey Grigorev" and "@DataTalks.Club" in LinkedIn, pick them
from the dropdown so they become real tags.

---

🎯 ML Zoomcamp 2026, Module 3: classification, and the feature that "hurt" my model

This week in #mlzoomcamp by DataTalks.Club I built a lead-conversion classifier on 5,000 leads. The question was whether each lead signs up. I used one-hot encoding (DictVectorizer) and logistic regression.

What I did:
🔹 Data prep: 6 of 9 columns had missing values. Categoricals got 'NA', numericals got 0.0
🔹 Correlation: interaction_count and lead_score are almost the same signal (r = 0.92)
🔹 Mutual information: lead_source is the most informative categorical feature. industry and location carry ~0
🔹 Logistic regression: validation accuracy 0.645
🔹 Regularization sweep: C = 0.001 was best. At C ≤ 1e-5 the model falls back to the majority-class baseline (0.598)

The interesting part came from feature elimination. Removing annual_income *raised* accuracy from 0.645 to 0.724.

The feature isn't useless. It's unscaled. Income is around 54,000 while every other column is between 0 and 10. Logistic regression with an L2 penalty isn't scale invariant, so income dominated the fit. Its weight even came out negative, though converters earn more on average. The model predicted "converted" for 87% of leads when the real rate is 60%.

One StandardScaler on the numeric columns (fit on train only) gave:
✅ 0.645 → 0.728 accuracy
✅ stable above 0.71 for every C I tried

Takeaways:
1️⃣ Scale numeric features before regularized logistic regression
2️⃣ A feature that "hurts" in elimination may just be badly scaled
3️⃣ Always compare against the majority-class baseline

Code, notebook and plots:
👉 https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution

A big thank you to @Alexey Grigorev and the @DataTalks.Club team for a free, hands-on course where every week makes you build something real. 🙏

#MachineLearning #DataScience #Classification #LogisticRegression #ScikitLearn #Python #LearningInPublic #mlzoomcamp #DataTalksClub
