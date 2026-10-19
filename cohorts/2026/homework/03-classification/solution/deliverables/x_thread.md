# X (Twitter) thread

Post tweet 1 with `../images/c_sweep_scaling.png` attached, then reply with the rest.
Each tweet is under 280 characters.

---

**1/6**
Module 3 of #mlzoomcamp done ✅

I built a lead-conversion classifier with logistic regression on 5k leads.

The surprise: removing a feature took accuracy from 0.645 to 0.724. Thread 🧵👇

@DataTalksClub @Al_Grigor

**2/6**
EDA:
• industry mode → technology
• interaction_count vs lead_score: r = 0.92
• Mutual information: lead_source wins (0.03). industry and location ≈ 0

Baseline check: 60% of validation leads convert, so always guessing "yes" already scores 0.60.

**3/6**
Logistic regression + one-hot (DictVectorizer), liblinear, C=1 → val accuracy 0.645.

Regularization sweep: C=0.001 is best. At C ≤ 1e-5 the weights collapse and accuracy falls to 0.598, the majority-class baseline.

**4/6**
Feature elimination: drop annual_income and accuracy jumps to 0.724 🤯

The feature isn't useless. It's unscaled. Income is ~54,000 while everything else is 0–10, and L2-regularized logistic regression isn't scale invariant.

**5/6**
Symptoms: the income weight came out negative even though converters earn more, and the model said "converted" for 87% of leads.

Fix: StandardScaler on the numeric columns (fit on train only) → 0.728, stable above 0.71 for every C.

**6/6**
Lessons:
• scale before regularized LR
• a feature that "hurts" may just be badly scaled
• always check the baseline

Code + notebook 👇
https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution

Thanks @Al_Grigor & @DataTalksClub 🙏

---

## Single-post version (if you prefer one tweet)

Module 3 of #mlzoomcamp ✅ Logistic regression for lead conversion → 0.645 val accuracy.

TIL: dropping annual_income gave 0.724. Unscaled income (~54k) swamps L2-regularized LR, and StandardScaler gets 0.728.

Thanks @Al_Grigor @DataTalksClub 🙏
https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw03-classification/cohorts/2026/homework/03-classification/solution
