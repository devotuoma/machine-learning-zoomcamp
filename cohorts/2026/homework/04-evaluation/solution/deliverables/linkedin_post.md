# LinkedIn post — Homework 4

My model lost to one of its own features. 📉

Week 4 of #mlzoomcamp was evaluation metrics for classification: ROC AUC, precision, recall, F1 and K-fold cross-validation, on a lead-conversion dataset.

The surprise came from comparing two numbers:
🔹 lead_score used alone as a score: AUC 0.783
🔹 Full logistic regression, 8 features including lead_score: AUC 0.732

More features gave a worse ranking. The cause was annual_income: values around 50,000 sitting next to features between 0 and 10. L2-regularized logistic regression isn't scale invariant, so income took over the fit.

One StandardScaler later:
✅ AUC 0.732 → 0.788
✅ The "best C" question disappeared: every C scored 0.792 in 5-fold CV

Other things this week made clear:
1️⃣ Precision and recall crossed at threshold 0.63, but F1 peaked at 0.41. Choose the threshold for the metric you actually care about.
2️⃣ 5-fold CV AUC had std 0.007, so one validation split wasn't a fluke.
3️⃣ A single strong feature is the cheapest baseline you can build. If your model can't beat it, look at preprocessing before tuning.

Code, notebook and plots:
👉 https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution

Thanks to @Alexey Grigorev and the @DataTalks.Club team for a free course that keeps making me question my own results. 🙏

#MachineLearning #DataScience #ModelEvaluation #ROCAUC #ScikitLearn #Python #LearningInPublic #mlzoomcamp #DataTalksClub
