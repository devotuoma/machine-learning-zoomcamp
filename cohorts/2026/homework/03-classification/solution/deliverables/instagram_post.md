# Instagram post (carousel)

Instagram doesn't support clickable links in captions. Put the GitHub URL in your
bio or link sticker and keep "link in bio" in the caption.

## Carousel slides (5)

1. **Cover**: "I trained a classifier to predict who signs up 🎯 | ML Zoomcamp 2026 · Module 3" on a plain dark background
2. **Correlation**: `../images/correlation_matrix.png` with the overlay "interaction_count ↔ lead_score: r = 0.92"
3. **What matters**: `../images/mutual_information.png` with the overlay "lead_source carries the most signal. Industry and location ≈ 0"
4. **The twist**: `../images/feature_elimination.png` with the overlay "Removing annual_income IMPROVED accuracy 0.645 → 0.724 😮"
5. **The fix**: `../images/c_sweep_scaling.png` with the overlay "It wasn't a bad feature, it was unscaled. StandardScaler → 0.728 ✅"

## Caption

Module 3 of #mlzoomcamp is done ✅🎯

This week I predicted which of 5,000 leads would sign up for a course, using logistic regression with one-hot encoding.

📊 What I learned:
• Check the baseline first. 60% of leads convert, so guessing "yes" every time already scores 0.60
• Mutual information shows which categories matter. lead_source does, location doesn't
• Smaller C means stronger regularization. Push it too far and the model just predicts the majority class
• Scale your numeric features! Income (~54,000) swamped everything else. Dropping it "helped", but scaling it helped more: 0.645 → 0.728

Full code and notebook: link in bio 🔗

Huge thanks to Alexey Grigorev and @datatalksclub for a free course that teaches by building 🙏

#machinelearning #datascience #python #scikitlearn #logisticregression #classification #learninpublic #mlzoomcamp #ai #mlengineer #datascientist #techcareers
