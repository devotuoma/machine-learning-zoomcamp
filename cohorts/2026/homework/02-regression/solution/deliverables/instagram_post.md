# Instagram post (carousel)

Instagram doesn't support clickable links in captions. Put the GitHub URL in your
bio or link sticker and keep "link in bio" in the caption.

## Carousel slides (5)

1. **Cover**: "I built linear regression from scratch 🚗📉 | ML Zoomcamp 2026 · Module 2" on a plain dark background
2. **Target distribution**: `../images/target_distribution.png` with the overlay "Target is symmetric → no log transform needed"
3. **Missing values**: "877 missing horsepower values. Fill with 0 → RMSE 2.205. Fill with train mean → RMSE 2.202 ✅"
4. **The twist**: "Ridge made it WORSE 😮 r=0: 2.205 → r=100: 2.429. Why? The intercept got penalized. Fix: don't regularize the bias + scale your features"
5. **Results**: `../images/seed_rmse.png` with the overlay "Stable across 10 splits (std 0.029). Final test RMSE: 2.236 mpg"

## Caption

Module 2 of #mlzoomcamp is done ✅🚗

This week I predicted car fuel efficiency (MPG) for 10,000 cars using linear regression written from scratch. I used only NumPy and the normal equation, with no ML libraries.

📊 What I learned:
• Always check the target distribution. This one was symmetric, so no log transform
• Impute missing values using training data only, to avoid data leakage
• Ridge regression shouldn't penalize the intercept. When it does and your features aren't scaled, the model breaks
• Test the model across several random splits. Mine stayed within ±0.03 RMSE

🎯 Final test error: 2.236 MPG

Full code and notebook: link in bio 🔗

Thanks @datatalksclub for the free course 🙏

#machinelearning #datascience #python #numpy #linearregression #learninpublic #mlzoomcamp #ai #coding #techcareers #datascientist #mlengineer
