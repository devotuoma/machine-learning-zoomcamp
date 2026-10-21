# Instagram carousel — Homework 4

**Format:** 5 slides, 1080×1350. Plots from [../images/](../images/).

| Slide | Visual | Text |
|---|---|---|
| 1 | "0.732 < 0.783" in red | **My model lost to ONE feature** |
| 2 | `roc_curves.png` | lead_score alone: 0.783 · Full model: 0.732 |
| 3 | "50,000 vs 0–10" | **Unscaled income.** StandardScaler → 0.788 ✅ |
| 4 | `precision_recall.png` | P = R at 0.63 · Max F1 at 0.41 |
| 5 | Checklist | ✔ Beat your best single feature ✔ Scale first ✔ CV: 0.732 ± 0.007 |

## Caption

My model scored worse than one of its own inputs. 📉

lead_score alone: AUC 0.783. My 8-feature model: 0.732. The cause was unscaled income (~50,000 vs features between 0 and 10). StandardScaler → 0.788.

Lesson: beat your best single feature before tuning anything. 💡

Code: https://github.com/devotuoma/machine-learning-zoomcamp/tree/hw04-evaluation/cohorts/2026/homework/04-evaluation/solution (also in bio)

Thanks @datatalksclub 🙏

#machinelearning #datascience #python #learninpublic #mlzoomcamp
