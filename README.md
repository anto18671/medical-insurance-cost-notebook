# 🩺 Medical Insurance Cost Prediction — Supervised Learning Project

## 📌 Project Overview

This project builds a **supervised regression model** to predict individual medical insurance charges based on demographic and lifestyle features. Using the [Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance), we explore data, engineer features, train multiple models, and interpret the results.

---

## 🎯 Objective

Predict the **medical charges (`charges`)** billed by health insurance providers using the following features:

* `age` — Age of the individual
* `sex` — Gender (male/female)
* `bmi` — Body mass index
* `children` — Number of dependents
* `smoker` — Smoking status (yes/no)
* `region` — Residential region (northeast, northwest, southeast, southwest)

Performance is evaluated using **RMSE**, **MAE**, and **R²**.

---

## 🧪 Workflow

1. **Problem Definition** — Define prediction goal and success criteria
2. **Exploratory Data Analysis (EDA)** — Understand distributions, correlations, and feature relationships
3. **Data Preprocessing** — Handle categorical encoding and prepare data for modeling
4. **Model Training** — Train and evaluate:

   * Linear Regression
   * Ridge Regression
   * Random Forest Regressor
5. **Model Evaluation** — Compare models on test data
6. **Error Analysis & Interpretation** — Analyze residuals, subgroup performance, and feature importance
7. **Conclusion & Next Steps** — Summarize insights and propose improvements

---

## 📊 Results Summary

| Model             | Test RMSE | Test MAE | R²     |
| ----------------- | --------- | -------- | ------ |
| Random Forest     | 4431.96   | 2441.49  | 0.8735 |
| Linear Regression | 5796.28   | 4181.19  | 0.7836 |
| Ridge Regression  | 5798.27   | 4187.30  | 0.7834 |

✅ **Random Forest** achieved the best performance, capturing non-linear relationships and feature interactions.

---

## 🔍 Key Insights

* **Smoking status**, **age**, and **BMI** are the most impactful features.
* Smokers have significantly higher predicted charges than non-smokers.
* Non-linear models significantly outperform linear baselines.

---

## 🚀 Next Steps

* Explore **gradient boosting** models for further improvements.
* Add **uncertainty estimation** and **calibration**.
* Conduct **fairness analysis** across multiple demographic subgroups.

---

## 🧰 Tech Stack

* **Python 3.11+**
* `numpy`, `pandas`, `matplotlib`
* `scikit-learn` for modeling and evaluation

---

## 📂 Reproducibility

The notebook follows best practices:

* Modular and commented code
* `main()` entry point
* Matplotlib for all plots (one chart per figure)
* Consistent preprocessing via `Pipeline` and `ColumnTransformer`

---

### 📜 License

This project is for educational purposes and distributed under the MIT License.
