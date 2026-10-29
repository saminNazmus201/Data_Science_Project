# Data Science Projects

**One week, one project.** A growing collection of end-to-end data science projects: from raw data and exploratory analysis to feature engineering, model training and evaluation.

Each project lives in its own notebook and can be opened directly in Google Colab.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-1B6AC6)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)

---

## Projects

| # | Project | Type | Data | Models | Best result | Notebook |
|---|---------|------|------|--------|-------------|----------|
| 1 | [Uppsala Commute Delay Prediction](#1-uppsala-commute-delay-prediction) | Regression | UL public transport (GTFS + GTFS Realtime) | Random Forest, XGBoost | **MAE 30.7 s, R² 0.91** (XGBoost) | [Open in Colab](https://colab.research.google.com/github/saminNazmus201/Data_Science_Project/blob/master/UppsalaCommuteDelayPrediction.ipynb) |
| 2 | [Customer Churn Prediction](#2-customer-churn-prediction) | Classification | Telco Customer Churn (7,043 customers) | Logistic Regression, KNN, Decision Tree | **F1 0.64, churn recall 0.83** (tuned Logistic Regression) | [Open in Colab](https://colab.research.google.com/github/saminNazmus201/Data_Science_Project/blob/master/churnprediction.ipynb) |
| … | *More coming every week* | | | | | |

---

## 1. Uppsala Commute Delay Prediction

**Goal:** Predict how late (or early) a bus or train will be at its next stop in the Uppsala region, using real public transport data.

### Data
- **Static GTFS timetable** from UL (Upplands Lokaltrafik): `trips`, `stop_times`, `stops`, `routes`.
- **GTFS Realtime TripUpdates** for 1 October 2025, parsed from protobuf files with `gtfs-realtime-bindings` (≈151,000 delay records).
- Static and realtime data were joined on `trip_id` and `stop_id`, giving **84,108 unique stop events** after removing duplicates.

### What I did
1. **Data engineering:** extracted the archives, parsed the binary realtime feed into a DataFrame, converted GTFS times to seconds and merged four static tables with the realtime feed.
2. **Exploratory analysis:**
   - Average delay ≈ **30 s**, standard deviation ≈ **157 s**: most vehicles run close to schedule, with a long tail of large delays.
   - IQR outlier analysis found **5,029 outliers**: 1,332 very early and 3,697 very late arrivals.
   - Plotted how delay **accumulates along a single route** and how delay changes **by hour of the day**.
3. **Feature engineering:** added lag features (`delay_lag1`, `delay_lag2`) so the model knows the delay at the previous two stops of the same trip, together with `hour`, `stop_sequence` and stop coordinates.
4. **Modelling:** trained and compared Random Forest and XGBoost regressors, and checked the residual plot for bias.

### Results

| Model | MAE (seconds) | R² |
|-------|---------------|----|
| Random Forest (100 trees, depth 5) | 32.96 | 0.893 |
| **XGBoost** (100 trees, depth 5, lr 0.1) | **30.70** | **0.909** |

On average, the XGBoost prediction is about 31 seconds away from the real delay.

**Key insight:** delay is strongly "sticky". A vehicle that is late at one stop is very likely to be late at the next, which is why the lag features carry most of the predictive power.

### Next steps
- Split train and test **by trip or by time**, not randomly by row, so stops from the same trip never appear in both sets.
- Use several days of realtime data to capture weekday and weather effects.
- Add route-level and stop-level features such as route type, distance between stops and historical stop averages.

---

## 2. Customer Churn Prediction

**Goal:** Predict which telecom customers are likely to cancel their subscription, so a business can act before they leave.

### Data
- **Telco Customer Churn** dataset: 7,043 customers, 21 columns covering demographics, account details and subscribed services.
- Class balance: **5,174 stay vs 1,869 churn (≈26.5% churn)**, so accuracy alone is misleading and F1 and recall matter most.

### What I did
1. **EDA:** distribution plots for every categorical feature and churn comparisons by gender, senior citizen status, partner, dependents, contract type, internet service, tech support, payment method, tenure and charges.
2. **Cleaning and preprocessing:** dropped `customerID`, converted `TotalCharges` to numeric and filled missing values, encoded categorical columns, and scaled features with `StandardScaler`.
3. **Correlation analysis:** among the positively correlated features, internet service type, monthly charges, paperless billing and senior citizen status had the strongest relationship with churn.
4. **Modelling and tuning:** trained three model families, then tuned each with `GridSearchCV`. Logistic Regression was also evaluated with **nested cross-validation** for an unbiased estimate.

### Results (test set, 1,409 customers)

| Model | Accuracy | F1 (churn) | Notes |
|-------|----------|------------|-------|
| Logistic Regression (baseline) | **0.818** | 0.633 | Highest accuracy, but misses 41% of churners |
| **Logistic Regression (tuned, balanced)** | 0.756 | **0.643** | Catches **83%** of churners (310 / 373) |
| KNN (baseline) | 0.766 | 0.533 | |
| KNN (tuned, k = 29) | 0.795 | 0.595 | |
| Decision Tree (baseline) | 0.732 | 0.492 | Overfits |
| Decision Tree (tuned, balanced) | 0.754 | 0.639 | Catches 82% of churners |

Nested 3-fold CV for Logistic Regression gave a **mean F1 of 0.628**, which confirms the test-set result is stable.

**Key insight:** for churn, missing a leaving customer is usually more costly than contacting one who would have stayed. Accepting slightly lower accuracy in exchange for much higher recall (59% → 83%) is the better business trade-off.

### Next steps
- One-hot encode nominal features instead of mapping them to integers.
- Try ensemble models (Random Forest, XGBoost) and threshold tuning.
- Explain predictions with SHAP to show *why* a given customer is at risk.

---

## Tech stack

- **Language:** Python
- **Data:** pandas, NumPy, gtfs-realtime-bindings (protobuf)
- **Visualisation:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn, XGBoost
- **Environment:** Jupyter Notebook, Google Colab, Kaggle

## How to run

**Option 1 – Google Colab (easiest):** click an *Open in Colab* link in the table above.

**Option 2 – locally:**

```bash
git clone https://github.com/saminNazmus201/Data_Science_Project.git
cd Data_Science_Project
pip install pandas numpy matplotlib seaborn scikit-learn xgboost gtfs-realtime-bindings jupyter
jupyter notebook
```

> **Note:** the datasets are not stored in this repository. Download them from the original sources and update the file paths at the top of each notebook (they currently point to `/content/` on Colab or `/kaggle/input/` on Kaggle).

## Repository structure

```
Data_Science_Project/
├── UppsalaCommuteDelayPrediction.ipynb   # Project 1 – transit delay regression
├── churnprediction.ipynb                 # Project 2 – customer churn classification
└── README.md
```

---

## About me

**MD Nazmus Sadat Samin** – MSc student in Data Science with Machine Learning and Statistics at Uppsala University.
Interested in machine learning, natural language processing and data science.

📫 sadatnazmus567@gmail.com · Stockholm, Sweden

⭐ If you find these projects useful, consider giving the repository a star!
