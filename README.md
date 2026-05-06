# Customer Churn Prediction — Telecom

> End-to-end churn prediction pipeline on **3,150 telecom customer records** with significant missing data (15–21% per feature). Compared Decision Tree, Gradient Boosting, and Random Forest with RandomOverSampler for class imbalance — **Random Forest achieves 90% accuracy, 0.78 precision, and 0.60 recall on churners** after GridSearchCV tuning across 120 parameter combinations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest%20%7C%20GBM-orange?logo=scikit-learn)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-90%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-3%2C150%20customers-purple)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Results

### Model Comparison

| Model | Train Accuracy | Test Accuracy | Churn Precision | Churn Recall | Churn F1 |
|-------|:--------------:|:-------------:|:---------------:|:------------:|:--------:|
| Baseline (majority class) | — | 85% | — | — | — |
| Decision Tree | 99.4% | 88.7% | 0.69 | 0.65 | 0.67 |
| Gradient Boosting | 90.4% | 86.4% | 0.57 | **0.84** | 0.68 |
| **Random Forest (tuned)** | **100%** | **90.0%** | **0.78** | 0.60 | **0.68** |

**Best model:** Tuned Random Forest — highest test accuracy and best precision for churners.

**Note on trade-off:** Gradient Boosting achieves higher churn recall (0.84 vs 0.60) at the cost of lower precision (0.57 vs 0.78). In a real deployment, the business objective determines the right choice — if missing a churner is more costly than a false alarm, Gradient Boosting is preferable. Random Forest is selected here for overall accuracy and balanced precision-recall.

### Best Hyperparameters (GridSearchCV, 5-fold CV, 120 combinations)

```
simpleimputer__strategy:              mean
randomforestclassifier__n_estimators: 75
randomforestclassifier__max_depth:    40
```

### Feature Importances (Random Forest, Gini)

| Rank | Feature | Importance | Business Meaning |
|------|---------|:----------:|-----------------|
| 1 | `Frequency of use` | 0.187 | How often the customer uses the service |
| 2 | `Seconds of Use` | 0.172 | Total usage volume (correlated with above) |
| 3 | `Complains` | 0.155 | **Strongest behavioral churn signal** |
| 4 | `Frequency of SMS` | 0.128 | SMS engagement |
| 5 | `Subscription Length` | 0.118 | Customer tenure |
| 6 | `Distinct Called Numbers` | 0.099 | Network breadth |
| 7 | `Charge Amount` | 0.074 | Billing level |
| 8 | `Call Failure` | 0.068 | Service quality signal |

**Key insight:** `Complains` has the 3rd highest importance despite being a binary feature (0/1). Churning customers complain at 39.8% vs only 1.6% for retained customers — a 25× difference. Customer service complaint resolution is the most actionable lever for churn prevention.

---

## Dataset

- **Source:** Telecom customer behavior dataset (`tech_task_dataset.csv`)
- **Records:** 3,150 customers
- **Target:** `Churn` (0 = retained, 1 = churned)
- **Class distribution:** 84.3% retained / **15.7% churned** (imbalanced)
- **Missing values:** 15–21% per feature (addressed with median imputation)

### Features

| Feature | Type | Missing | Description |
|---------|------|:-------:|-------------|
| `Call Failure` | float | 19.9% | Number of call failures |
| `Complains` | binary | 18.8% | Whether customer filed a complaint |
| `Subscription Length` | float | 16.4% | Months as a customer |
| `Charge Amount` | int | 0% | Monthly charge tier |
| `Seconds of Use` | float | 18.7% | Total seconds of usage |
| `Frequency of use` | float | 21.3% | Number of calls made |
| `Frequency of SMS` | float | 21.0% | Number of SMS sent |
| `Distinct Called Numbers` | float | 20.2% | Unique numbers called |

### Key EDA Findings

- **Class imbalance:** 85% retained vs 15% churned → addressed with `RandomOverSampler`
- **Skewed features:** `Seconds of Use`, `Frequency of use`, `Frequency of SMS` are all right-skewed → median imputation chosen over mean
- **Multicollinearity:** `Seconds of Use` and `Frequency of use` have r = 0.94 (very high). Tree-based models were selected specifically because they are robust to multicollinearity
- **Complains signal:** Churners complain at 39.8% rate vs 1.6% for non-churners — the clearest behavioral predictor

---

## Pipeline Architecture

```
Input (3,150 customers × 8 features)
    ↓
Stratified 80/20 Train/Test Split
    ↓
RandomOverSampler                    → balance minority class (churn) in training set
    ↓
Pipeline per model:
  SimpleImputer(strategy='median')   → fill 15–21% missing values
  Classifier (DT / GBM / RF)
    ↓
GridSearchCV (5-fold CV, 120 fits)   → tune RandomForest hyperparameters
    ↓
Evaluation: accuracy, precision, recall, F1, confusion matrix, feature importances
```

---

## Business Recommendations

**1. Prioritize complaint resolution.** Customers who complain churn at 25× the rate of non-complainers. A dedicated complaint fast-track or proactive outreach after a complaint is filed would directly reduce churn.

**2. Monitor low-usage customers.** `Frequency of use` and `Seconds of Use` are the top two predictors. Customers with declining usage are early churn signals — trigger a retention offer when usage drops below their historical baseline.

**3. Flag long-tenure customers who suddenly complain.** `Subscription Length` is the 5th most important feature. Long-term customers who file a complaint are especially high-risk — they may represent high lifetime value about to be lost.

**4. Call failure rate matters.** `Call Failure` has the lowest importance (0.068) but still contributes — network quality issues accumulate and eventually drive churn.

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── churn_pred.ipynb        # Full pipeline: EDA → preprocessing → modeling → evaluation
├── dataset.csv             # Telecom customer dataset (3,150 records)
├── requirements.txt        # Dependencies
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/Bufatima-Nk/Customer-Churn-Prediction
cd Customer-Churn-Prediction
pip install -r requirements.txt
jupyter notebook churn_pred.ipynb
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Models | scikit-learn (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier) |
| Imbalance | imbalanced-learn (RandomOverSampler) |
| Imputation | scikit-learn SimpleImputer |
| Tuning | GridSearchCV (5-fold CV, 120 combinations) |
| Data | pandas, NumPy |
| Visualization | Matplotlib, Seaborn, ConfusionMatrixDisplay |

---

## Limitations & Future Work

- **Random Forest training accuracy = 100%** suggests overfitting — max_depth of 40 is very deep. Lowering to 15–20 would likely close the train/test gap without hurting test performance.
- **Gradient Boosting recall = 0.84** is compelling for production use — if the cost of missing a churner outweighs the cost of a false positive, GBM should be the deployment choice.
- **SMOTE vs RandomOverSampler** — SMOTE generates synthetic minority samples rather than duplicating existing ones. Worth testing as a more robust alternative.
- **Threshold tuning** — lowering the classification threshold from 0.5 → 0.3 on Random Forest would increase churn recall significantly, likely the right move for a real deployment.
- **Feature engineering** — adding `call_failure_rate` (call failures / frequency of use) and `sms_to_call_ratio` could improve signal quality.

---

## Author

**Bufatima N.K.**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-bufatima--n--k-blue?logo=linkedin)](https://linkedin.com/in/bufatima-n-k)
[![GitHub](https://img.shields.io/badge/GitHub-Bufatima--Nk-black?logo=github)](https://github.com/Bufatima-Nk)
