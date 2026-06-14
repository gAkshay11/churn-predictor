# Customer Churn Predictor

A machine learning app that predicts customer churn for a telecom company and explains *why* each prediction was made, using XGBoost and SHAP.

## Live Demo

🔗 [Live App](https://churn-predictor-07.streamlit.app/)

## Overview

Customer churn is one of the most expensive problems for subscription businesses. This project builds an end-to-end pipeline that predicts whether a customer is likely to churn and breaks down exactly which factors are driving that prediction, for both the overall model and each individual customer.

The model is trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features), covering contract types, billing details, service subscriptions, and demographics.

## Key Results

- **AUC-ROC: 0.8356** on held-out test data
- **Recall on churn class: 0.74** — catches 74% of customers who actually churn
- **Top churn drivers identified via SHAP:**
  1. Month-to-month contract type
  2. Short tenure
  3. High monthly charges
  4. Total charges
  5. No online security add-on

## Features

- **Real-time churn prediction** — input a customer profile and get a churn probability instantly
- **Risk classification** — flags each customer as High or Low risk
- **Global feature importance** — summary and bar plots showing what drives churn across all customers
- **Per-customer explanations** — SHAP waterfall plot showing exactly which features pushed *this specific customer's* prediction up or down

## Tech Stack

- **Python** — core language
- **XGBoost** — gradient boosting classifier
- **SHAP** — model explainability (TreeExplainer)
- **scikit-learn** — preprocessing pipeline (ColumnTransformer, imputation, encoding, scaling)
- **Streamlit** — web app framework
- **pandas / numpy** — data handling
- **matplotlib** — visualization

## Project Structure
churn-predictor/
├── data/ # Dataset (Telco Customer Churn)
├── notebooks/
│ └── 01_eda.ipynb # Exploratory data analysis
├── src/
│ ├── preprocess.py # Data cleaning + preprocessing pipeline
│ ├── train.py # Model training + evaluation
│ └── explain.py # SHAP explainability + plot generation
├── app/
│ ├── app.py # Streamlit application
│ ├── shap_summary.png # Global SHAP summary plot
│ └── shap_bar.png # Global SHAP feature importance (bar)
├── models/
│ └── churn_model.pkl # Trained pipeline (preprocessor + XGBoost)
└── requirements.txt

## How It Works

1. **Preprocessing** — Raw customer data is cleaned (fixing data type issues, handling missing values) and transformed using a scikit-learn pipeline: numeric features are median-imputed and scaled, categorical features are mode-imputed and one-hot encoded.
2. **Model** — An XGBoost classifier is trained on the processed data, with `scale_pos_weight` set to handle the dataset's class imbalance (26.5% churn rate).
3. **Explainability** — SHAP's TreeExplainer computes feature contributions both globally (which features matter most overall) and locally (why this specific customer is predicted to churn or stay).
4. **App** — A Streamlit interface lets users input a customer profile via sidebar controls and instantly see the churn probability, risk label, and SHAP-based explanation.

## How to Run Locally

```bash
# Clone the repo
git clone <https://github.com/gAkshay11/churn-predictor>
cd churn-predictor

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features, sourced from Kaggle.

## Author

Akshay — MS Business Analytics & AI, UT Dallas