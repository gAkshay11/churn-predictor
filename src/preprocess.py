import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# ── Feature lists (based on EDA) ──────────────────────────────────────────────
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# ── Load + clean ───────────────────────────────────────────────────────────────
def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.drop(columns=['customerID'])
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

# ── Preprocessor ───────────────────────────────────────────────────────────────
def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

# ── Split + verify ─────────────────────────────────────────────────────────────
def load_and_split(filepath):
    df = load_and_clean(filepath)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_split('../data/telco_churn.csv')
    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Train shape: {X_train_transformed.shape}")
    print(f"Test shape:  {X_test_transformed.shape}")
    print(f"Churn rate train: {y_train.mean():.3f}")
    print(f"Churn rate test:  {y_test.mean():.3f}")
    print("Preprocessor smoke test passed.")