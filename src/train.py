import sys
import os
sys.path.append(os.path.dirname(__file__))

import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from preprocess import load_and_clean, load_and_split, build_preprocessor

def train_model():
    # Step 1: Load and clean raw CSV
    df = load_and_clean(r"C:\Akshay\PROJECTS\churn-predictor\data\telco_churn.csv")
    
    # Step 2: Split into train and test sets
    X_train, X_test, y_train, y_test = load_and_split(df)

    # Step 3: Build preprocessor
    preprocessor = build_preprocessor()

    # Step 4: Build full pipeline (preprocessor + model in one object)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2.77,
            eval_metric='auc',
            random_state=42
        ))
    ])

    # Step 5: Train
    model.fit(X_train, y_train)

    # Step 6: Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAUC-ROC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Step 7: Save full pipeline to disk
    os.makedirs(r"C:\Akshay\PROJECTS\churn-predictor\models", exist_ok=True)
    save_path = r"C:\Akshay\PROJECTS\churn-predictor\models\churn_model.pkl"
    joblib.dump(model, save_path)
    print(f"\nModel saved to: {save_path}")

if __name__ == '__main__':
    train_model()