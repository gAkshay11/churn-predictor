import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from preprocess import load_and_clean, load_and_split

# ── Load pipeline ──────────────────────────────────────────
pipeline = joblib.load("models/churn_model.pkl")

# ── Reload and split data ──────────────────────────────────
df = load_and_clean("data/telco_churn.csv")
X_train, X_test, y_train, y_test = load_and_split(df)

# ── Extract components from pipeline ──────────────────────
preprocessor = pipeline.named_steps["preprocessor"]
xgb_model    = pipeline.named_steps["classifier"]

# ── Transform test data ────────────────────────────────────
X_test_transformed = preprocessor.transform(X_test)

# ── Get feature names after one-hot encoding ──────────────
feature_names = preprocessor.get_feature_names_out()

# ── SHAP explainer ─────────────────────────────────────────
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# ── Plot 1: Summary plot ───────────────────────────────────
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("app/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")

# ── Plot 2: Bar plot ───────────────────────────────────────
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("app/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_bar.png")

# ── Plot 3: Force plot (single row) ───────────────────────
shap.initjs()
force = shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test_transformed[0],
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.savefig("app/shap_force.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_force.png")

print("SHAP complete.")