import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

@st.cache_resource
def get_explainer():
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    return preprocessor, explainer

preprocessor, explainer = get_explainer()

st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
TotalCharges = st.sidebar.slider("Total Charges ($)", 0.0, 8700.0, 1000.0)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
Partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
Dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
PhoneService = st.sidebar.selectbox("Phone Service", ["No", "Yes"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
PaymentMethod = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

predict_button = st.sidebar.button("Predict Churn")
st.title("Customer Churn Prediction")
st.write("Enter customer details in the sidebar and click **Predict Churn**.")

if predict_button:
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if SeniorCitizen == "Yes" else 0],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.1%}")

    if probability >= 0.5:
        st.error("High risk of churn")
    else:
        st.success("Low risk of churn")

    st.subheader("What's Driving This Prediction")
    st.write("Global feature importance from model training:")

    col1, col2 = st.columns(2)
    with col1:
        st.image("app/shap_summary.png", caption="Feature Impact Direction")
    with col2:
        st.image("app/shap_bar.png", caption="Feature Importance Ranking")

    st.subheader("Why This Specific Customer")
    st.write("Force plot showing how each feature pushed this prediction:")

    transformed_input = preprocessor.transform(input_data)
    feature_names = preprocessor.get_feature_names_out()
    shap_values = explainer(transformed_input)
    shap_values.feature_names = list(feature_names)

    st.caption("Note: f(x) shows the model's raw output (log-odds), not the churn probability shown above.")

    fig, ax = plt.subplots(figsize=(12, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)