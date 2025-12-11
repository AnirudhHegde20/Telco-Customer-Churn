import os
import sys

# --- Make sure we can import from src/ when running via Streamlit ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.model_io import load_model

import streamlit as st
import pandas as pd


# ---------- Load model + config (cached) ----------
@st.cache_resource
def get_model_and_config():
    pipeline, config = load_model(model_dir="models")
    return pipeline, config


pipeline, config = get_model_and_config()
BEST_THRESHOLD = config["best_threshold"]


def risk_badge(label: str) -> str:
    """Return HTML for a colored pill based on risk level."""
    colors = {
        "HIGH": "#f94144",
        "MEDIUM": "#f9c74f",
        "LOW": "#43aa8b",
    }
    color = colors.get(label, "#4a4a4a")
    return f"""
    <span style="
        background-color:{color};
        color:white;
        padding:4px 10px;
        border-radius:999px;
        font-weight:600;
        font-size:0.9rem;
    ">{label}</span>
    """


# ---------- Page layout ----------
st.set_page_config(
    page_title="Telco Churn Risk & Retention Tool",
    layout="wide",
)

st.title("📉 Telco Customer Churn Risk & Retention Tool")
st.caption(
    "End-to-end churn prediction, profit optimization, and retention strategy for telecom customers."
)

st.markdown(
    """
This app uses a trained **Random Forest churn model** with a **profit-optimized threshold**
to estimate churn risk and guide retention decisions for a single customer.
"""
)

st.sidebar.header("Customer Features")

# Feature list must match training columns (except customerID)
feature_names = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# ---------- Sidebar inputs: customer features ----------
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.number_input(
    "Tenure (months)", min_value=0, max_value=1000, value=12
)

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No phone service", "No", "Yes"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security", ["No internet service", "No", "Yes"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup", ["No internet service", "No", "Yes"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection", ["No internet service", "No", "Yes"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support", ["No internet service", "No", "Yes"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No internet service", "No", "Yes"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No internet service", "No", "Yes"]
)

contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)

paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=100000.0,
    value=float(tenure) * float(monthly_charges),
    help="Approximate as Tenure × MonthlyCharges if unknown.",
)

# ---------- Sidebar: business settings ----------
st.sidebar.markdown("---")
st.sidebar.subheader("Business Settings")

C_CHURN = st.sidebar.number_input(
    "Estimated loss if customer churns ($)",
    min_value=50.0,
    max_value=1000.0,
    value=float(config["c_churn"]),
    step=10.0,
)

C_OFFER = st.sidebar.number_input(
    "Cost of retention offer ($)",
    min_value=0.0,
    max_value=200.0,
    value=float(config["c_offer"]),
    step=5.0,
)

# ---------- Build input DataFrame ----------
input_data = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

input_df = pd.DataFrame([input_data], columns=feature_names)

st.subheader("Current Customer Profile")
st.dataframe(input_df, use_container_width=True)

# ---------- Prediction + business impact ----------
if st.button("Predict churn risk"):
    # Churn probability
    proba = pipeline.predict_proba(input_df)[0, 1]

    # Risk classification based on best threshold
    if proba >= BEST_THRESHOLD * 2:
        risk_label = "HIGH"
    elif proba >= BEST_THRESHOLD:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    expected_loss = proba * C_CHURN
    expected_gain_if_targeted = proba * C_CHURN - C_OFFER

    st.markdown("### 🔮 Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.metric("Churn probability", f"{proba:.3f}")
    with col2:
        st.markdown(
            f"Risk level (threshold = {BEST_THRESHOLD:.2f})<br>"
            f"{risk_badge(risk_label)}",
            unsafe_allow_html=True,
        )

    st.markdown("### 💸 Business Impact")
    col3, col4 = st.columns(2)

    with col3:
        st.metric(
            "Expected loss if we do nothing",
            f"${expected_loss:.2f}",
        )
    with col4:
        st.metric(
            f"Expected gain if we target with a "
            f"${C_OFFER:.0f} offer",
            f"${expected_gain_if_targeted:.2f}",
        )

    # Recommendation text
    if risk_label == "HIGH":
        st.success(
            "PRIORITY RETENTION: consider a loyalty discount, plan optimization, "
            "and call-center outreach."
        )
    elif risk_label == "MEDIUM":
        st.info(
            "MODERATE RISK: a targeted email/SMS offer or smaller incentive could be effective."
        )
    else:
        st.write(
            "LOW RISK: no immediate action needed. Monitor the customer but save budget "
            "for higher-risk segments."
        )
