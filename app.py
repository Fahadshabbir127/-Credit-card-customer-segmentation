import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved files
kmeans_model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
log_columns = joblib.load('log_columns.pkl')

# Cluster names mapping (based on our analysis)
cluster_names = {
    0: "Low Activity / Dormant Customer",
    1: "Cash Advance Revolver (Risky Segment)",
    2: "Premium / High-Value Customer",
    3: "Budget-Conscious / Responsible Shopper"
}

cluster_descriptions = {
    0: "Low balance, low purchases, minimal card usage. Recommendation: Re-engagement offers.",
    1: "High balance, low purchases, high cash advance usage. Recommendation: Repayment plans, risk monitoring.",
    2: "High credit limit, high purchases, high payments. Recommendation: Loyalty rewards, credit limit increase.",
    3: "Moderate purchases, high full-payment ratio. Recommendation: Reward responsible behavior."
}

st.set_page_config(page_title="Credit Card Customer Segmentation", layout="centered")
st.title("💳 Credit Card Customer Segmentation")
st.write("Enter customer details below to predict their behavioral segment.")

# Input widgets
col1, col2 = st.columns(2)

with col1:
    balance = st.number_input("Balance", min_value=0.0, value=1000.0)
    balance_frequency = st.slider("Balance Frequency", 0.0, 1.0, 0.8)
    purchases = st.number_input("Purchases", min_value=0.0, value=500.0)
    oneoff_purchases = st.number_input("One-off Purchases", min_value=0.0, value=300.0)
    installments_purchases = st.number_input("Installment Purchases", min_value=0.0, value=200.0)
    cash_advance = st.number_input("Cash Advance", min_value=0.0, value=0.0)
    purchases_frequency = st.slider("Purchases Frequency", 0.0, 1.0, 0.5)
    oneoff_purchases_frequency = st.slider("One-off Purchases Frequency", 0.0, 1.0, 0.3)
    purchases_installments_frequency = st.slider("Installments Purchases Frequency", 0.0, 1.0, 0.3)

with col2:
    cash_advance_frequency = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0)
    cash_advance_trx = st.number_input("Cash Advance Transactions", min_value=0, value=0)
    purchases_trx = st.number_input("Purchases Transactions", min_value=0, value=10)
    credit_limit = st.number_input("Credit Limit", min_value=0.0, value=4000.0)
    payments = st.number_input("Payments", min_value=0.0, value=1000.0)
    minimum_payments = st.number_input("Minimum Payments", min_value=0.0, value=300.0)
    prc_full_payment = st.slider("% Full Payment", 0.0, 1.0, 0.1)
    tenure = st.number_input("Tenure (months)", min_value=0, value=12)

if st.button("Predict Segment"):
    input_data = pd.DataFrame({
        'BALANCE': [balance],
        'BALANCE_FREQUENCY': [balance_frequency],
        'PURCHASES': [purchases],
        'ONEOFF_PURCHASES': [oneoff_purchases],
        'INSTALLMENTS_PURCHASES': [installments_purchases],
        'CASH_ADVANCE': [cash_advance],
        'PURCHASES_FREQUENCY': [purchases_frequency],
        'ONEOFF_PURCHASES_FREQUENCY': [oneoff_purchases_frequency],
        'PURCHASES_INSTALLMENTS_FREQUENCY': [purchases_installments_frequency],
        'CASH_ADVANCE_FREQUENCY': [cash_advance_frequency],
        'CASH_ADVANCE_TRX': [cash_advance_trx],
        'PURCHASES_TRX': [purchases_trx],
        'CREDIT_LIMIT': [credit_limit],
        'PAYMENTS': [payments],
        'MINIMUM_PAYMENTS': [minimum_payments],
        'PRC_FULL_PAYMENT': [prc_full_payment],
        'TENURE': [tenure]
    })

    # Apply same preprocessing: log transform
    input_log = input_data.copy()
    for col in log_columns:
        input_log[col] = np.log1p(input_log[col])

    # Scale
    input_scaled = scaler.transform(input_log)

    # Predict
    cluster = kmeans_model.predict(input_scaled)[0]

    st.success(f"### Predicted Segment: {cluster_names[cluster]}")
    st.info(cluster_descriptions[cluster])
