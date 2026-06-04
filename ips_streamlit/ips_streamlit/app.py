"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
Loads preprocessed data, runs predictions using trained models,
and displays results with a link to the GitHub repository.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from ips_train.preprocessing import preprocess

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DATA_PATH         = "data/predictive_maintenance.csv"
LGBM_PATH         = "artifacts/lgbm_model.joblib"
LR_PATH           = "artifacts/lr_model.joblib"
PREPROCESSOR_PATH = "artifacts/preprocessor.joblib"
GITHUB_URL        = "https://github.com/mila151107/IPS_solution_predictive_maintenance"

# ──────────────────────────────────────────────
# Load resources (cached so they load only once)
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    lgbm = joblib.load(LGBM_PATH)
    lr   = joblib.load(LR_PATH)
    return lgbm, lr

@st.cache_data
def get_processed_data():
    df = load_data()
    X, y, _ = preprocess(df, fit=False)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    return X, y

# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🔧",
    layout="wide"
)

# Header
st.title("🔧 Predictive Maintenance — Failure Classifier")
st.markdown(
    f"Predicts whether a machine is likely to fail based on sensor readings. "
    f"[View source code on GitHub]({GITHUB_URL})"
)
st.divider()

# Load everything
with st.spinner("Loading data and models..."):
    df_raw      = load_data()
    lgbm, lr    = load_models()
    X, y        = get_processed_data()

# ──────────────────────────────────────────────
# Sidebar — controls
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.radio(
    "Select model",
    ["LightGBM", "Logistic Regression"]
)

model = lgbm if model_name == "LightGBM" else lr

sample_size = st.sidebar.slider(
    "Number of rows to predict",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

# ──────────────────────────────────────────────
# Predictions
# ──────────────────────────────────────────────

X_sample  = X.iloc[:sample_size]
y_sample  = y.iloc[:sample_size]

y_pred    = model.predict(X_sample)
y_proba   = model.predict_proba(X_sample)[:, 1]

# ──────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────

total     = len(y_pred)
failures  = int(y_pred.sum())
no_failure = total - failures

col1, col2, col3 = st.columns(3)
col1.metric("Total machines", total)
col2.metric("🔴 Predicted failures", failures)
col3.metric("🟢 Predicted no failure", no_failure)

st.divider()

# ──────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────

st.subheader("📋 Prediction Results")

results_df = X_sample.copy()
results_df["Actual"]            = y_sample.values
results_df["Predicted"]         = y_pred
results_df["Failure Probability"] = (y_proba * 100).round(2)
results_df["Result"] = results_df["Predicted"].map({1: "🔴 Failure", 0: "🟢 No Failure"})

st.dataframe(
    results_df[["Result", "Failure Probability", "Actual"]].join(
        X_sample[["Rotational_speed__rpm_", "Torque__Nm_", "Tool_wear__min_"]]
    ),
    use_container_width=True,
    height=400
)

st.divider()

# ──────────────────────────────────────────────
# Single row prediction — manual input
# ──────────────────────────────────────────────

st.subheader("🔍 Predict a Single Machine")
st.markdown("Adjust the sensor values and click **Predict** to check if the machine will fail.")

c1, c2, c3 = st.columns(3)
with c1:
    rpm     = st.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500)
    torque  = st.number_input("Torque [Nm]", min_value=1.0, max_value=100.0, value=40.0)
with c2:
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=300, value=100)
    air_temp  = st.number_input("Air temperature [K]", min_value=295.0, max_value=305.0, value=300.0)
with c3:
    proc_temp = st.number_input("Process temperature [K]", min_value=305.0, max_value=315.0, value=310.0)
    prod_type = st.selectbox("Product type", ["L", "M", "H"])

if st.button("🔮 Predict", use_container_width=True):
    # Build a single-row DataFrame matching raw input format
    input_df = pd.DataFrame([{
        "UDI": 1,
        "Product ID": f"{prod_type}-00001",
        "Type": prod_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Target": 0,
        "Failure Type": "No Failure",
    }])

    X_input, _, _ = preprocess(input_df, fit=False)
    X_input.columns = X_input.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    pred  = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"🔴 **Failure predicted** — {proba*100:.1f}% probability of failure")
    else:
        st.success(f"🟢 **No failure predicted** — {proba*100:.1f}% probability of failure")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(
    f"Dataset: AI4I 2020 Predictive Maintenance | "
    f"Model: {model_name} | "
    f"[GitHub]({GITHUB_URL})"
)
