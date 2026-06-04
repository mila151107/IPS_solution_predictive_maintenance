"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from preprocessing import preprocess

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

DATA_PATH  = "predictive_maintenance.csv"
LGBM_PATH  = "artifacts/lgbm_model.joblib"
XGB_PATH   = "artifacts/xgb_model.joblib"
RF_PATH    = "artifacts/rf_model.joblib"
LR_PATH    = "artifacts/lr_model.joblib"
MM_PATH    = "artifacts/mm_scaler.joblib"
GITHUB_URL = "https://github.com/mila151107/IPS_solution_predictive_maintenance"


# ──────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    lgbm = joblib.load(LGBM_PATH)
    xgb  = joblib.load(XGB_PATH)
    rf   = joblib.load(RF_PATH)
    lr   = joblib.load(LR_PATH)
    mm   = joblib.load(MM_PATH)
    return lgbm, xgb, rf, lr, mm

@st.cache_data
def get_processed_data():
    df = load_data()
    X, y, _ = preprocess(df, fit=False)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    return X, y


# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance — Failure Risk Dashboard")
st.markdown(f"Real-time machine failure risk assessment. [View source on GitHub]({GITHUB_URL})")
st.divider()

with st.spinner("Loading data and models..."):
    df_raw                    = load_data()
    lgbm, xgb, rf, lr, mm_scaler = load_models()
    X, y                      = get_processed_data()


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")

model_name = st.sidebar.radio(
    "Select model",
    ["LightGBM", "XGBoost", "Random Forest", "Logistic Regression"]
)

threshold = st.sidebar.slider(
    "Failure prediction threshold",
    min_value=0.10,
    max_value=0.50,
    value=0.25,
    step=0.05,
    help="Lower = catch more failures. Higher = fewer false alarms."
)

sample_size = st.sidebar.slider(
    "Number of rows to predict",
    min_value=10, max_value=500, value=100, step=10,
)

st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")


# ──────────────────────────────────────────────
# Predictions
# ──────────────────────────────────────────────

model_map = {
    "LightGBM":            lgbm,
    "XGBoost":             xgb,
    "Random Forest":       rf,
    "Logistic Regression": lr,
}
model    = model_map[model_name]
X_sample = X.iloc[:sample_size]
y_sample = y.iloc[:sample_size]

X_input = pd.DataFrame(
    mm_scaler.transform(X_sample), columns=X_sample.columns
) if model_name == "Logistic Regression" else X_sample

y_proba = model.predict_proba(X_input)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)


# ──────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────

total      = len(y_pred)
failures   = int(y_pred.sum())
no_failure = total - failures

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total machines",        total)
c2.metric("🔴 Predicted failures", failures)
c3.metric("🟢 No failure",         no_failure)
c4.metric("Threshold",             f"{threshold:.2f}")
st.divider()


# ──────────────────────────────────────────────
# Distribution chart
# ──────────────────────────────────────────────

st.subheader("🎯 Failure Probability Distribution")

results_df = X_sample.copy()
results_df["Failure Probability"] = (y_proba * 100).round(2)
results_df["Predicted"]           = y_pred
results_df["Actual"]              = y_sample.values
results_df["Risk Level"]          = results_df["Failure Probability"].apply(
    lambda x: "🔴 HIGH" if x >= 70 else "🟡 MEDIUM" if x >= 30 else "🟢 LOW"
)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=results_df["Failure Probability"],
    nbinsx=20,
    marker_color="#3498db",
    opacity=0.75,
))
fig.add_vline(
    x=threshold * 100, line_dash="dash", line_color="red",
    annotation_text=f"Threshold ({threshold*100:.0f}%)"
)
fig.update_layout(
    xaxis_title="Failure Probability (%)",
    yaxis_title="Number of Machines",
    plot_bgcolor="white",
    height=350,
)
st.plotly_chart(fig, use_container_width=True)
st.divider()


# ──────────────────────────────────────────────
# Results table
# ──────────────────────────────────────────────

st.subheader("📋 Prediction Results")

extra_cols = [c for c in ["Rotational_speed__rpm_", "Torque__Nm_", "Tool_wear__min_"]
              if c in results_df.columns]

st.dataframe(
    results_df[["Risk Level", "Failure Probability", "Predicted", "Actual"] + extra_cols],
    use_container_width=True,
    height=400,
)
st.divider()


# ──────────────────────────────────────────────
# Single machine prediction
# ──────────────────────────────────────────────

st.subheader("🔍 Predict a Single Machine")
st.markdown("Adjust sensor values and click **Predict**.")

c1, c2, c3 = st.columns(3)
with c1:
    rpm       = st.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500)
    torque    = st.number_input("Torque [Nm]", min_value=1.0, max_value=100.0, value=40.0)
with c2:
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=300, value=100)
    air_temp  = st.number_input("Air temperature [K]", min_value=295.0, max_value=305.0, value=300.0)
with c3:
    proc_temp = st.number_input("Process temperature [K]", min_value=305.0, max_value=315.0, value=310.0)
    prod_type = st.selectbox("Product type", ["L", "M", "H"])

if st.button("🔮 Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        "UDI":                     1,
        "Product ID":              f"{prod_type}-00001",
        "Type":                    prod_type,
        "Air temperature [K]":     air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]":  rpm,
        "Torque [Nm]":             torque,
        "Tool wear [min]":         tool_wear,
        "Target":                  0,
        "Failure Type":            "No Failure",
    }])

    X_single, _, _ = preprocess(input_df, fit=False)
    X_single.columns = X_single.columns.str.replace(
        r"[\[\]/]", "_", regex=True).str.strip()

    if model_name == "Logistic Regression":
        X_single = pd.DataFrame(
            mm_scaler.transform(X_single), columns=X_single.columns)

    proba = model.predict_proba(X_single)[0][1]
    pred  = int(proba >= threshold)

    if pred == 1:
        st.error(f"🔴 **Failure predicted** — {proba*100:.1f}% probability")
    else:
        st.success(f"🟢 **No failure predicted** — {proba*100:.1f}% probability")


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(
    f"Dataset: AI4I 2020 Predictive Maintenance | "
    f"Model: {model_name} | Threshold: {threshold} | "
    f"[GitHub]({GITHUB_URL})"
)
