"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
Random Forest drives the dashboard visuals.
XGBoost and Logistic Regression run in background.
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score, roc_auc_score

from preprocessing import (
    compute_features, CategoricalEncoder, NumericalScaler,
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

GITHUB_URL   = "https://github.com/mila151107/IPS_solution_predictive_maintenance"
DATA_PATH    = "data/predictive_maintenance.csv"
ARTIFACT_DIR = "artifacts"

COLORS_MAP = {
    "Power Failure":            "#e74c3c",
    "Tool Wear Failure":        "#3498db",
    "Overstrain Failure":       "#f39c12",
    "Heat Dissipation Failure": "#2ecc71",
    "Random Failures":          "#9b59b6",
    "No Failure":               "#95a5a6",
}

# ──────────────────────────────────────────────
# Load models & data (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    rf  = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.joblib"))
    xgb = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    lr  = joblib.load(os.path.join(ARTIFACT_DIR, "lr_model.joblib"))
    mm  = joblib.load(os.path.join(ARTIFACT_DIR, "mm_scaler.joblib"))
    return {"Random Forest": rf, "XGBoost": xgb, "Logistic Regression": lr}, mm


@st.cache_data
def load_and_prepare():
    df  = pd.read_csv(DATA_PATH)
    df1 = df.copy()
    df  = compute_features(df)

    cols_drop = [TARGET_COL, "UDI", "Process temperature [K]",
                 "Air temperature [K]", "Torque/RPM ratio", "Product ID"]
    X = df.drop(columns=cols_drop, errors="ignore")
    y = df[TARGET_COL]

    encoder = CategoricalEncoder(columns=CATEGORICAL_COLS)
    X = encoder.fit_transform(X)
    scaler = NumericalScaler(columns=NUMERICAL_COLS)
    X = scaler.fit_transform(X)
    X.columns = X.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    return X, y, df1


# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance — Failure Risk Dashboard")
st.markdown(f"Real-time machine failure risk assessment for maintenance engineers. "
            f"[View source on GitHub]({GITHUB_URL})")
st.divider()

with st.spinner("⚙️ Loading models and data..."):
    models, mm_scaler = load_models()
    X, y, df1         = load_and_prepare()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
threshold  = st.sidebar.slider("Failure prediction threshold",
                                min_value=0.10, max_value=0.50,
                                value=0.25, step=0.05,
                                help="Lower = catch more failures. Higher = fewer false alarms.")
mix_high   = st.sidebar.slider("High risk machines",   1, 10, 5)
mix_low    = st.sidebar.slider("Medium risk machines", 1, 10, 5)
st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

# ──────────────────────────────────────────────
# Predictions — RF drives visuals
# XGBoost + LR run in background
# ──────────────────────────────────────────────

rf_proba  = models["Random Forest"].predict_proba(X)[:, 1]
rf_pred   = (rf_proba >= threshold).astype(int)

xgb_proba = models["XGBoost"].predict_proba(X)[:, 1]
lr_proba  = models["Logistic Regression"].predict_proba(
    pd.DataFrame(mm_scaler.transform(X), columns=X.columns)
)[:, 1]

results = X.copy()
results["failure_probability"] = (rf_proba * 100).round(2)
results["risk_level"]          = results["failure_probability"].apply(
    lambda x: "🔴 HIGH" if x >= 70 else "🟡 MEDIUM" if x >= 30 else "🟢 LOW"
)
results["predicted_failure"]  = rf_pred
results["Failure Type Text"]  = df1["Failure Type"].values
results["Actual Target"]      = y.values

# ──────────────────────────────────────────────
# Model performance metrics
# ──────────────────────────────────────────────

st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Random Forest F1",  f"{f1_score(y, rf_pred):.4f}")
c2.metric("Random Forest AUC", f"{roc_auc_score(y, rf_proba):.4f}")
c3.metric("XGBoost AUC",       f"{roc_auc_score(y, xgb_proba):.4f}")
c4.metric("Threshold",         f"{threshold:.2f}")
st.divider()

# ──────────────────────────────────────────────
# Machine risk assessment — 2 charts
# ──────────────────────────────────────────────

st.subheader("🎯 Machine Risk Assessment")

high_risk = results[results["failure_probability"] >= 70]
med_risk  = results[(results["failure_probability"] >= 30) &
                    (results["failure_probability"] < 70)]

n_high = min(mix_high, le