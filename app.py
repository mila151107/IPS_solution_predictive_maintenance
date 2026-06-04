"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
XGBoost drives the dashboard visuals.
Random Forest and Logistic Regression run in background.
"""

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocessing import compute_features, CategoricalEncoder, NumericalScaler, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

GITHUB_URL   = "https://github.com/mila151107/IPS_solution_predictive_maintenance"
DATA_PATH    = "predictive_maintenance.csv"
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
    xgb = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.joblib"))
    rf  = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.joblib"))
    lr  = joblib.load(os.path.join(ARTIFACT_DIR, "lr_model.joblib"))
    mm  = joblib.load(os.path.join(ARTIFACT_DIR, "mm_scaler.joblib"))
    return {"XGBoost": xgb, "Random Forest": rf, "Logistic Regression": lr}, mm


@st.cache_data
def load_and_prepare():
    df  = pd.read_csv(DATA_PATH)
    df1 = df.copy()
    df  = compute_features(df)

    cols_drop = [TARGET_COL, "UDI", "Process temperature [K]",
                 "Air temperature [K]", "Torque/RPM ratio"]
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
mix_high   = st.sidebar.slider("High risk machines",   1, 10, 3)
mix_medium = st.sidebar.slider("Medium risk machines", 1, 10, 3)
mix_low    = st.sidebar.slider("Low risk machines",    1, 10, 3)
st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

# ──────────────────────────────────────────────
# Predictions — XGBoost drives visuals
# RF + LR run in background
# ──────────────────────────────────────────────

xgb_proba = models["XGBoost"].predict_proba(X)[:, 1]
xgb_pred  = (xgb_proba >= threshold).astype(int)

rf_proba  = models["Random Forest"].predict_proba(X)[:, 1]
lr_proba  = models["Logistic Regression"].predict_proba(
    pd.DataFrame(mm_scaler.transform(X), columns=X.columns)
)[:, 1]

results = X.copy()
results["failure_probability"] = (xgb_proba * 100).round(2)
results["risk_level"]          = results["failure_probability"].apply(
    lambda x: "🔴 HIGH" if x >= 70 else "🟡 MEDIUM" if x >= 30 else "🟢 LOW"
)
results["predicted_failure"]  = xgb_pred
results["Failure Type Text"]  = df1["Failure Type"].values
results["Actual Target"]      = y.values

# ──────────────────────────────────────────────
# Metrics row
# ──────────────────────────────────────────────

st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("XGBoost Accuracy",  f"{accuracy_score(y, xgb_pred):.4f}")
c2.metric("XGBoost AUC",       f"{roc_auc_score(y, xgb_proba):.4f}")
c3.metric("Random Forest AUC", f"{roc_auc_score(y, rf_proba):.4f}")
c4.metric("Threshold",         f"{threshold:.2f}")
st.divider()

# ──────────────────────────────────────────────
# Machine risk assessment — 2 charts
# ──────────────────────────────────────────────

st.subheader("🎯 Machine Risk Assessment")

high_risk = results[results["failure_probability"] >= 70]
med_risk  = results[(results["failure_probability"] >= 30) &
                    (results["failure_probability"] < 70)]
low_risk  = results[results["failure_probability"] < 30]

n_high   = min(mix_high,   len(high_risk))
n_medium = min(mix_medium, len(med_risk))
n_low    = min(mix_low,    len(low_risk))

sample = pd.concat([
    high_risk.sample(n_high,   random_state=42) if n_high   > 0 else pd.DataFrame(),
    med_risk.sample(n_medium,  random_state=42) if n_medium > 0 else pd.DataFrame(),
    low_risk.sample(n_low,     random_state=42) if n_low    > 0 else pd.DataFrame(),
]).sort_values("failure_probability", ascending=True)

labels = [f"Machine {idx}" for idx in sample.index]

def risk_color(p):
    if p >= 70:   return "#e74c3c"
    elif p >= 30: return "#f39c12"
    else:         return "#2ecc71"

bar_colors = [risk_color(p) for p in sample["failure_probability"]]

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Failure Probability by Machine",
                                    "Failure Type vs Probability"))

# Chart 1 — horizontal bar colored by risk level
fig.add_trace(go.Bar(
    x=sample["failure_probability"].round(2),
    y=labels,
    ori