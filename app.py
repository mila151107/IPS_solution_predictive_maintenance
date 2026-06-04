"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
Loads data from GitHub repo, trains models on startup, and displays
interactive failure risk predictions for maintenance engineers.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from preprocessing import (
    compute_features, CategoricalEncoder, NumericalScaler,
    COLUMNS_TO_DROP, CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

GITHUB_URL   = "https://github.com/mila151107/IPS_solution_predictive_maintenance"
DATA_PATH    = "predictive_maintenance.csv"
ARTIFACT_DIR = "artifacts"
LGBM_PATH    = os.path.join(ARTIFACT_DIR, "lgbm_model.joblib")
RF_PATH      = os.path.join(ARTIFACT_DIR, "rf_model.joblib")
ENCODER_PATH = os.path.join(ARTIFACT_DIR, "encoder.joblib")
SCALER_PATH  = os.path.join(ARTIFACT_DIR, "scaler.joblib")

COLORS_MAP = {
    "Power Failure":            "#e74c3c",
    "Tool Wear Failure":        "#3498db",
    "Overstrain Failure":       "#f39c12",
    "Heat Dissipation Failure": "#2ecc71",
    "Random Failures":          "#9b59b6",
    "No Failure":               "#95a5a6"
}

# ──────────────────────────────────────────────
# Load & train (cached)
# ──────────────────────────────────────────────

@st.cache_resource
def load_and_train():
    # Load raw data
    df   = pd.read_csv(DATA_PATH)
    df1  = df.copy()

    # Feature engineering
    df = compute_features(df)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[TARGET_COL]
    )

    # Separate features/target
    cols_drop = [TARGET_COL] + ["UDI", "Process temperature [K]",
                                "Air temperature [K]", "Torque/RPM ratio", "Product ID"]
    X_train = train_df.drop(columns=cols_drop, errors="ignore")
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=cols_drop, errors="ignore")
    y_test  = test_df[TARGET_COL]

    # Encode — fit on train only
    encoder = CategoricalEncoder(columns=CATEGORICAL_COLS)
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test  = encoder.transform(X_test)

    # Scale — fit on train only
    scaler = NumericalScaler(columns=NUMERICAL_COLS)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train.columns = X_train.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    X_test.columns  = X_test.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    # SMOTE
    # smote = SMOTE(sampling_strategy=0.3, random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train models
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15,
                           max_depth=5, min_child_samples=20, class_weight="balanced",
                           importance_type="gain", random_state=42)
    rf   = RandomForestClassifier(n_estimators=100, max_depth=5,
                                   class_weight="balanced", random_state=42)
    lgbm.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions
    y_proba_rf   = rf.predict_proba(X_test)[:, 1]
    y_pred_rf    = (y_proba_rf >= 0.3).astype(int)

    # Results DataFrame
    results = X_test.copy()
    results["failure_probability"] = (y_proba_rf * 100).round(2)
    results["risk_level"]          = results["failure_probability"].apply(
        lambda x: "🔴 HIGH" if x >= 70 else "🟡 MEDIUM" if x >= 30 else "🟢 LOW")
    results["predicted_failure"]   = y_pred_rf
    results["Failure Type Text"]   = df1.loc[results.index, "Failure Type"].values
    results["Actual Target"]       = df1.loc[results.index, "Target"].values

    metrics = {
        "lgbm_f1":  f1_score(y_test, lgbm.predict(X_test)),
        "lgbm_auc": roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1]),
        "rf_f1":    f1_score(y_test, rf.predict(X_test)),
        "rf_auc":   roc_auc_score(y_test, y_proba_rf),
    }

    return lgbm, rf, results, y_test, metrics


# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")

st.title("🔧 Predictive Maintenance — Failure Risk Dashboard")
st.markdown(f"Real-time machine failure risk assessment for maintenance engineers. "
            f"[View source on GitHub]({GITHUB_URL})")
st.divider()

with st.spinner("⚙️ Loading data and training models..."):
    lgbm, rf, results, y_test, metrics = load_and_train()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
n_machines = st.sidebar.slider("Number of machines to display", 5, 20, 10)
mix_high   = st.sidebar.slider("High risk machines", 1, 10, 5)
mix_low    = st.sidebar.slider("Medium risk machines", 1, 10, 5)
st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

# ──────────────────────────────────────────────
# Model metrics
# ──────────────────────────────────────────────

st.subheader("📊 Model Performance")
c1, c2, c3, c4 = st.columns(4)
c1.metric("LightGBM F1",    f"{metrics['lgbm_f1']:.4f}")
c2.metric("LightGBM AUC",   f"{metrics['lgbm_auc']:.4f}")
c3.metric("Random Forest F1",  f"{metrics['rf_f1']:.4f}")
c4.metric("Random Forest AUC", f"{metrics['rf_auc']:.4f}")
st.divider()

# ──────────────────────────────────────────────
# Sample machines — high + medium risk mix
# ──────────────────────────────────────────────

st.subheader("🎯 Machine Risk Assessment")

high_risk = results[results["failure_probability"] >= 70]
med_risk  = results[(results["failure_probability"] >= 30) & (results["failure_probability"] < 70)]

n_high = min(mix_high, len(high_risk))
n_med  = min(mix_low,  len(med_risk))

sample = pd.concat([
    high_risk.sample(n_high, random_state=42) if n_high > 0 else pd.DataFrame(),
    med_risk.sample(n_med,   random_state=42) if n_med  > 0 else pd.DataFrame(),
]).sort_values("failure_probability", ascending=True)

labels     = [f"Machine {idx}" for idx in sample.index]
bar_colors = [COLORS_MAP.get(f, "gray") for f in sample["Failure Type Text"]]

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("Failure Probability by Machine",
                                    "Failure Type vs Probability"))

# Chart 1 — horizontal bar
fig.add_trace(go.Bar(
    x=sample["failure_probability"].round(2),
    y=labels,
    orientation="h",
    marker_color=bar_colors,
    text=[f"{p:.2f}% — {f}" for p, f in zip(sample["failure_probability"], sample["Failure Type Text"])],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
), row=1, col=1)

# Chart 2 — scatter by failure type
for ftype, group in sample.groupby("Failure Type Text"):
    idx_positions = [list(sample.index).index(i) for i in group.index]
    fig.add_trace(go.Scatter(
        x=group["failure_probability"].round(2),
        y=[labels[i] for i in idx_positions],
        mode="markers",
        name=ftype,
        marker=dict(color=COLORS_MAP.get(ftype, "gray"), size=14),
        hovertemplate=f"<b>%{{y}}</b><br>Type: {ftype}<br>Probability: %{{x:.2f}}%<extra></extra>",
    ), row=1, col=2)

fig.add_vline(x=70, line_dash="dash", line_color="red",    annotation_text="High (70%)",   row=1, col=2)
fig.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Medium (30%)", row=1, col=2)
fig.update_layout(height=500, showlegend=True, plot_bgcolor="white")
fig.update_xaxes(title_text="Failure Probability (%)", range=[0, 130], row=1, col=1)
fig.update_xaxes(title_text="Failure Probability (%)", range=[0, 120], row=1, col=2)

st.plotly_chart(fig, use_container_width=True)
st.divider()

# ──────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────

st.subheader("📋 Risk Summary Table")
display_cols = ["failure_probability", "risk_level", "Failure Type Text", "Actual Target"]
col_rename   = {"failure_probability": "Probability (%)", "risk_level": "Risk Level",
                "Failure Type Text": "Failure Type", "Actual Target": "Actual Failure"}
st.dataframe(sample[display_cols].rename(columns=col_rename), use_container_width=True)
st.divider()

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.caption(f"Dataset: AI4I 2020 Predictive Maintenance | Models: LightGBM + Random Forest | "
           f"[GitHub]({GITHUB_URL})")


