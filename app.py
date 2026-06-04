"""
app.py
------
Streamlit app for the AI4I 2020 Predictive Maintenance Dataset.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from preprocessing import (
    compute_features, CategoricalEncoder, NumericalScaler,
    CATEGORICAL_COLS, NUMERICAL_COLS, TARGET_COL
)

GITHUB_URL = "https://github.com/mila151107/IPS_solution_predictive_maintenance"
DATA_PATH  = "predictive_maintenance.csv"

COLORS_MAP = {
    "Power Failure":            "#e74c3c",
    "Tool Wear Failure":        "#3498db",
    "Overstrain Failure":       "#f39c12",
    "Heat Dissipation Failure": "#2ecc71",
    "Random Failures":          "#9b59b6",
    "No Failure":               "#95a5a6",
}

@st.cache_resource
def load_and_train():
    df  = pd.read_csv(DATA_PATH)
    df1 = df.copy()
    df  = compute_features(df)

    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[TARGET_COL]
    )

    cols_drop = [TARGET_COL, "UDI", "Process temperature [K]",
                 "Air temperature [K]", "Torque/RPM ratio", "Product ID"]
    X_train = train_df.drop(columns=cols_drop, errors="ignore")
    y_train = train_df[TARGET_COL]
    X_test  = test_df.drop(columns=cols_drop, errors="ignore")
    y_test  = test_df[TARGET_COL]

    encoder = CategoricalEncoder(columns=CATEGORICAL_COLS)
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test  = encoder.transform(X_test)

    scaler  = NumericalScaler(columns=NUMERICAL_COLS)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train.columns = X_train.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()
    X_test.columns  = X_test.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    # MinMax-scaled copies for Logistic Regression
    mm = MinMaxScaler()
    X_train_mm = pd.DataFrame(mm.fit_transform(X_train), columns=X_train.columns)
    X_test_mm  = pd.DataFrame(mm.transform(X_test),      columns=X_test.columns)

    # Fit tree models
    lgbm = LGBMClassifier(n_estimators=200, learning_rate=0.01, num_leaves=10,
                           max_depth=4, min_child_samples=30, class_weight="balanced",
                           importance_type="gain", random_state=42)
    rf   = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   class_weight="balanced", random_state=42)
    lr   = LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced",
                               solver="saga", random_state=42)
    lgbm.fit(X_train, y_train)
    rf.fit(X_train,   y_train)
    lr.fit(X_train_mm, y_train)

    # RF probabilities drive the risk table (kept as before)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf  = (y_proba_rf >= 0.3).astype(int)

    results = X_test.copy()
    results["failure_probability"] = (y_proba_rf * 100).round(2)
    results["risk_level"]          = results["failure_probability"].apply(
        lambda x: "🔴 HIGH" if x >= 70 else "🟡 MEDIUM" if x >= 30 else "🟢 LOW"
    )
    results["predicted_failure"]  = y_pred_rf
    results["Failure Type Text"]  = df1.loc[results.index, "Failure Type"].values
    results["Actual Target"]      = df1.loc[results.index, "Target"].values

    metrics = {
        "LightGBM":            f1_score(y_test, lgbm.predict(X_test)),
        "Random Forest":       f1_score(y_test, rf.predict(X_test)),
        "Logistic Regression": f1_score(y_test, lr.predict(X_test_mm)),
    }

    return lgbm, rf, lr, results, y_test, metrics


# ── Layout ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Predictive Maintenance", page_icon="🔧", layout="wide")
st.title("🔧 Predictive Maintenance — Failure Risk Dashboard")
st.markdown(f"Real-time machine failure risk assessment. [View source on GitHub]({GITHUB_URL})")
st.divider()

with st.spinner("⚙️ Loading data and training models..."):
    lgbm, rf, lr, results, y_test, metrics = load_and_train()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Settings")
n_machines = st.sidebar.slider("Number of machines to display", 5, 20, 10)
mix_high   = st.sidebar.slider("High risk machines",   1, 10, 5)
mix_low    = st.sidebar.slider("Medium risk machines", 1, 10, 5)
st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

# ── Model metrics (F1 only) ───────────────────────────────────────────────────

st.subheader("📊 Model Performance — F1 Score")
cols = st.columns(len(metrics))
for col, (name, f1) in zip(cols, metrics.items()):
    col.metric(f"{name} F1", f"{f1:.4f}")
st.divider()

# ── Risk assessment chart ─────────────────────────────────────────────────────

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

fig.add_trace(go.Bar(
    x=sample["failure_probability"].round(2),
    y=labels, orientation="h",
    marker_color=bar_colors,
    text=[f"{p:.2f}% — {f}" for p, f in
          zip(sample["failure_probability"], sample["Failure Type Text"])],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
), row=1, col=1)

for ftype, group in sample.groupby("Failure Type Text"):
    idx_positions = [list(sample.index).index(i) for i in group.index]
    fig.add_trace(go.Scatter(
        x=group["failure_probability"].round(2),
        y=[labels[i] for i in idx_positions],
        mode="markers", name=ftype,
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

# ── Summary table ─────────────────────────────────────────────────────────────

st.subheader("📋 Risk Summary Table")
display_cols = ["failure_probability", "risk_level", "Failure Type Text", "Actual Target"]
col_rename   = {"failure_probability": "Probability (%)", "risk_level": "Risk Level",
                "Failure Type Text": "Failure Type", "Actual Target": "Actual Failure"}
st.dataframe(sample[display_cols].rename(columns=col_rename), use_container_width=True)
st.divider()

st.caption(f"Dataset:  Predictive Maintenance | Models: LightGBM + Random Forest + Logistic Regression | "
           f"[GitHub]({GITHUB_URL})")