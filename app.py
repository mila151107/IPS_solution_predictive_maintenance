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

    # Keep Product ID and Failure Type — model was trained with them
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
threshold = st.sidebar.slider("Failure prediction threshold",
                               min_value=0.10, max_value=0.50,
                               value=0.25, step=0.05,
                               help="Lower = catch more failures. Higher = fewer false alarms.")
mix_high  = st.sidebar.slider("High risk machines",   1, 10, 5)
mix_low   = st.sidebar.slider("Medium risk machines", 1, 10, 5)
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
# Metrics row
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
    text=[f"{p:.2f}% — {f}" for p, f in
          zip(sample["failure_probability"], sample["Failure Type Text"])],
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

fig.add_vline(x=70, line_dash="dash", line_color="red",
              annotation_text="High (70%)",   row=1, col=2)
fig.add_vline(x=30, line_dash="dash", line_color="orange",
              annotation_text="Medium (30%)", row=1, col=2)
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
                "Failure Type Text":   "Failure Type",    "Actual Target": "Actual Failure"}
st.dataframe(sample[display_cols].rename(columns=col_rename), use_container_width=True)
st.divider()

# ──────────────────────────────────────────────
# Single machine prediction
# ──────────────────────────────────────────────

st.subheader("🔍 Predict a Single Machine")
st.markdown("Adjust the sensor values and click **Predict** to check if the machine will fail.")

c1, c2, c3 = st.columns(3)
with c1:
    rpm       = st.number_input("Rotational speed [rpm]", min_value=1000, max_value=3000, value=1500)
    torque    = st.number_input("Torque [Nm]", min_value=1.0, max_value=100.0, value=40.0)
with c2:
    tool_wear = st.number_input("Tool wear [min]", min_value=0, max_value=300, value=100)
    prod_type = st.selectbox("Product type", ["L", "M", "H"])
with c3:
    air_temp  = st.number_input("Air temperature [K]", min_value=295.0, max_value=305.0, value=300.0)
    proc_temp = st.number_input("Process temperature [K]", min_value=305.0, max_value=315.0, value=310.0)

if st.button("🔮 Predict", use_container_width=True):
    input_raw = pd.DataFrame([{
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

    input_eng = compute_features(input_raw)
    cols_drop = [TARGET_COL, "UDI", "Process temperature [K]",
                 "Air temperature [K]", "Torque/RPM ratio"]
    X_input = input_eng.drop(columns=cols_drop, errors="ignore")

    encoder = CategoricalEncoder(columns=CATEGORICAL_COLS)
    X_input = encoder.fit_transform(X_input)
    scaler  = NumericalScaler(columns=NUMERICAL_COLS)
    X_input = scaler.fit_transform(X_input)
    X_input.columns = X_input.columns.str.replace(r"[\[\]/]", "_", regex=True).str.strip()

    pred  = models["Random Forest"].predict(X_input)[0]
    proba = models["Random Forest"].predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"🔴 **Failure predicted** — {proba*100:.1f}% probability of failure")
    else:
        st.success(f"🟢 **No failure predicted** — {proba*100:.1f}% probability of failure")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(f"Dataset: AI4I 2020 Predictive Maintenance | "
           f"Dashboard: Random Forest | Threshold: {threshold} | "
           f"[GitHub]({GITHUB_URL})")