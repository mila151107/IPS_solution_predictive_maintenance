st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Failure prediction threshold",
                               min_value=0.10, max_value=0.50,
                               value=0.25, step=0.05,
                               help="Lower = catch more failures. Higher = fewer false alarms.")
mix_high   = st.sidebar.slider("High risk machines",   1, 10, 3)
mix_medium = st.sidebar.slider("Medium risk machines", 1, 10, 3)
mix_low    = st.sidebar.slider("Low risk machines",    1, 10, 3)
st.sidebar.divider()
st.sidebar.markdown(f"🔗 [GitHub Repository]({GITHUB_URL})")

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

labels     = [f"Machine {idx}" for idx in sample.index]
bar_colors = [COLORS_MAP.get(f, "gray") for f in sample["Failure Type Text"]]

# Risk band background colors
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
    orientation="h",
    marker_color=bar_colors,
    text=[f"{p:.1f}% — {r}" for p, r in
          zip(sample["failure_probability"], sample["risk_level"])],
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
), row=1, col=1)

# Chart 2 — scatter colored by failure type
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