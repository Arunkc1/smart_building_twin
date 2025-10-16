import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF
import io, datetime, os

st.set_page_config(page_title="🌿 AI-Powered IoT Twin — Smart Building", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.alpha": 0.25,
})

# ==============================================================
# 💡 Safe Reset Logic (final version)
# ==============================================================
DEFAULTS = {
    "mode": "⚡ Fast Demo Mode",
    "model_kind": "RandomForest",
    "fi_mode": "🟠 Simulated Dynamic",
    "num_buildings": 3,
    "hour": 10,
    "temperature": 22,
    "humidity": 50,
    "occupancy": 2,
    "ventilation_on": False,
}

# Step 1: detect first load or post-reset
if "initialized" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["initialized"] = True

# Step 2: trigger reset safely (no direct overwrite mid-session)
if st.sidebar.button("🔁 Reset Filters"):
    st.session_state.clear()
    st.session_state["reset_pending"] = True
    st.rerun()

# Step 3: perform reset cleanly on the next run *before* widgets appear
if st.session_state.get("reset_pending", False):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["reset_pending"] = False
    st.toast("✅ Filters reset to default values", icon="🔁")

# ==============================================================
# Helpers
# ==============================================================
FEATURES = ["hour", "temperature", "humidity", "occupancy", "ventilation_on"]

def compute_comfort(temp, hum):
    comfort = 100 - 4 * np.abs(temp - 22) - 0.5 * np.abs(hum - 50)
    return np.clip(comfort, 0, 100)

def compute_smart_score(df):
    energy_norm = (df["appliance_load_kw"] / df["appliance_load_kw"].max()) * 100.0
    co2_norm = (df["co2_ppm"] / 2000.0) * 100.0
    smart = (0.4 * (100 - energy_norm) +
             0.3 * (100 - co2_norm) +
             0.3 * df["comfort_index"])
    return smart.round(2)

def recommendation_rule(row):
    recs = []
    if row["occupancy"] <= 1 and 6 <= row["hour"] <= 17:
        recs.append("Run only essential appliances")
    elif row["appliance_load_kw"] > 2.5:
        recs.append("Reduce appliance usage")
    if row["co2_ppm"] > 1000 and int(row["ventilation_on"]) == 0:
        recs.append("Turn ON ventilation")
    if row["comfort_index"] < 60:
        if row["temperature"] < 20:
            recs.append("Increase heating")
        elif row["temperature"] > 26:
            recs.append("Use air conditioning")
        if row["humidity"] < 40:
            recs.append("Use humidifier")
        elif row["humidity"] > 60:
            recs.append("Use dehumidifier")
    return "; ".join(recs) if recs else "No action needed"

@st.cache_data(show_spinner=False)
def load_default_csv(mode):
    df = pd.read_csv("iot_smart_building_data.csv")
    if mode == "🎓 Research Accuracy Mode":
        st.info("Using full 1-year dataset for research accuracy (may take 10-15 s)")
    else:
        df = df.iloc[:168].copy()  # 7-day subset for fast demo
    if "comfort_index" not in df.columns:
        df["comfort_index"] = compute_comfort(df["temperature"], df["humidity"])
    if "smart_score" not in df.columns:
        df["smart_score"] = compute_smart_score(df)
    if "recommendation" not in df.columns:
        df["recommendation"] = df.apply(recommendation_rule, axis=1)
    return df

@st.cache_resource(show_spinner=False)
def get_models(df, model_kind="RandomForest", mode="⚡ Fast Demo Mode"):
    X = df[FEATURES]
    y_e, y_c, y_t = df["appliance_load_kw"], df["co2_ppm"], df["comfort_index"]
    X_train, X_test, ye_train, ye_test = train_test_split(X, y_e, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(X, y_c, test_size=0.2, random_state=42)
    _, _, yt_train, yt_test = train_test_split(X, y_t, test_size=0.2, random_state=42)

    if model_kind == "Linear":
        M_energy = LinearRegression().fit(X_train, ye_train)
        M_co2 = LinearRegression().fit(X_train, yc_train)
        M_comfort = LinearRegression().fit(X_train, yt_train)
    else:
        if mode == "⚡ Fast Demo Mode":
            M_energy = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train, ye_train)
            M_co2 = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train, yc_train)
            M_comfort = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42).fit(X_train, yt_train)
        else:
            M_energy = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42).fit(X_train, ye_train)
            M_co2 = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42).fit(X_train, yc_train)
            M_comfort = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42).fit(X_train, yt_train)

    metrics = {
        "energy": {"r2": r2_score(ye_test, M_energy.predict(X_test)),
                   "mse": mean_squared_error(ye_test, M_energy.predict(X_test))},
        "co2": {"r2": r2_score(yc_test, M_co2.predict(X_test)),
                "mse": mean_squared_error(yc_test, M_co2.predict(X_test))},
        "comfort": {"r2": r2_score(yt_test, M_comfort.predict(X_test)),
                    "mse": mean_squared_error(yt_test, M_comfort.predict(X_test))}
    }
    return (M_energy, M_co2, M_comfort), metrics

def predict_triplet(models, row):
    M_energy, M_co2, M_comfort = models
    return (float(M_energy.predict(row)[0]),
            float(M_co2.predict(row)[0]),
            float(M_comfort.predict(row)[0]))

# ==============================================================
# Sidebar
# ==============================================================
st.sidebar.header("⚙️ Mode & Model Settings")
mode = st.sidebar.radio("Select Mode:", ["⚡ Fast Demo Mode", "🎓 Research Accuracy Mode"], index=0, key="mode")
model_kind = st.sidebar.selectbox("Model type", ["RandomForest", "Linear"], index=0, key="model_kind")
fi_mode = st.sidebar.radio("Feature Importance Mode:", ["🟠 Simulated Dynamic", "🟢 Retrain Live"], index=0, key="fi_mode")
num_buildings = st.sidebar.slider("Simulate number of buildings", 1, 5, 3, key="num_buildings")

st.sidebar.header("🔧 Input Simulation")
hour = st.sidebar.slider("Hour", 0, 23, 10, key="hour")
temperature = st.sidebar.slider("Temperature (°C)", 15, 30, 22, key="temperature")
humidity = st.sidebar.slider("Humidity (%)", 30, 80, 50, key="humidity")
occupancy = st.sidebar.slider("Occupancy", 0, 10, 2, key="occupancy")
ventilation_on = st.sidebar.checkbox("Ventilation On", value=False, key="ventilation_on")

# -----------------------------
# 📥 Optional: Upload CSV to override default dataset
# -----------------------------
st.sidebar.header("📥 Data Source")

uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"], key="uploader")

REQUIRED_COLS = {
    "hour", "temperature", "humidity", "occupancy", "ventilation_on",
    "appliance_load_kw", "co2_ppm"  # timestamp optional
}

df = None
if uploaded is not None:
    try:
        tmp = pd.read_csv(uploaded)

        # Basic schema checks
        missing = REQUIRED_COLS - set(tmp.columns)
        if missing:
            st.sidebar.error(f"Uploaded CSV is missing columns: {sorted(list(missing))}")
        else:
            # Coerce types safely
            for c in ["hour","temperature","humidity","occupancy","ventilation_on",
                      "appliance_load_kw","co2_ppm"]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

            # Clean rows with required nulls
            tmp = tmp.dropna(subset=["hour","temperature","humidity","occupancy",
                                     "ventilation_on","appliance_load_kw","co2_ppm"])

            # Clip and normalize some fields
            tmp["hour"] = tmp["hour"].clip(0, 23).astype(int)
            tmp["ventilation_on"] = tmp["ventilation_on"].round().clip(0, 1).astype(int)

            # Derive comfort/smart score if not present
            if "comfort_index" not in tmp.columns:
                tmp["comfort_index"] = compute_comfort(tmp["temperature"], tmp["humidity"])
            if "smart_score" not in tmp.columns:
                tmp["smart_score"] = compute_smart_score(tmp)

            # Respect user's file as-is (no downsampling for demo)
            df = tmp.reset_index(drop=True)
            st.sidebar.success("Using uploaded dataset ✅")
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")

# Fallback to default if no valid upload
if df is None:
    if not os.path.exists("iot_smart_building_data.csv"):
        st.error("Missing iot_smart_building_data.csv in this folder.")
        st.stop()
    df = load_default_csv(mode)

# Ensure building_id exists and matches slider
labels = [f"Building_{i+1}" for i in range(num_buildings)]
if "building_id" not in df.columns:
    df["building_id"] = np.random.choice(labels, size=len(df), replace=True)
else:
    df["building_id"] = df["building_id"].where(df["building_id"].isin(labels),
                                                np.random.choice(labels, size=len(df), replace=True))

# Now build the models (always after df is finalized)
input_row = pd.DataFrame([[hour, temperature, humidity, occupancy, int(ventilation_on)]], columns=FEATURES)
models, metrics = get_models(df, model_kind, mode)

# -----------------------------
# Header
# -----------------------------
st.title("🌿 AI-Powered IoT Twin — Smart Sustainable Building")
st.caption("Toggle between ⚡ fast demo or 🎓 research-grade accuracy.")

# -----------------------------
# Live Predictions
# -----------------------------
pred_energy, pred_co2, pred_comfort = predict_triplet(models, input_row)
smart_score_live = float((0.4 * (100 - (pred_energy / df['appliance_load_kw'].max()) * 100)
                          + 0.3 * (100 - (pred_co2 / 2000.0) * 100)
                          + 0.3 * pred_comfort))

c1, c2, c3, c4 = st.columns(4)
c1.metric("💡 Energy (kW)", f"{pred_energy:.2f}")
c2.metric("🫧 CO₂ (ppm)", f"{pred_co2:.0f}")
c3.metric("🛋️ Comfort (/100)", f"{pred_comfort:.1f}")
c4.metric("🌟 Smart Score", f"{smart_score_live:.1f}")

baseline_energy = df["appliance_load_kw"].max()
data_co2_factor = 0.92
avg_carbon_savings = (baseline_energy - pred_energy) * data_co2_factor
st.metric("🌎 Carbon Savings", f"{avg_carbon_savings:.2f} kg/day")

st.success(recommendation_rule({
    "occupancy": occupancy, "hour": hour,
    "appliance_load_kw": pred_energy, "co2_ppm": pred_co2,
    "ventilation_on": int(ventilation_on),
    "comfort_index": pred_comfort,
    "temperature": temperature, "humidity": humidity
}))

# -----------------------------
# Dynamic Tabs (Responsive to Sliders)
# -----------------------------
st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    f"📈 Time Series — Hour: {hour}",
    "📅 Hourly Averages",
    "🧩 Correlations",
    "🌲 Feature Importance",
    "✅ Recommendations"
])

# Copy and dynamically adjust dataset based on sliders
df_dynamic = df.copy()

# Filter data around the selected hour (±2 hours for context)
df_dynamic = df_dynamic[df_dynamic["hour"].between(max(0, hour-2), min(23, hour+2))]
if df_dynamic.empty:
    st.info("No rows in the selected hour window — expanding to full dataset.")
    df_dynamic = df.copy()

# Apply slider-based environmental shifts
df_dynamic["temperature"] += (temperature - df["temperature"].mean()) * 0.2
df_dynamic["humidity"] += (humidity - df["humidity"].mean()) * 0.2
df_dynamic["occupancy"] = np.clip(df_dynamic["occupancy"] + (occupancy - 2), 0, 10)
df_dynamic["ventilation_on"] = int(ventilation_on)

# Recompute comfort & smart score
df_dynamic["comfort_index"] = compute_comfort(df_dynamic["temperature"], df_dynamic["humidity"])
df_dynamic["smart_score"] = compute_smart_score(df_dynamic)

# Predict using trained models
df_dynamic["pred_energy"] = models[0].predict(df_dynamic[FEATURES])
df_dynamic["pred_co2"] = models[1].predict(df_dynamic[FEATURES])
df_dynamic["pred_comfort"] = models[2].predict(df_dynamic[FEATURES])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(df_dynamic.index, df_dynamic["pred_energy"], label="Energy (kW)")
    ax1.plot(df_dynamic.index, df_dynamic["pred_co2"], label="CO₂ (ppm)")
    ax1.plot(df_dynamic.index, df_dynamic["pred_comfort"], label="Comfort")
    ax1.set_title("Dynamic Predictions over Time")
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    hourly_grp = df_dynamic.groupby("hour").mean(numeric_only=True)
    fig2, ax2 = plt.subplots(figsize=(10,4))
    hourly_grp[["pred_energy","pred_co2","pred_comfort"]].plot(ax=ax2)
    ax2.set_title("Hourly Averages (Dynamic)")
    ax2.set_xlabel("Hour")
    st.pyplot(fig2)

with tab3:
    cols = ["hour","temperature","humidity","occupancy","ventilation_on",
            "pred_energy","pred_co2","pred_comfort","smart_score"]
    corr = df_dynamic[cols].corr()
    fig3, ax3 = plt.subplots(figsize=(8,6))
    im = ax3.imshow(corr, cmap="coolwarm", aspect="auto")
    fig3.colorbar(im)
    ax3.set_xticks(range(len(cols))); ax3.set_yticks(range(len(cols)))
    ax3.set_xticklabels(cols, rotation=45, ha="right"); ax3.set_yticklabels(cols)
    ax3.set_title("Correlation Matrix (Dynamic)")
    st.pyplot(fig3)

with tab4:
    st.write("🌲 Feature Importance (adjustable)")

    if fi_mode == "🟠 Simulated Dynamic":
        # Base importances from trained model
        if model_kind == "Linear":
            importances = pd.Series(np.abs(models[0].coef_), index=FEATURES)
        else:
            importances = pd.Series(models[0].feature_importances_, index=FEATURES)

        # 🔸 Stronger visual scaling based on how far sliders moved
        slider_effect = 1 + 0.5 * (
            abs(temperature - df["temperature"].mean())/5 +
            abs(humidity - df["humidity"].mean())/10 +
            abs(occupancy - df["occupancy"].mean())/3
        )

        dynamic_importances = (importances * slider_effect).sort_values(ascending=True)

        # 🔸 Animate color intensity
        color_intensity = min(1.0, 0.5 + 0.05 * abs(temperature - df["temperature"].mean()))
        bar_color = (1, color_intensity, 0)  # RGB for orange-to-yellow gradient

        fig4, ax4 = plt.subplots(figsize=(8, 4))
        dynamic_importances.plot(kind="barh", ax=ax4, color=bar_color)
        ax4.set_title(f"Feature Importance ({model_kind}) – Simulated Dynamic")
        ax4.set_xlabel("Relative Importance")
        ax4.set_xlim(0, dynamic_importances.max() * 1.3)  # force axis rescale
        ax4.grid(alpha=0.3)
        st.pyplot(fig4)

    else:
        # 🟢 Retrain mode (true recomputation)
        @st.cache_resource
        def retrain_models_live(df_dynamic, model_kind, mode):
            return get_models(df_dynamic, model_kind, mode)

        with st.spinner("Re-training models for live feature importance..."):
            models_live, _ = retrain_models_live(df_dynamic, model_kind, mode)

            if model_kind == "Linear":
                importances = pd.Series(np.abs(models_live[0].coef_), index=FEATURES)
            else:
                importances = pd.Series(models_live[0].feature_importances_, index=FEATURES)

            importances = importances.sort_values(ascending=True)
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            importances.plot(kind="barh", ax=ax4, color="tab:green")
            ax4.set_title(f"Feature Importance ({model_kind}) – Retrain Live")
            ax4.set_xlabel("True Importance (after retraining)")
            ax4.grid(alpha=0.3)
            st.pyplot(fig4)

with tab5:
    st.write("📋 Updated Recommendations based on new conditions:")
    df_dynamic["recommendation"] = df_dynamic.apply(recommendation_rule, axis=1)
    st.dataframe(df_dynamic[["hour","temperature","humidity","occupancy",
                             "pred_energy","pred_co2","pred_comfort","smart_score","recommendation"]],
                 use_container_width=True)

# -----------------------------
# Scenario Simulator
# -----------------------------
st.divider()
st.subheader("🎛️ Scenario Simulator — What-If Analysis")
var = st.selectbox("Variable to vary:", ["Temperature", "Occupancy", "Humidity"])
if var == "Temperature":
    vals = np.linspace(15, 30, 20)
    df_sim = pd.DataFrame({"hour": [hour]*len(vals), "temperature": vals,
                           "humidity": [humidity]*len(vals),
                           "occupancy": [occupancy]*len(vals),
                           "ventilation_on": [int(ventilation_on)]*len(vals)})
elif var == "Occupancy":
    vals = np.arange(0, 11)
    df_sim = pd.DataFrame({"hour": [hour]*len(vals), "temperature": [temperature]*len(vals),
                           "humidity": [humidity]*len(vals), "occupancy": vals,
                           "ventilation_on": [int(ventilation_on)]*len(vals)})
else:
    vals = np.linspace(30, 80, 20)
    df_sim = pd.DataFrame({"hour": [hour]*len(vals), "temperature": [temperature]*len(vals),
                           "humidity": vals, "occupancy": [occupancy]*len(vals),
                           "ventilation_on": [int(ventilation_on)]*len(vals)})

pred_e = models[0].predict(df_sim)
pred_c = models[1].predict(df_sim)
pred_t = models[2].predict(df_sim)
smart_sim = 0.4*(100 - (pred_e/df["appliance_load_kw"].max())*100) + 0.3*(100 - (pred_c/2000)*100) + 0.3*pred_t
fig_sim, ax_sim = plt.subplots(figsize=(8,4))
ax_sim.plot(vals, smart_sim, color='teal', label='Smart Score')
ax_sim.set_xlabel(var); ax_sim.set_ylabel("Smart Score")
ax_sim.set_title(f"Smart Score vs {var}"); ax_sim.legend()
st.pyplot(fig_sim)

# -----------------------------
# Multi-Building Summary
# -----------------------------
st.subheader("🏢 Multi-Building Summary")

# Guard: create building_id if missing (extra safety)
if "building_id" not in df.columns:
    labels = [f"Building_{i+1}" for i in range(num_buildings)]
    df["building_id"] = np.random.choice(labels, size=len(df), replace=True)

summary = df.groupby("building_id").agg({
    "appliance_load_kw": "mean",
    "co2_ppm": "mean",
    "comfort_index": "mean",
    "smart_score": "mean"
}).round(2).reset_index()

st.dataframe(summary, use_container_width=True)

# -----------------------------
# PDF Report Generator
# -----------------------------
st.divider()
st.subheader("🧾 Export Performance Report")

def create_report(df, metrics):
    buf = io.BytesIO()
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, "AI-Powered IoT Twin – Smart Building Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.multi_cell(0, 8, f"Mode: {mode}")
    pdf.multi_cell(0, 8, f"Model: {model_kind}")
    pdf.multi_cell(0, 8, f"R² – Energy: {metrics['energy']['r2']:.3f}, CO₂: {metrics['co2']['r2']:.3f}, Comfort: {metrics['comfort']['r2']:.3f}")
    pdf.multi_cell(0, 8, f"MSE – Energy: {metrics['energy']['mse']:.2f}, CO₂: {metrics['co2']['mse']:.2f}, Comfort: {metrics['comfort']['mse']:.2f}")
    pdf.multi_cell(0, 8, f"Avg Smart Score: {df['smart_score'].mean():.2f}")
    pdf.output(buf)
    return buf.getvalue()

if st.button("📄 Generate PDF Report"):
    pdf_data = create_report(df, metrics)
    st.download_button("⬇️ Download Report", data=pdf_data,
                       file_name="smart_building_report.pdf", mime="application/pdf")
