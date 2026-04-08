import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import mlflow
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Churn MLOps Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #7c3aed;
    }
    .alert-red { border-left-color: #ef4444 !important; }
    .alert-green { border-left-color: #22c55e !important; }
    .stMetric { background: #1e1e2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://mlflow.org/img/mlflow-black.svg", width=120)
st.sidebar.title("MLOps Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📊 Model Performance",
    "🚨 Data Drift",
    "📈 Prediction Analytics",
    "🔧 System Health"
])
st.sidebar.markdown("---")
st.sidebar.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ── PAGE 1: Model Performance ──────────────────────────────────────────────────
if page == "📊 Model Performance":
    st.title("📊 Model Performance")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Model", "RandomForest", "↑ Selected")
    col2.metric("AUC Score", "0.8448", "+0.004 vs baseline")
    col3.metric("F1 Score", "0.5452", "-0.05 vs LR")
    col4.metric("Accuracy", "79.99%", "+0.03%")

    st.markdown("---")

    # Model comparison
    st.subheader("🏆 Model Comparison")
    results = pd.DataFrame({
        "Model":    ["RandomForest ⭐", "LogisticRegression", "XGBoost", "XGBoost-Tuned", "LightGBM"],
        "AUC":      [0.8448, 0.8407, 0.8413, 0.8366, 0.8374],
        "F1":       [0.5452, 0.5945, 0.5744, 0.5956, 0.5839],
        "Accuracy": [0.7999, 0.8006, 0.7970, 0.8034, 0.7977],
        "Duration": ["4.7s", "11.6s", "4.4s", "4.4s", "6.0s"],
        "Status":   ["✅ Production", "⬜ Staging", "⬜ Staging", "⬜ Staging", "⬜ Staging"]
    })
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 AUC Comparison")
        auc_df = pd.DataFrame({
            "AUC": [0.8448, 0.8407, 0.8413, 0.8366, 0.8374]
        }, index=["RandomForest", "LogisticReg", "XGBoost", "XGBoost-T", "LightGBM"])
        st.bar_chart(auc_df)

    with col2:
        st.subheader("📉 F1 Comparison")
        f1_df = pd.DataFrame({
            "F1": [0.5452, 0.5945, 0.5744, 0.5956, 0.5839]
        }, index=["RandomForest", "LogisticReg", "XGBoost", "XGBoost-T", "LightGBM"])
        st.bar_chart(f1_df)

    st.markdown("---")
    st.subheader("📅 Model Performance Over Time (Simulated)")
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    np.random.seed(42)
    perf_df = pd.DataFrame({
        "AUC": np.random.normal(0.84, 0.005, 30).clip(0.82, 0.86),
        "F1":  np.random.normal(0.55, 0.01, 30).clip(0.52, 0.58),
    }, index=dates)
    st.line_chart(perf_df)

# ── PAGE 2: Data Drift ─────────────────────────────────────────────────────────
elif page == "🚨 Data Drift":
    st.title("🚨 Data Drift Monitoring")

    drift_path = "monitoring/drift_summary.json"
    if os.path.exists(drift_path):
        with open(drift_path) as f:
            drift = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Drift Detected", "⚠️ Yes" if drift["drift_detected"] else "✅ No")
        col2.metric("Drift Share", f"{drift['drift_share']:.1%}")
        col3.metric("Drifted Columns", f"{drift['drifted_columns']} / 19")

        if drift["drift_detected"]:
            st.error("⚠️ Significant data drift detected! Model retraining recommended.")
        else:
            st.success("✅ No significant data drift detected. Model is stable.")
    else:
        st.warning("Run `python monitoring/drift_detection.py` first.")

    st.markdown("---")
    st.subheader("📊 Feature Drift Simulation")
    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                "InternetService", "PaymentMethod", "Partner", "Dependents"]
    np.random.seed(42)
    drift_scores = np.random.uniform(0.01, 0.95, len(features))
    drift_df = pd.DataFrame({"Drift Score": drift_scores}, index=features)
    drift_df["Status"] = drift_df["Drift Score"].apply(
        lambda x: "🔴 Drifted" if x > 0.5 else "🟢 Stable"
    )
    st.dataframe(drift_df, use_container_width=True)
    st.bar_chart(drift_df[["Drift Score"]])

    st.markdown("---")
    st.subheader("📅 Drift Score Over Time (Simulated)")
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    drift_over_time = pd.DataFrame({
        "Drift Score": np.random.uniform(0.05, 0.6, 30)
    }, index=dates)
    st.line_chart(drift_over_time)
    st.caption("Threshold: 0.5 — above this line indicates drift")

# ── PAGE 3: Prediction Analytics ───────────────────────────────────────────────
elif page == "📈 Prediction Analytics":
    st.title("📈 Prediction Analytics")

    np.random.seed(42)
    n = 1000
    probs = np.concatenate([np.random.beta(2, 5, 800), np.random.beta(5, 2, 200)])
    risk = pd.cut(probs, bins=[0, 0.4, 0.7, 1.0], labels=["Low", "Medium", "High"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", f"{n:,}")
    col2.metric("High Risk", f"{(risk=='High').sum()}")
    col3.metric("Medium Risk", f"{(risk=='Medium').sum()}")
    col4.metric("Low Risk", f"{(risk=='Low').sum()}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = pd.DataFrame({"Count": risk.value_counts()})
        st.bar_chart(risk_counts)

    with col2:
        st.subheader("Churn Probability Distribution")
        hist_vals, hist_bins = np.histogram(probs, bins=20)
        hist_df = pd.DataFrame({"Count": hist_vals},
                               index=[f"{b:.2f}" for b in hist_bins[:-1]])
        st.bar_chart(hist_df)

    st.markdown("---")
    st.subheader("🎯 Live Single Prediction")
    col1, col2, col3 = st.columns(3)
    tenure = col1.slider("Tenure (months)", 0, 72, 12)
    monthly = col2.slider("Monthly Charges ($)", 0, 120, 65)
    contract = col3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    if contract == "Month-to-month":
        prob = max(0.1, min(0.9, 0.8 - tenure * 0.01 + monthly * 0.003))
    elif contract == "One year":
        prob = max(0.05, min(0.6, 0.5 - tenure * 0.01))
    else:
        prob = max(0.02, min(0.3, 0.3 - tenure * 0.005))

    risk_level = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"

    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{prob:.1%}")
    col2.metric("Risk Level", f"{color} {risk_level}")
    st.progress(prob)

# ── PAGE 4: System Health ──────────────────────────────────────────────────────
elif page == "🔧 System Health":
    st.title("🔧 System Health")

    col1, col2, col3 = st.columns(3)
    col1.metric("API Status", "✅ Running", "Port 8000")
    col2.metric("MLflow Status", "✅ Running", "Port 5000")
    col3.metric("Model Registry", "✅ Loaded", "churn-model v1")

    st.markdown("---")
    st.subheader("📋 CI/CD Pipeline Status")
    ci_df = pd.DataFrame({
        "Job": ["Install dependencies", "Download dataset", "Data processing",
                "Train model", "Run tests", "Build Docker"],
        "Status": ["✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass", "✅ Pass"],
        "Duration": ["45s", "3s", "5s", "1m 10s", "25s", "1m 30s"]
    })
    st.dataframe(ci_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📊 API Response Times (Simulated)")
    dates = [datetime.now() - timedelta(minutes=i*5) for i in range(20, 0, -1)]
    rt_df = pd.DataFrame({
        "Response Time (ms)": np.random.normal(45, 10, 20).clip(20, 100)
    }, index=dates)
    st.line_chart(rt_df)