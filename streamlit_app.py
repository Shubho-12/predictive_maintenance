# app/streamlit_app.py
import sys
from pathlib import Path

# ---- Ensure project root is on sys.path so 'src' can be imported ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- Now normal imports ----
import streamlit as st
import pandas as pd
from time import sleep

try:
    from src.feature_engineering import create_rolling_features
    from src.inference import predict_from_list, _ordered_feature_names
except Exception as e:
    st.error("Import error: could not import project modules from 'src'.\n"
             "Make sure src/__init__.py exists or run Streamlit from project root.\n\n"
             f"Details: {e}")
    raise

ROOT = PROJECT_ROOT
RAW = ROOT / "data" / "raw" / "simulated_sensor_data.csv"
PROCESSED = ROOT / "data" / "processed" / "processed_features.csv"

st.set_page_config(page_title="Predictive Maintenance Demo", layout="wide")
st.title("🔧 Predictive Maintenance — Demo")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Sensor Data (raw)")
    if RAW.exists():
        raw_df = pd.read_csv(RAW)
        machine = st.selectbox("Select machine_id", options=sorted(raw_df['machine_id'].unique()))
        rows = st.slider("Show last N rows", 10, 500, 50)
        st.dataframe(raw_df[raw_df['machine_id'] == machine].tail(rows))
    else:
        st.warning("No raw sensor CSV found. Run the data generator.")

with col2:
    st.subheader("Controls")
    run_mode = st.radio("Mode", ("Snapshot", "Simulate stream"))
    window = st.number_input("Feature window size", min_value=10, max_value=500, value=50, step=10)
    horizon = st.number_input("Failure horizon (future rows)", min_value=1, max_value=100, value=10)
    threshold = st.slider("Alert probability threshold", 0.0, 1.0, 0.7)

st.markdown("---")

# Prepare processed features on demand
if st.button("(Re)build features & preprocess"):
    if RAW.exists():
        feats = create_rolling_features(pd.read_csv(RAW), window=window, horizon=horizon)
        PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        feats.to_csv(PROCESSED, index=False)
        st.success("✅ Features rebuilt and saved to data/processed/processed_features.csv")
    else:
        st.error("Raw data missing.")

# Snapshot mode: show predictions for most recent processed features
if run_mode == "Snapshot":
    if PROCESSED.exists():
        df = pd.read_csv(PROCESSED)
        st.subheader("Latest feature rows and predictions")
        preview = df.groupby('machine_id').tail(1).reset_index(drop=True)
        names = _ordered_feature_names()
        preds = []
        for _, row in preview.iterrows():
            values = row[names].values.tolist()
            pred, prob = predict_from_list(values)
            preds.append((pred, prob))
        preview['pred'] = [p for p, _ in preds]
        preview['prob'] = [round(q, 3) if q is not None else None for _, q in preds]
        display_cols = ['machine_id', 'end_time'] + names + ['pred', 'prob']
        # protect against missing columns
        display_cols = [c for c in display_cols if c in preview.columns]
        st.dataframe(preview[display_cols].sort_values('machine_id'))
        st.write("Machines with alerts (prob >=", threshold, "):")
        alerts = preview[preview['prob'] >= threshold]
        if not alerts.empty:
            st.dataframe(alerts[['machine_id', 'end_time', 'prob', 'pred']])
        else:
            st.info("No alerts at this threshold.")
    else:
        st.info("No processed features. Click '(Re)build features & preprocess' to create them.")

# Simulate stream mode: iterate last N rows and show live predictions
if run_mode == "Simulate stream":
    if RAW.exists():
        st.subheader("Simulated streaming (latest data → window → prediction)")
        raw_df = pd.read_csv(RAW)  # ensure available in this branch
        machine = st.selectbox("Machine to simulate", options=sorted(raw_df['machine_id'].unique()), key="sim_machine")
        df_machine = raw_df[raw_df['machine_id'] == machine].reset_index(drop=True)
        latest_n = st.number_input("Simulate using last N records from raw", min_value=50, max_value=2000, value=300, step=10)
        # ensure we have enough rows
        max_i = min(len(df_machine), latest_n)
        if max_i <= window:
            st.warning(f"Not enough rows ({len(df_machine)}) for the selected window ({window}). Reduce window or regenerate data.")
        else:
            progress = st.empty()
            metric = st.empty()
            for i in range(window, max_i, 5):
                win = df_machine.iloc[i - window:i]
                # build one feature row using same aggregations
                features = []
                for col in ['temperature', 'vibration', 'pressure', 'rpm']:
                    arr = win[col].values.astype(float)
                    features += [arr.mean(), arr.std(), arr.min(), arr.max(), ((arr[-1] - arr[0]) / (len(arr) - 1 + 1e-9))]
                pred, prob = predict_from_list(features)
                status_text = "FAIL" if pred == 1 else "OK"
                metric.metric(label=f"Machine {machine} — iter {i}", value=status_text, delta=f"Prob: {round(prob,3) if prob is not None else 'NA'}")
                if prob is not None and prob >= threshold:
                    st.warning(f"Alert! Machine {machine} failure probability {prob:.2f} >= {threshold}")
                progress.text(f"Simulated record {i}/{max_i}")
                sleep(0.35)
            st.success("Simulation finished.")
    else:
        st.error("No raw data found. Run data generator first.")
