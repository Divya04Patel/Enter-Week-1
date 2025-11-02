import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from functools import lru_cache

@lru_cache(maxsize=1)
def load_model(path: str):
    p = Path(path)
    if p.exists():
        try:
            return joblib.load(str(p))
        except Exception:
            return None
    return None

def predict_page():
    st.title("🔮 EV Range Predictor")
    st.write("Enter vehicle specs and get a predicted driving range. If no trained model is available, a simple physics-based fallback is used.")

    col1, col2 = st.columns(2)
    with col1:
        battery = st.slider("Battery capacity (kWh)", min_value=10, max_value=120, value=50, step=1)
        power = st.slider("Motor power (hp)", min_value=20, max_value=500, value=150, step=1)
    with col2:
        efficiency = st.slider("Efficiency (Wh/km)", min_value=80, max_value=300, value=160, step=1)
        weight = st.slider("Vehicle weight (kg)", min_value=800, max_value=3000, value=1600, step=10)

    st.markdown("**Notes:** Efficiency is energy consumed per km (Wh/km). Higher efficiency → lower range.")

    model_path = Path(__file__).resolve().parents[2] / "models" / "ev_range_model.pkl"
    model = load_model(str(model_path))

    if st.button("Predict range"):
        X = pd.DataFrame([{
            "Battery_Capacity_kWh": battery,
            "Power_hp": power,
            "Efficiency_WhPerKm": efficiency,
            "Weight_kg": weight
        }])

        if model is not None:
            try:
                pred = model.predict(X)[0]
                st.success(f"Predicted range: {pred:.0f} km (model)")
            except Exception as e:
                st.error(f"Model inference failed: {e}")
        else:
            fallback = battery * 1000.0 / float(efficiency)
            st.info("No trained model found — using heuristic fallback:")
            st.success(f"Estimated range: {fallback:.0f} km (heuristic)")
            st.write("Heuristic = battery_kWh * 1000 / efficiency_WhPerKm")

    with st.expander("Advanced: show input features"):
        st.json(X.to_dict(orient="records")[0])