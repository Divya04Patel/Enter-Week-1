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
    st.markdown('<div class="hero-card"><h1>⚡ EV Vehicle Range Predictor 🚗</h1><p>Estimate driving range instantly — adjust battery, power, efficiency and weight.</p></div>', unsafe_allow_html=True)

    cols = st.columns([3, 4, 3])
    with cols[0]:
        st.markdown("### 🔎 EV Insights")
        st.write("- Typical Battery Capacity: 40–75 kWh")
        st.write("- Average Driving Range: 300–500 km")
        st.write("- Efficiency improves at moderate speeds")
        st.markdown("### 💡 Smart Driving Tip")
        st.info("Use regenerative braking effectively in urban driving to increase range.")

    with cols[1]:
        st.markdown("### 🧩 Input Parameters")
        battery = st.slider("Battery capacity (kWh)", min_value=10, max_value=120, value=50, step=1)
        power = st.slider("Motor power (hp)", min_value=20, max_value=500, value=150, step=1)
        efficiency = st.slider("Efficiency (Wh/km)", min_value=80, max_value=300, value=160, step=1)
        weight = st.slider("Vehicle weight (kg)", min_value=800, max_value=3000, value=1600, step=10)
        st.markdown("")

        if 'input_X' not in st.session_state:
            st.session_state['input_X'] = None

        if st.button("🚀 Predict Range", key="predict_button"):
            X = pd.DataFrame([{
                "Battery_Capacity_kWh": battery,
                "Power_hp": power,
                "Efficiency_WhPerKm": efficiency,
                "Weight_kg": weight
            }])
            st.session_state['input_X'] = X.to_dict(orient="records")[0]

            model_path = Path(__file__).resolve().parents[2] / "models" / "ev_range_model.pkl"
            model = load_model(str(model_path))

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

        with st.expander("Show input features"):
            st.json(st.session_state.get('input_X', {}))

    with cols[2]:
        st.markdown("### 📈 Quick Stats")
        st.markdown("- Energy Efficiency: **91%**")
        st.markdown("- Charging Infrastructure: **82%** coverage")
        st.markdown("- Top Efficient Models: **Model 3, Kona, Leaf**")
        st.markdown("- Avg User Range: **412 km**")