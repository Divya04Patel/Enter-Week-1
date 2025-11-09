import streamlit as st
import pandas as pd
from pathlib import Path
from functools import lru_cache
import joblib
import pandas as pd
import numpy as np
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_model = None
_pre = None

def _load_artifacts():
    global _model, _pre
    if _model is None:
        mfp = MODELS_DIR / "ev_range_model.pkl"
        if mfp.exists():
            _model = joblib.load(mfp)
    if _pre is None:
        pfp = MODELS_DIR / "preprocessor.pkl"
        if pfp.exists():
            _pre = joblib.load(pfp)

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Battery_Capacity_kWh' in df.columns:
        df['battery_Wh'] = df['Battery_Capacity_kWh'] * 1000.0
    if 'battery_Wh' in df.columns and 'Efficiency_WhPerKm' in df.columns:
        df['battery_over_eff'] = df['battery_Wh'] / df['Efficiency_WhPerKm'].replace(0, np.nan)
    if 'Battery_Capacity_kWh' in df.columns and 'Weight_kg' in df.columns:
        df['energy_density_kWh_per_kg'] = df['Battery_Capacity_kWh'] / df['Weight_kg'].replace(0, np.nan)
    return df

def _prepare_model_input(df: pd.DataFrame):
    """Return numpy array ready for model.predict — applies preprocessor if present."""
    _load_artifacts()
    df = _add_derived_features(df.copy())

    if _pre is not None:
        numeric_cols = _pre.get("numeric_cols", [])
        missing = [c for c in numeric_cols if c not in df.columns]
        if missing:
            raise ValueError(f"The model expects these feature columns; they are missing: {missing}")
        Xnum = df[numeric_cols].astype(float).copy()
        Xnum.iloc[:, :] = _pre["imputer"].transform(Xnum)
        Xnum.iloc[:, :] = _pre["scaler"].transform(Xnum)
        X = Xnum.values
    else:
        X = df.select_dtypes(include=[np.number]).values

    if _model is not None and hasattr(_model, "n_features_in_"):
        if X.shape[1] != _model.n_features_in_:
            raise ValueError(f"Model expects { _model.n_features_in_ } features, got { X.shape[1] }.")
    return X

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
            X_df = pd.DataFrame([{
                "Battery_Capacity_kWh": battery,
                "Power_hp": power,
                "Efficiency_WhPerKm": efficiency,
                "Weight_kg": weight
            }])
            st.session_state['input_X'] = X_df.to_dict(orient="records")[0]

            # Prefer the preloaded artifacts code path (keeps error handling consistent)
            try:
                _load_artifacts()
                if _model is None:
                    raise FileNotFoundError("No trained model found in models/")
                X_in = _prepare_model_input(X_df)
                pred = _model.predict(X_in)[0]
                st.success(f"Predicted range: {pred:.0f} km (model)")
            except FileNotFoundError:
                fallback = battery * 1000.0 / float(efficiency)
                st.info("No trained model found — using heuristic fallback:")
                st.success(f"Estimated range: {fallback:.0f} km (heuristic)")
                st.write("Heuristic = battery_kWh * 1000 / efficiency_WhPerKm")
            except Exception as e:
                st.error(f"Model inference failed: {e}")

        with st.expander("Show input features"):
            st.json(st.session_state.get('input_X', {}))

    with cols[2]:
        st.markdown("### 📈 Quick Stats")
        st.markdown("- Energy Efficiency: **91%**")
        st.markdown("- Charging Infrastructure: **82%** coverage")
        st.markdown("- Top Efficient Models: **Model 3, Kona, Leaf**")
        st.markdown("- Avg User Range: **412 km**")