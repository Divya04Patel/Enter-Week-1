# =====================================
# 📊 Electric Vehicle Range Prediction
# Step 3: EDA + Data Preprocessing
# =====================================

# ✅ Import libraries
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

def _find_column(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean uploaded EV dataframe and return cleaned DataFrame.
    - Normalize common column names
    - Convert to numeric, drop rows with missing values in selected columns
    - Drop duplicates
    """
    df = df.copy()
    mapping = {}
    battery_col = _find_column(df, ['Battery', 'Battery_Capacity_kWh', 'battery_kwh', 'battery_kwh'])
    power_col = _find_column(df, ['Power', 'Power_hp', 'Power_kW', 'Motor_Power'])
    efficiency_col = _find_column(df, ['Efficiency', 'Efficiency_WhPerKm', 'Energy_Consumption', 'consumption'])
    weight_col = _find_column(df, ['Weight', 'Weight_kg', 'Vehicle_Weight'])
    range_col = _find_column(df, ['Range', 'Range_km', 'range_km', 'range'])

    if battery_col:
        mapping[battery_col] = 'Battery_Capacity_kWh'
    if power_col:
        mapping[power_col] = 'Power_hp'
    if efficiency_col:
        mapping[efficiency_col] = 'Efficiency_WhPerKm'
    if weight_col:
        mapping[weight_col] = 'Weight_kg'
    if range_col:
        mapping[range_col] = 'Range_km'

    if mapping:
        df = df.rename(columns=mapping)

    # keep only the expected numeric columns that exist
    wanted = [c for c in ['Battery_Capacity_kWh', 'Power_hp', 'Efficiency_WhPerKm', 'Weight_kg', 'Range_km'] if c in df.columns]
    if not wanted:
        # return copy so caller can inspect available columns
        return df.copy()

    df = df[wanted].copy()

    # convert to numeric and drop rows with NaNs
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# Optional: keep previous script behavior behind a guard so imports are safe
if __name__ == "__main__":
    # Local test / debug behavior only (won't run on import)
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns

    base_dir = Path(__file__).resolve().parent.parent
    sample = base_dir / "data" / "ev_dataset.csv"
    if not sample.exists():
        print("No sample dataset found at", sample)
    else:
        df = pd.read_csv(sample)
        print("Raw shape:", df.shape)
        cleaned = clean_data(df)
        print("Cleaned shape:", cleaned.shape)
        # quick EDA for local runs only
        if 'Range_km' in cleaned.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(cleaned['Range_km'], kde=True)
            plt.title("Range_km distribution (local preview)")
            plt.show()
        # save preprocessed files for local testing
        out = base_dir / "data" / "cleaned_sample.csv"
        cleaned.to_csv(out, index=False)
        print(f"Saved cleaned sample to {out}")
