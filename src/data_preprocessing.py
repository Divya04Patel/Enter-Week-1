# =====================================
# 📊 Electric Vehicle Range Prediction
# Step 3: EDA + Data Preprocessing
# =====================================

# ✅ Import libraries
from pathlib import Path
import pandas as pd
import numpy as np

def _find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean uploaded EV dataframe and return cleaned DataFrame.
    - Normalizes common column names
    - Converts to numeric, drops rows with missing values in selected columns
    - Drops duplicates
    """
    df = df.copy()
    mapping = {}
    battery_col = _find_column(df, ['Battery', 'Battery_Capacity_kWh', 'battery_kwh'])
    power_col = _find_column(df, ['Power', 'Power_hp', 'Power_kW', 'Motor_Power'])
    efficiency_col = _find_column(df, ['Efficiency', 'Efficiency_WhPerKm', 'Energy_Consumption'])
    weight_col = _find_column(df, ['Weight', 'Weight_kg', 'Vehicle_Weight'])
    range_col = _find_column(df, ['Range', 'Range_km', 'range_km'])

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

    wanted = [c for c in ['Battery_Capacity_kWh', 'Power_hp', 'Efficiency_WhPerKm', 'Weight_kg', 'Range_km'] if c in df.columns]
    if not wanted:
        # nothing to clean meaningfully — return original (or empty) dataframe
        return df.copy()

    df = df[wanted].copy()

    # convert columns to numeric where appropriate
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

# Optional: keep previous script behavior behind a guard so imports are safe
if __name__ == "__main__":
    # simple CLI behavior for local testing only (won't run on import)
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "data" / "ev_dataset.csv"
    if file_path.exists():
        df = pd.read_csv(file_path)
        cleaned = clean_data(df)
        out = base_dir / "data" / "cleaned_sample.csv"
        cleaned.to_csv(out, index=False)
        print(f"Saved cleaned sample to {out}")
    else:
        print("No sample dataset found at", file_path)

# ✅ Load dataset
base_dir = Path(__file__).resolve().parent.parent  # repo root (one level up from src)
file_path = base_dir / "data" / "ev_dataset.csv"
df = pd.read_csv(file_path)

# ✅ Display basic info
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nPreview of Data:")
try:
    from IPython.display import display
    display(df.head())
except Exception:
    print(df.head().to_string())

# ✅ Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# ✅ Drop duplicate rows if any
df.drop_duplicates(inplace=True)

# ✅ Basic statistics
display(df.describe())

# =====================================
# 🔍 Exploratory Data Analysis
# =====================================

# 1️⃣ Distribution of target variable (Range)
plt.figure(figsize=(8, 5))
sns.histplot(df['Range_km'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of EV Range (km)")
plt.xlabel("Range (km)")
plt.ylabel("Count")
plt.show()

# 2️⃣ Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 3️⃣ Scatter between Range and Battery Capacity
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Battery_Capacity_kWh', y='Range_km', data=df)
plt.title("Battery Capacity vs EV Range")
plt.show()

# =====================================
# ⚙️ Data Cleaning & Preprocessing
# =====================================

# ✅ Select important numeric columns
numeric_cols = ['Battery_Capacity_kWh', 'Power_hp', 'Efficiency_WhPerKm', 'Weight_kg', 'Range_km']
df = df[numeric_cols].dropna()

# ✅ Encode / scale data if needed
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('Range_km', axis=1))

# ✅ Convert back to DataFrame
feature_cols = df.drop(columns='Range_km').columns.tolist()
X = pd.DataFrame(scaled_data, columns=feature_cols)
y = df['Range_km']

print("\n✅ Data is ready for model training!")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ✅ Save preprocessed data
out_dir = base_dir / "data"
out_dir.mkdir(parents=True, exist_ok=True)
X.to_csv(out_dir / "X_preprocessed.csv", index=False)
y.to_frame(name='Range_km').to_csv(out_dir / "y_preprocessed.csv", index=False)

print("\n✅ Preprocessed data saved successfully!")
