import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from pathlib import Path

nb = new_notebook()
cells = []

cells.append(new_markdown_cell("# EV Range — preprocessing and model training\nRun cells sequentially. This notebook loads data, cleans it, creates features, trains a RandomForest baseline and saves the model."))
cells.append(new_code_cell("""# 1) imports
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path.cwd()
DATA_PATH = BASE / "data" / "ev_dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)
print("BASE", BASE)
"""))

cells.append(new_code_cell("""# 2) cleaning function
def _find_column(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None

def clean_data(df):
    mapping = {}
    battery_col = _find_column(df, ['Battery', 'Battery_Capacity_kWh', 'battery_kwh'])
    power_col = _find_column(df, ['Power', 'Power_hp', 'Power_kW', 'Motor_Power'])
    efficiency_col = _find_column(df, ['Efficiency', 'Efficiency_WhPerKm', 'Energy_Consumption'])
    weight_col = _find_column(df, ['Weight', 'Weight_kg', 'Vehicle_Weight'])
    range_col = _find_column(df, ['Range', 'Range_km', 'range_km', 'range'])
    if battery_col: mapping[battery_col] = 'Battery_Capacity_kWh'
    if power_col: mapping[power_col] = 'Power_hp'
    if efficiency_col: mapping[efficiency_col] = 'Efficiency_WhPerKm'
    if weight_col: mapping[weight_col] = 'Weight_kg'
    if range_col: mapping[range_col] = 'Range_km'
    if mapping:
        df = df.rename(columns=mapping)
    wanted = [c for c in ['Battery_Capacity_kWh','Power_hp','Efficiency_WhPerKm','Weight_kg','Range_km'] if c in df.columns]
    if not wanted:
        return df.copy()
    df = df[wanted].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df
"""))

cells.append(new_code_cell("""# 3) load & clean
if not DATA_PATH.exists():
    raise FileNotFoundError(f\"Dataset not found at {DATA_PATH}\")
raw = pd.read_csv(DATA_PATH)
print('raw shape', raw.shape)
df = clean_data(raw)
print('cleaned shape', df.shape)
df.head()"""))

cells.append(new_code_cell("""# 4) feature engineering
if 'Battery_Capacity_kWh' in df.columns:
    df['battery_Wh'] = df['Battery_Capacity_kWh'] * 1000.0
if 'Efficiency_WhPerKm' in df.columns and 'battery_Wh' in df.columns:
    df['battery_over_eff'] = df['battery_Wh'] / df['Efficiency_WhPerKm']
if 'Battery_Capacity_kWh' in df.columns and 'Weight_kg' in df.columns:
    df['energy_density_kWh_per_kg'] = df['Battery_Capacity_kWh'] / df['Weight_kg'].replace(0, np.nan)
df.shape"""))

cells.append(new_code_cell("""# 5) prepare train/test and preprocess numeric features
if 'Range_km' not in df.columns:
    raise SystemExit(\"Missing target Range_km\")
feature_cols = [c for c in df.columns if c != 'Range_km']
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print('numeric features:', numeric_cols)
for c in numeric_cols:
    lo, hi = df[c].quantile([0.01, 0.99])
    df[c] = df[c].clip(lower=lo, upper=hi)
X = df[feature_cols].copy()
y = df['Range_km'].astype(float).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_train_num = imputer.fit_transform(X_train[numeric_cols])
X_train_num = scaler.fit_transform(X_train_num)
X_test_num = imputer.transform(X_test[numeric_cols])
X_test_num = scaler.transform(X_test_num)
import pandas as pd
X_train_p = pd.DataFrame(X_train_num, columns=numeric_cols)
X_test_p = pd.DataFrame(X_test_num, columns=numeric_cols)
out_dir = BASE / 'data'
out_dir.mkdir(exist_ok=True)
X_train_p.to_csv(out_dir / 'X_train_preprocessed.csv', index=False)
X_test_p.to_csv(out_dir / 'X_test_preprocessed.csv', index=False)
y_train.to_csv(out_dir / 'y_train.csv', index=False)
y_test.to_csv(out_dir / 'y_test.csv', index=False)
joblib.dump({'imputer': imputer, 'scaler': scaler, 'numeric_cols': numeric_cols}, MODEL_DIR / 'preprocessor.pkl')
print('Saved preprocessed data and preprocessor')"""))

cells.append(new_code_cell("""# 6) train baseline RandomForest (small grid)
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {'n_estimators':[100,300], 'max_depth':[8,16,None], 'min_samples_leaf':[1,3]}
gs = GridSearchCV(rf, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1, verbose=1)
gs.fit(X_train_p, y_train)
best = gs.best_estimator_
print('best params:', gs.best_params_)
cv_mae = -cross_val_score(best, X_train_p, y_train, cv=5, scoring='neg_mean_absolute_error').mean()
print('Train CV MAE:', cv_mae)
y_pred = best.predict(X_test_p)
print('Test MAE:', mean_absolute_error(y_test, y_pred))
print('Test RMSE:', mean_squared_error(y_test, y_pred, squared=False))
print('Test R2:', r2_score(y_test, y_pred))
joblib.dump(best, MODEL_DIR / 'ev_range_model.pkl')
print('Saved model to', MODEL_DIR / 'ev_range_model.pkl')"""))

nb['cells'] = cells
out_path = Path.cwd() / "notebooks" / "EV_Range_Training.ipynb"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Notebook written to", out_path)