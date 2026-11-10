from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np

import sys
from pathlib import Path
current_file = Path(__file__).resolve()

project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

PARQUET_PATH = project_root / "corporacion_favorita_dataset" / "final_data_favorita.parquet"

print(" Loading Favorita Dataset from Parquet...")
df_raw = pd.read_parquet(PARQUET_PATH)

df_raw = df_raw.drop(columns=[ "date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

means = df_raw.groupby("Is_Christmas_Season")["unit_sales"].mean()
std = df_raw.groupby("Is_Christmas_Season")["unit_sales"].std()
print(" unit_sales Mean by Christmas Season:\n", means)
print(" unit_sales Std by Christmas Season:\n", std)

cols_to_encode = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

df_encoded = df_raw.copy()
for col in cols_to_encode:
    df_encoded[col] = df_encoded[col].astype('category')

# --- Τυχαίος διαχωρισμός 70/30 (Χωρίς Domain Shift λογική) ---
train_df, test_df = train_test_split(
    df_encoded,
    test_size=0.30,
    random_state=42
)

# Sampling στο Train για αποφυγή Memory Errors
train_df = train_df.sample(n=min(1000000, len(train_df)), random_state=42).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

TARGET = "unit_sales"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on random 70/30 split...")
model.fit(X_train, y_train, categorical_feature=cols_to_encode)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n PREDICTION METRICS")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")