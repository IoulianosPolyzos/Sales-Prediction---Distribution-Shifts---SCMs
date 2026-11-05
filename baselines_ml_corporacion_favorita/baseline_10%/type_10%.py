from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
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

# Clean up columns
df_raw = df_raw.drop(
    columns=[ "date", "Season", "DayOfWeek", "Month",
             "WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Manual One-Hot Encoding


cols_to_encode = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

# Source Domain: Store Types C, B, D, E (In-Distribution)
# Target Domain: Store Type A (Out-of-Distribution)
source_df = df_raw[df_raw['store_type'].isin(['C','B','D','E'])].copy()
target_df = df_raw[df_raw['store_type'].isin(['A'])].copy()

source_df = source_df.sample(n=min(500000, len(source_df)), random_state=42).reset_index(drop=True)

adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,
    random_state=42
)

adaptation_df = adaptation_df.sample(n=min(100000, len(adaptation_df)), random_state=42).reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

TARGET = "unit_sales"

train_combined = pd.concat([source_df, adaptation_df], ignore_index=True)

X_train_combined = train_combined.drop(columns=[TARGET])
y_train_combined = train_combined[TARGET]

X_target_test = holdout_test_df.drop(columns=[TARGET])
y_target_test = holdout_test_df[TARGET]

print(f" Source samples (Subsampled): {len(source_df)}")
print(f" Target samples added to train set (Adaptation): {len(adaptation_df)}")
print(f" Total combined training samples: {len(X_train_combined)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
model.fit(X_train_combined, y_train_combined, categorical_feature=cols_to_encode)
y_pred = model.predict(X_target_test)
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_target_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_target_test, y_pred))
mape = mean_absolute_percentage_error(y_target_test, y_pred)
r2 = r2_score(y_target_test, y_pred)

print(f"MAE on Target Domain (90%): {mae:.2f}")
print(f"RMSE on Target Domain (90%): {rmse:.2f}")
print(f"MAPE on Target Domain (90%): {mape:.2f}")
print(f"R² on Target Domain (90%): {r2:.2f}")
