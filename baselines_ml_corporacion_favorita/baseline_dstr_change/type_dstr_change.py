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
df_raw = df_raw.drop(columns=[ "date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Manual One-Hot Encoding


cols_to_encode = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

source_df = df_raw[df_raw['store_type'].isin(['C','B','D','E'])].copy()
target_df = df_raw[df_raw['store_type'].isin(['A'])].copy()
# Features / target
TARGET = "unit_sales"
source_df = source_df.sample(n=min(1000000, len(source_df)), random_state=42).reset_index(drop=True)
target_df = target_df.sample(n=min(1000000, len(target_df)), random_state=42).reset_index(drop=True)
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on Dtrain...")
model.fit(X_train_combined, y_train_combined, categorical_feature=cols_to_encode)
y_pred = model.predict(X_test_combined)


mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = mean_absolute_percentage_error(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")

