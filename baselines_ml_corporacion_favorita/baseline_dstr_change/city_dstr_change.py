from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np



# Path to your specific Walmart CSV file
PARQUET_PATH = r"../../../corporacion_favorita_dataset/final_data_favorita.parquet"
print(" Loading Favorita Dataset from Parquet...")
df_raw = pd.read_parquet(PARQUET_PATH)





df_raw = df_raw.drop(columns=["Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Define domains based on city
source_cities = ["Quito", "Guayaquil", "Cuenca", "Santo Domingo", "Machala", "Loja", "Manta"]
target_cities = [c for c in df_raw['city'].unique() if c not in source_cities]


source_df = df_raw[df_raw['city'].isin(source_cities)].copy()
target_df = df_raw[df_raw['city'].isin(target_cities)].copy()

# 2. ONE-HOT encoding
cols_to_encode = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)
TARGET = "unit_sales"
# Features / target
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


lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}
model = LGBMRegressor(**lgb_params)
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)


# Predict ONLY on the 90% test portion to avoid data leakage
y_pred = model.predict(X_test_combined)

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = mean_absolute_percentage_error(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")