from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np


PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"

print(" Loading Favorita Dataset from Parquet...")
df_raw = pd.read_parquet(PARQUET_PATH)
df_raw = df_raw.drop(
    columns=[ "Date", "Season", "DayOfWeek", "Month",
             "WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()
df_raw['store_nbr'] = df_raw['store_nbr'].astype(int)

source_df = df_raw[df_raw['store_nbr'].between(1, 41)].copy()
target_df = df_raw[df_raw['store_nbr'].between(41, 55)].copy()
cols_to_encode = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]
source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)

mean_src = source_df['unit_sales'].mean()
std_src = source_df['unit_sales'].std()

mean_trg = target_df['unit_sales'].mean()
std_trg = target_df['unit_sales'].std()

print("Stores 1-41 -> mean:", mean_src, ", std:", std_src)
print("Store 41-55 -> mean:", mean_trg, ", std:", std_trg)
TARGET = "unit_sales"

X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]


X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, train_size=0.10, random_state=42
)

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%: {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")

# Combine ALL source data with the 10%portion of target data
X_train_combined = pd.concat([X_source, X_target_train], ignore_index=True)
y_train_combined = pd.concat([y_source, y_target_train], ignore_index=True)
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
model.fit(X_train_combined, y_train_combined)
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