from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np




PARQUET_PATH = r"../../../corporacion_favorita_dataset/final_data_favorita.parquet"


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

# Source Domain: Store Types C, B, D, E (In-Distribution)
# Target Domain: Store Type A (Out-of-Distribution)
source_df = df_raw[df_raw['store_type'].isin(['C','B','D','E'])].copy()
target_df = df_raw[df_raw['store_type'].isin(['A'])].copy()

source_df = source_df.sample(n=min(1000000, len(source_df)), random_state=42).reset_index(drop=True)


adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.20,
    random_state=42
)

TARGET = "unit_sales"

X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = holdout_test_df.drop(columns=[TARGET])
y_target = holdout_test_df[TARGET]

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
model.fit(X_source, y_source, categorical_feature=cols_to_encode)
y_pred = model.predict(X_target)


mae = mean_absolute_error(y_target, y_pred)
rmse = np.sqrt(mean_squared_error(y_target, y_pred))
mape = mean_absolute_percentage_error(y_target,y_pred)
r2 = r2_score(y_target, y_pred)
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")

