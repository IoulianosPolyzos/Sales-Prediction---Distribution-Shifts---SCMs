from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape



CSV_PATH = r"../../../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]

print(" Converting categorical columns to 'category' dtype...")
for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

source_df = df_raw[df_raw['Store'].astype(int).between(1, 30)].copy()
target_df = df_raw[df_raw['Store'].astype(int).between(31, 45)].copy()


mean_src = source_df['Weekly_Sales'].mean()
std_src = source_df['Weekly_Sales'].std()

mean_trg = target_df['Weekly_Sales'].mean()
std_trg = target_df['Weekly_Sales'].std()

print("Stores 1-30 -> mean:", mean_src, ", std:", std_src)
print("Store 31-45 -> mean:", mean_trg, ", std:", std_trg)
TARGET = "Weekly_Sales"

X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

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
mape = custom_retail_mape(y_target,y_pred)
r2 = r2_score(y_target, y_pred)
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")