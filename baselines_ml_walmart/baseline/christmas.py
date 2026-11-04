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
means = df_raw.groupby("Is_Christmas_Season")["Weekly_Sales"].mean()
std = df_raw.groupby("Is_Christmas_Season")["Weekly_Sales"].std()
print(" Weekly_Sales Mean by Christmas Season:\n", means)
print(" Weekly_Sales Std by Christmas Season:\n", std)

cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)
df_encoded = df_encoded.fillna(0)

# Source Domain: Normal Days (In-Distribution)
# Target Domain: Holidays (Out-of-Distribution)
source_df = df_encoded[df_encoded['Is_Christmas_Season'] == 0].copy()
target_df = df_encoded[df_encoded['Is_Christmas_Season'] == 1].copy()

# Define feature columns (exclude target, proxies, and the splitting variable)
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
model.fit(X_source, y_source)
y_pred = model.predict(X_target)


mae = mean_absolute_error(y_target, y_pred)
rmse = np.sqrt(mean_squared_error(y_target, y_pred))
mape = custom_retail_mape(y_target,y_pred)
r2 = r2_score(y_target, y_pred)
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")
