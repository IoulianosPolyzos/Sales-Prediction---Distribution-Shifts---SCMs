import sys
from pathlib import Path
current_file = Path(__file__).resolve()

project_root = current_file.parents[2]


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

target_weather = ['Clear']

cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]

print(" Converting categorical columns to 'category' dtype...")
for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

target_df = df_raw[df_raw['weather_condition'].isin(target_weather)].copy()


TARGET = "Weekly_Sales"




X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]




print(" Splitting 10% of target data for training...")
# Split the target set: 10% goes to train, the remaining 90% stays for testing
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, train_size=0.1, random_state=42
)

print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")



lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}
model = LGBMRegressor(**lgb_params)

print(" Training LightGBM on 10% target domain...")
model.fit(X_target_train, y_target_train, categorical_feature=cols_to_encode)

print(" Predicting on the remaining 90% target test set...")
# Predict ONLY on the 90% test portion to avoid data leakage
y_pred = model.predict(X_target_test)

mae = mean_absolute_error(y_target_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_target_test, y_pred))
mape = custom_retail_mape(y_target_test, y_pred)
r2 = r2_score(y_target_test, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")