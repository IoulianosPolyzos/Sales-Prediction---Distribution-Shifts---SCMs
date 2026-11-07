import sys
from pathlib import Path
current_file = Path(__file__).resolve()

project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw.drop(columns=[ "Date","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

seasons_124 = df_raw[df_raw['Season'].isin([1,2,4])]['Weekly_Sales']
mean_124 = seasons_124.mean()
std_124 = seasons_124.std()

seasons_3 = df_raw[df_raw['Season'] == 3]['Weekly_Sales']
mean_3 = seasons_3.mean()
std_3 = seasons_3.std()

print("Season 1,2,4 -> mean:", mean_124, ", std:", std_124)
print("Season 3 -> mean:", mean_3, ", std:", std_3)

# Manual One-Hot Encoding


cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]

print(" Converting categorical columns to 'category' dtype...")
for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

# Source Domain: Seasons 1, 2, 4 (In-Distribution)
# Target Domain: Season 3 (Out-of-Distribution)
source_df = df_raw[df_raw['Season'].isin([1, 2, 4])].copy()
target_df = df_raw[df_raw['Season'].isin([3])].copy()

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
