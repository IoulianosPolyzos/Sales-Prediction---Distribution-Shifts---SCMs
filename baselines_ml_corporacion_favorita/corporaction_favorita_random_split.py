#%%
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np

import os
import warnings
import sys
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

config_path = project_root / "scm_corporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]



# Path to your specific Walmart CSV file
PARQUET_PATH = project_root / "corporacion_favorita_dataset" / "final_data_favorita.parquet"



print(" Loading Favorita Dataset from Parquet...")
df_raw = pd.read_parquet(PARQUET_PATH)

df_raw = df_raw.drop(columns=[ "date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()


print(" Applying Manual One-Hot Encoding...")

source_df,target_df = train_test_split(df_raw, test_size=0.3, random_state=42)
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)

TARGET = "unit_sales"

X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

from lightgbm import LGBMRegressor

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
model.fit(X_source, y_source)
y_pred = model.predict(X_target)
from sklearn.metrics import mean_squared_error, r2_score


mae = mean_absolute_error(y_target, y_pred)
mse = mean_squared_error(y_target, y_pred)
rmse = np.sqrt(mean_squared_error(y_target, y_pred))
r2 = r2_score(y_target, y_pred)
print(" MAE on Target Domain :", mae)
print(" MSE on Target Domain :", mse)
print(" RMSE on Target Domain :", rmse)
print(" R² on Target Domain:", r2)
