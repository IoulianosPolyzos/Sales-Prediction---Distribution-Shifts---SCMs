#%%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np





# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"



print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()


print(" Applying Manual One-Hot Encoding...")

source_df,target_df = train_test_split(df_raw, test_size=0.3, random_state=42)
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)

TARGET = "Weekly_Sales"

X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'n_jobs': 1
}
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
