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

# Clean up columns
df_raw = df_raw.drop(
    columns=[ "Date", "Season", "DayOfWeek", "Month",
             "WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Manual One-Hot Encoding


cols_to_encode = ["city", "Type", "weather_condition",
                  "Store", "Dept"]

print(" Converting categorical columns to 'category' dtype...")
for col in cols_to_encode:
    df_raw[col] = df_raw[col].astype('category')

# Source Domain: Store Type C (In-Distribution)
# Target Domain: Store Types A and B (Out-of-Distribution)
source_df = df_raw[df_raw['Type'].isin(['C'])].copy()
target_df = df_raw[df_raw['Type'].isin(['A', 'B'])].copy()

# 'Weekly_Sales' is dropped from features because it's our target now
TARGET = "Weekly_Sales"

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
model.fit(X_train_combined, y_train_combined, categorical_feature=cols_to_encode)
y_pred = model.predict(X_target_test)
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_target_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_target_test, y_pred))
mape = custom_retail_mape(y_target_test, y_pred)
r2 = r2_score(y_target_test, y_pred)

print(f"MAE on Target Domain (90%): {mae:.2f}")
print(f"RMSE on Target Domain (90%): {rmse:.2f}")
print(f"MAPE on Target Domain (90%): {mape:.2f}")
print(f"R² on Target Domain (90%): {r2:.2f}")
