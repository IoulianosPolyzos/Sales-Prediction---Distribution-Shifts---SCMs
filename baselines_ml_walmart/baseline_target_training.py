#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Define domains based on city
target_cities = ['New York', 'Los Angeles', 'Chicago']

print(" Applying Manual One-Hot Encoding...")


target_df = df_raw[df_raw['city'].isin(target_cities)].copy()
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]


target_df = pd.get_dummies(target_df, columns=cols_to_encode)


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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

target_weather = ['Clear']

print(" Applying Manual One-Hot Encoding...")

target_df = df_raw[df_raw['weather_condition'].isin(target_weather)].copy()
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]


target_df = pd.get_dummies(target_df, columns=cols_to_encode)


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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()



print(" Applying Manual One-Hot Encoding...")
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)
df_encoded = df_encoded.fillna(0)

target_df = df_encoded[df_encoded['Is_Christmas_Season'] == 1].copy()
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]




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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()



print(" Applying Manual One-Hot Encoding...")
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]


target_df = df_raw[df_raw['Store'].between(31, 45)].copy()
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]




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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()



print(" Applying Manual One-Hot Encoding...")

cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]

target_df = df_raw[df_raw['Season'].isin([3])].copy()
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]




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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()



print(" Applying Manual One-Hot Encoding...")

cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]

target_df = df_raw[df_raw['Type'].isin(['A','B'])].copy()
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]




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
model.fit(X_target_train, y_target_train)

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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape

# Path to your specific Walmart CSV file
CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()



print(" Applying Manual One-Hot Encoding...")
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)
df_encoded = df_encoded.fillna(0)

target_df = df_encoded[df_encoded['IsHoliday'] == 1].copy()
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]




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
model.fit(X_target_train, y_target_train)

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