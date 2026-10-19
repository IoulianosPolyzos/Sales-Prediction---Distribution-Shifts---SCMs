#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()
means = df_raw.groupby("IsHoliday")["Weekly_Sales"].mean()
std = df_raw.groupby("IsHoliday")["Weekly_Sales"].std()
print(" Weekly_Sales Mean by IsHoliday:\n", means)
print(" Weekly_Sales Std by IsHoliday:\n", std)

cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)
df_encoded = df_encoded.fillna(0)

# Source Domain: Normal Days (In-Distribution)
# Target Domain: Holidays (Out-of-Distribution)
source_df = df_encoded[df_encoded['IsHoliday'] == 0].copy()
target_df = df_encoded[df_encoded['IsHoliday'] == 1].copy()

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


#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape



CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"


print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)

# Clean up columns
df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Manual One-Hot Encoding


# Source Domain: Store Type C (In-Distribution)
# Target Domain: Store Types A and B (Out-of-Distribution)
source_df = df_raw[df_raw['Type'].isin(['C'])].copy()
target_df = df_raw[df_raw['Type'].isin(['A','B'])].copy()
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)

# 'Weekly_Sales' is dropped from features because it's our target now
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


#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

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


# Source Domain: Seasons 1, 2, 4 (In-Distribution)
# Target Domain: Season 3 (Out-of-Distribution)
source_df = df_raw[df_raw['Season'].isin([1, 2,4])].copy()
target_df = df_raw[df_raw['Season'].isin([3])].copy()
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

#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape



CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

print(" Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)
df_raw = df_raw.drop(columns=[ "Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()
df_raw['Store'] = df_raw['Store'].astype(int)



source_df = df_raw[df_raw['Store'].between(1, 30)].copy()
target_df = df_raw[df_raw['Store'].between(31, 45)].copy()
cols_to_encode = ["city", "Type", "weather_condition",
                    "Store", "Dept"]
source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)


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
#%%
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
from scm_walmart.custom_mape import custom_retail_mape


CSV_PATH = r"../walmart_dataset/final_data_walmart.csv"

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
# Define domains based on city
source_cities = ['Houston', 'Philadelphia', 'Phoenix', 'San Jose', 'Jacksonville', 'Austin']
target_cities = ['New York', 'Los Angeles', 'Chicago']

print(" Applying Manual One-Hot Encoding...")


source_df = df_raw[df_raw['city'].isin(source_cities)].copy()
target_df = df_raw[df_raw['city'].isin(target_cities)].copy()
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


# Define domains based on city
source_weather = ['Clouds','Rain','Snow']
target_weather = ['Clear']

print(" Applying Manual One-Hot Encoding...")


source_df = df_raw[df_raw['weather_condition'].isin(source_weather)].copy()
target_df = df_raw[df_raw['weather_condition'].isin(target_weather)].copy()
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
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain ...")
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

