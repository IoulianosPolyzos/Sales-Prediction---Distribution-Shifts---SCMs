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





df_raw = df_raw.drop(columns=["Date","Season","DayOfWeek","Month","WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Define domains based on city
source_cities = ['Houston', 'Philadelphia', 'Phoenix', 'San Jose', 'Jacksonville', 'Austin']
target_cities = ['New York', 'Los Angeles', 'Chicago']


source_df = df_raw[df_raw['city'].isin(source_cities)].copy()
target_df = df_raw[df_raw['city'].isin(target_cities)].copy()

# 2. ONE-HOT encoding
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]

source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)
TARGET = "Weekly_Sales"
# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")


lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}
model = LGBMRegressor(**lgb_params)
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)


# Predict ONLY on the 90% test portion to avoid data leakage
y_pred = model.predict(X_test_combined)

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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

df_raw = df_raw.drop(
    columns=[ "Date", "Season", "DayOfWeek", "Month",
             "WeekOfYear"], errors='ignore')
df_raw.columns = df_raw.columns.str.strip()

# Define domains based on city
source_weather = ['Clouds', 'Rain', 'Snow']
target_weather = ['Clear']


source_df = df_raw[df_raw['weather_condition'].isin(source_weather)].copy()
target_df = df_raw[df_raw['weather_condition'].isin(target_weather)].copy()

# 2. ONE-HOT encoding
cols_to_encode = ["city", "Type", "weather_condition", "Store", "Dept"]

source_df = pd.get_dummies(source_df, columns=cols_to_encode)
target_df = pd.get_dummies(target_df, columns=cols_to_encode)
source_df, target_df = source_df.align(target_df, join='left', axis=1, fill_value=0)

TARGET = "Weekly_Sales"

# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)

print(" Training LightGBM on ALL source + 10% target domain...")
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)


# Predict ONLY on the 90% test portion to avoid data leakage
y_pred = model.predict(X_test_combined)

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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
# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
model.fit(X_train_combined , y_train_combined)
y_pred = model.predict(X_test_combined)
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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
# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)
y_pred = model.predict(X_test_combined)
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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
# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)
y_pred = model.predict(X_test_combined)
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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
# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")
from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
print(" Training LightGBM on source domain...")
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)
y_pred = model.predict(X_test_combined)


mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

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
TARGET = "Weekly_Sales"

source_df = df_encoded[df_encoded['IsHoliday'] == 0].copy().reset_index(drop=True)
target_df = df_encoded[df_encoded['IsHoliday'] == 1].copy().reset_index(drop=True)

# Features / target
X_source = source_df.drop(columns=[TARGET])
y_source = source_df[TARGET]

X_target = target_df.drop(columns=[TARGET])
y_target = target_df[TARGET]

# Clean column names




# SOURCE DOMAIN: 90 train / 10 test

X_source_train, X_source_test, y_source_train, y_source_test = train_test_split(
    X_source,
    y_source,
    test_size=0.1,
    random_state=42
)


# TARGET DOMAIN: 10 train / 90 test

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target,
    y_target,
    train_size=0.1,
    random_state=42
)


# FINAL TRAIN SET
# 90% source + 10% target

X_train_combined = pd.concat(
    [X_source_train, X_target_train],
    ignore_index=True
)

y_train_combined = pd.concat(
    [y_source_train, y_target_train],
    ignore_index=True
)


# FINAL TEST SET
# 10% source + 90% target

X_test_combined = pd.concat(
    [X_source_test, X_target_test],
    ignore_index=True
)

y_test_combined = pd.concat(
    [y_source_test, y_target_test],
    ignore_index=True
)

# Shuffle train/test
train_idx = np.random.RandomState(42).permutation(len(X_train_combined))
X_train_combined = X_train_combined.iloc[train_idx].reset_index(drop=True)
y_train_combined = y_train_combined.iloc[train_idx].reset_index(drop=True)

test_idx = np.random.RandomState(42).permutation(len(X_test_combined))
X_test_combined = X_test_combined.iloc[test_idx].reset_index(drop=True)
y_test_combined = y_test_combined.iloc[test_idx].reset_index(drop=True)

# Info
print(f"Source train samples: {len(X_source_train)}")
print(f"Source test samples : {len(X_source_test)}")

print(f"Target train samples: {len(X_target_train)}")
print(f"Target test samples : {len(X_target_test)}")

print(f"\nFinal train set: {len(X_train_combined)}")
print(f"Final test set : {len(X_test_combined)}")

print(f" Original Source samples (100%): {len(X_source)}")
print(f" Target samples added to train set (10%): {len(X_target_train)}")
print(f" Remaining Target samples for testing (90%): {len(X_target_test)}")

# Combine ALL source data with the 10% portion of target data

from lightgbm import LGBMRegressor

lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': 1
}

model = LGBMRegressor(**lgb_params)
X_train_combined = X_train_combined.apply(pd.to_numeric, errors='coerce')
X_test_combined = X_test_combined.apply(pd.to_numeric, errors='coerce')

X_train_combined = X_train_combined.fillna(0).astype(np.float32)
X_test_combined = X_test_combined.fillna(0).astype(np.float32)
model.fit(X_train_combined, y_train_combined)
y_pred = model.predict(X_test_combined)


mae = mean_absolute_error(y_test_combined, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred))
mape = custom_retail_mape(y_test_combined, y_pred)
r2 = r2_score(y_test_combined, y_pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R²  : {r2:.4f}")
