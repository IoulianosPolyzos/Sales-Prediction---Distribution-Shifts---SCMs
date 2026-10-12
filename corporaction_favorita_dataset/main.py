#%%
import os
import pandas as pd
import numpy as np
import importlib
import gc

# Import custom modules
import data_join
import get_weather
import plotting
import data_preprocessing
import feature_engineering
import get_cpi_unemployment

# Reload modules to ensure latest changes are applied
importlib.reload(feature_engineering)
importlib.reload(plotting)
importlib.reload(data_preprocessing)
importlib.reload(data_join)
importlib.reload(get_weather)
importlib.reload(get_cpi_unemployment)

from feature_engineering import *
from data_preprocessing import *
from get_weather import *
from data_join import create_master_dataset
from get_cpi_unemployment import get_macro_data

#%%


print("1. LOADING DATASET")


# Load Favorita master dataset (full dataset, returned from Parquet)
data = create_master_dataset()

#%%


print("2. FEATURE ENGINEERING")


print("Creating time-based features...")
data = create_time_based_features(data, include_cyclic_features=True)

print("Fetching and adding macro features (CPI & Unemployment)...")
df_macro = get_macro_data()
data = data.merge(df_macro, on='date', how='left')

del df_macro
import gc; gc.collect()

print("\nChecking actual date format...")

print("\nChecking actual date format...")
sample_dates_train = data['date'].head(10).tolist() if 'date' in data.columns else "No Date column"
print(f"Sample dates from dataset: {sample_dates_train}")
print(f"Date column dtype: {data['date'].dtype}")

if 'date' in data.columns:
    unique_dates = data['date'].astype(str).unique()[:5]
    print(f"Unique date samples: {unique_dates}")

print(f"\nTotal columns before holiday features: {len(data.columns)}")

print("Creating holiday features...")
data = create_holiday_features(data)
print(f"Total columns after holiday features: {len(data.columns)}")

print("Fetching and adding weather features...")
data = add_weather_features(data)
print(f"Total columns after weather features: {len(data.columns)}")

print("\nFinal data check:")
print(f"Dataset shape: {data.shape}")
print(f"Remaining NaN values: {data.isna().sum().sum()}")

print("\nFinal data types:")
print(data.dtypes)

#%%


print("3. DATA CLEANING")


print("Negatives before cleaning:")
neg_report = check_negatives(data)
print(neg_report)

neg_cols = ["unit_sales"]
data = remove_negatives(data, neg_cols)

print("\nNegatives after cleaning:")
neg_report = check_negatives(data)
print(neg_report)

print("\nZeros Check:")
zero_percent = check_zeros(data)
print(zero_percent)

print("\nOutliers Check:")
outliers_report = check_outliers(data)
print(outliers_report)

#%%


print("4. FINALIZATION & TARGET EXTRACTION")


micro_sales = ((data['unit_sales'] > 0) & (data['unit_sales'] < 1)).sum()
small_sales = ((data['unit_sales'] >= 1) & (data['unit_sales'] < 10)).sum()
total_rows = len(data)

print(f"Total rows: {total_rows}")
print(f"Unit Sales between 0 and 1 (0, 1): {micro_sales} ({(micro_sales/total_rows)*100:.2f}%)")
print(f"Unit Sales between 1 and 10 [1, 10): {small_sales} ({(small_sales/total_rows)*100:.2f}%)")

# Extract final feature list
features, target = get_features(
    data=data,
    include_enhanced_holiday=True,
    include_cyclic_features=True
)
print(f"\nUsing {len(features)} features")
print(f"Target: {target}")

missing_features = [f for f in features if f not in data.columns]
if missing_features:
    print(f"\nMissing features: {missing_features}")
    print("Available features:")
    for col in data.columns:
        print(f"   - {col}")
else:
    print(f"\nAll {len(features)} expected features are available in the dataset")

#%%


print("5. SAVING FINAL DATASET")


print(f"Final dataset shape before saving: {data.shape}")

# Drop Unnamed columns if they accidentally appeared during merges
unnamed_columns = [col for col in data.columns if 'Unnamed' in col]
if unnamed_columns:
    print(f"Removing useless columns: {unnamed_columns}")
    data = data.drop(columns=unnamed_columns)

# Save ONLY to Parquet (CSV removed to prevent OOM)
output_file = "final_data_favorita.parquet"
print(f"Saving final dataset to {output_file}...")
data.to_parquet(output_file, index=False)
print("Success! Final parquet dataset saved.")