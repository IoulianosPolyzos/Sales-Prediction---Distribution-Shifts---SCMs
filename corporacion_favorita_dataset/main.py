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
df_macro.set_index('date', inplace=True)

if 'cpi' in df_macro.columns:
    data['cpi'] = data['date'].map(df_macro['cpi'])
if 'unemployment' in df_macro.columns:
    data['unemployment'] = data['date'].map(df_macro['unemployment'])

del df_macro
gc.collect()

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

print("\nOptimizing memory footprint (Downcasting)...")
for col in data.select_dtypes(include=['float64']).columns:
    data[col] = pd.to_numeric(data[col], downcast='float')
for col in data.select_dtypes(include=['int64']).columns:
    data[col] = pd.to_numeric(data[col], downcast='integer')
gc.collect()

print("\nFinal data types (After Optimization):")
print(data.dtypes)

#%%


print("3. DATA CLEANING")


print("Checking negatives for 'unit_sales'...")
neg_count = (data['unit_sales'] < 0).sum()
print(f"Negatives found: {neg_count}")

if neg_count > 0:
    print("Removing negatives in-place...")
    data.drop(data[data['unit_sales'] < 0].index, inplace=True)
    gc.collect()

print("\nChecking zeros for 'unit_sales'...")
zero_count = (data['unit_sales'] == 0).sum()
print(f"Zeros found: {zero_count} ({(zero_count/len(data))*100:.2f}%)")

print("\nOutliers Check: Skipped for full dataset to prevent memory crash.")
#%%


print("4. FINALIZATION & TARGET EXTRACTION")


micro_sales = ((data['unit_sales'] > 0) & (data['unit_sales'] < 1)).sum()
small_sales = ((data['unit_sales'] >= 1) & (data['unit_sales'] < 10)).sum()
total_rows = len(data)

print(f"Total rows: {total_rows}")
print(f"Unit Sales between 0 and 1 (0, 1): {micro_sales} ({(micro_sales/total_rows)*100:.2f}%)")
print(f"Unit Sales between 1 and 10 [1, 10): {small_sales} ({(small_sales/total_rows)*100:.2f}%)")


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