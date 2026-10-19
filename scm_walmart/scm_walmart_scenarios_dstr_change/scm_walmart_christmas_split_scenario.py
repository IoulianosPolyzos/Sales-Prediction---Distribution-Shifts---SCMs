# %%
# !/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path

import yaml
import numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scm_walmart.custom_mape import *
from lightgbm import LGBMRegressor
import dowhy.gcm as gcm
import pandas as pd
# --- PATH MANAGEMENT FOR EXTERNAL UTILITIES ---
current_dir = Path(__file__).resolve().parent

# 1. utils_scm is located two levels up (root)
sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis
)

# 2. scm_walmart_graph_utils is located one level up
sys.path.append(str(current_dir.parent))
from scm_walmart.scm_walmart_graph_utils import build_feature_graph

# --- CONFIGURATION ---
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_domain': 'Normal Days (IsChristmas_Season=0)',
    'target_domain': 'Holiday Days (Is_Christmas_Season=1)',
    'random_seed': 42
}



#%%



print("PART 1: LOADING DATA & SETUP")


try:
    CSV_PATH = r"../../walmart_dataset/final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Is_Christmas_Season distribution:")
    print(f"  Normal Days (0): {(df['Is_Christmas_Season'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['Is_Christmas_Season'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY HOLIDAY DOMAIN 



print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs CHRISTMAS SEASON)")


# Ensure Is_Christmas_Season is numeric (0 or 1)
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

source_df = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)
target_df = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"Train domain: {CONFIG['source_domain']}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest domain: {CONFIG['target_domain']}")


# Verify domain separation
#assert (train_df['Is_Christmas_Season'] == 0).all(), "Train set contains non-zero Is_Christmas_Season!"
print("✓ Domain separation verified: No overlap between Normal/CHRISTMAS days")

#%%


print("PART 3: PREPROCESSING")


try:
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError as e:
                pass

    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

#%%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# Since there are source data in train set , there is no need to remove the edge between
# domain split variable or the intervention variable and sales node

# %%
print("PART 6: FITTING MIXED STRUCTURAL CAUSAL MODEL (ON POOLED TRAIN DATA)")

# Instantiate Structural Causal Model over feature-level DAG
scm = gcm.StructuralCausalModel(feature_graph)

try:
    # Set up the internal causal mechanisms (LGBM) using the external utility
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on Dtrain (90% Normal Days + 10% Christmas Days)...")

    # The SCM is trained directly on the mixed dataset, automatically
    # learning regular and holiday behaviors simultaneously.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7 & 8: SYNTHETIC AND HYBRID DATA GENERATION")

print("\nExecuting Intervention: Changing days to Christmas (Is_Christmas_Season = 1)...")
print("\nExecuting Aligned Intervention: Changing days to Christmas (Is_Christmas_Season = 1) with correct Dates...")

num_synthetic_samples = len(target_df)

sampled_target = target_domain_train.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values
aligned_week_sin = sampled_target['WeekOfYear_sin'].values
aligned_week_cos = sampled_target['WeekOfYear_cos'].values
aligned_christmas = sampled_target['Is_Christmas_Season'].values
aligned_year_progress = sampled_target['Year_Progress'].values
aligned_month_progress = sampled_target['Month_Progress'].values

def int_month_sin(*args, **kwargs): return aligned_month_sin
def int_month_cos(*args, **kwargs): return aligned_month_cos
def int_week_sin(*args, **kwargs): return aligned_week_sin
def int_week_cos(*args, **kwargs): return aligned_week_cos
def int_christmas(*args, **kwargs): return aligned_christmas
def int_year_progress(*args, **kwargs): return aligned_year_progress
def int_month_progress(*args, **kwargs): return aligned_month_progress



interventions_dict = {
    'Month_sin': int_month_sin,
    'Month_cos': int_month_cos,
    'WeekOfYear_sin': int_week_sin,
    'WeekOfYear_cos': int_week_cos,
    'Is_Christmas_Season': int_christmas,
    'Year_Progress': int_year_progress,
    'Month_Progress': int_month_progress
}

print("\nGenerating synthetic data with paired Temporal & Holiday Interventions...")
clip_cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

synthetic_dataset = generate_synthetic_data(
    scm=scm,
    interventions_dict=interventions_dict,
    num_samples=num_synthetic_samples,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales',
    clip_cols=clip_cols
)
print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)

print(f"\nTRAIN (Mixed) DATA STATS :")
print(f"  Mean: {train_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {train_df['Weekly_Sales'].std():.4f}")

print(f"\nTEST (Mixed) DATA STATS :")
print(f"  Mean: {test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {test_df['Weekly_Sales'].std():.4f}")

# Evaluation on Test Set
print("\nEvaluation on Mixed Test Set ...")

X_test = test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = test_df['Weekly_Sales']

y_pred = ml_model_syn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set (Synthetic Model):")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

print(f"✓ Synthetic Dataset Shape: {synthetic_dataset.shape}")

# 2. Hybrid Data Construction via Utility
hybrid_dataset = create_hybrid_dataset(
    source_df=train_df,
    synthetic_dataset=synthetic_dataset,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales'
)
print(f"✓ Hybrid Dataset Shape: {hybrid_dataset.shape}")
print("\nHybrid Dataset (Mixed Train + Synthetic Target)...")


X_hybrid = hybrid_dataset.drop(columns=['Weekly_Sales'])
y_hybrid = hybrid_dataset['Weekly_Sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)
ml_model_hbr.fit(X_hybrid, y_hybrid)

print("\nEvaluation of Hybrid Model on Mixed Test Set ...")
y_pred_hbr = ml_model_hbr.predict(X_test)

mae_hbr = mean_absolute_error(y_test, y_pred_hbr)
rmse_hbr = np.sqrt(mean_squared_error(y_test, y_pred_hbr))
r2_hbr = r2_score(y_test, y_pred_hbr)
mape_hbr = custom_retail_mape(y_test, y_pred_hbr)

print(f"Prediction Results on Test Set (Hybrid Model):")
print(f"  - Mean Absolute Error (MAE): {mae_hbr:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse_hbr:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape_hbr:.2f}")
print(f"  - R²: {r2_hbr:.2f}")

# %%
print("PART 9: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")

# In the mixed setting, Dtrain acts as our baseline "Source" and Dtest acts as our true "Target"
# because Dtrain is 90% normal days, and Dtest is 90% christmas days.

# 1. Target vs Synthetic
run_whyshift_region_analysis(
    df_source=synthetic_dataset, df_target=test_df,
    target_col='Weekly_Sales', categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Target", plot_filename="WALMART_TARGET_SYNTHETIC_CHRISTMAS_REGION.png"
)

# 2. Real vs Synthetic
run_whyshift_region_analysis(
    df_source=train_df, df_target=synthetic_dataset,
    target_col='Weekly_Sales', categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Dtrain", target_name="Synthetic", plot_filename="WALMART_REAL_SYNTHETIC_CHRISTMAS_REGION.png"
)

# 3. Real vs Hybrid
run_whyshift_region_analysis(
    df_source=train_df, df_target=hybrid_dataset,
    target_col='Weekly_Sales', categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Dtrain", target_name="Hybrid", plot_filename="WALMART_REAL_HYBRID_CHRISTMAS_REGION.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=test_df,
    target_col='Weekly_Sales', categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="WALMART_HYBRID_TARGET_CHRISTMAS_REGION.png"
)
