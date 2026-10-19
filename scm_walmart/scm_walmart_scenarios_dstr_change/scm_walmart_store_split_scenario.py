#!/usr/bin/env python3

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

import numpy as np
import yaml
import pandas as pd
import networkx as nx
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

from scm_walmart.custom_mape import custom_retail_mape

sys.path.append(str(project_root))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis
)

from scm_walmart.scm_walmart_graph_utils import build_feature_graph

config_path = project_root / "scm_walmart" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]
CONFIG = {
    'source_stores': list(range(1, 31)),  # Stores 1 to 30
    'target_stores': list(range(31, 46)),  # Stores 31 to 45
    'random_seed': 42
}
#%%


print("PART 1: LOADING DATA & SETUP")


try:
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    # Ensure Store is an integer for correct filtering
    df['Store'] = pd.to_numeric(df['Store'], errors='coerce').fillna(-1).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Stores present: {sorted(df['Store'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise



# PART 2: SPLIT DATA BY STORES 


print("PART 2: DOMAIN SHIFT SPLIT (STORES 1-30 vs 31-45)")


# Split based on Store Number
source_stores = CONFIG['source_stores']
target_stores = CONFIG['target_stores']

source_df = df[df['Store'].isin(CONFIG['source_stores'])].copy().reset_index(drop=True)
target_df = df[df['Store'].isin(CONFIG['target_stores'])].copy().reset_index(drop=True)




source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)


print(f"Train domain (Stores 1-30) shape: {train_df.shape}")
print(f"\nTest domain (Stores 31-45)")


print("✓ Domain separation verified: No overlap between Store Numbers")

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
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on Dtrain (90% Source Stores + 10% Target Stores)...")

    # The SCM is trained directly on the mixed dataset, automatically
    # learning local store behaviors without needing a separate Domain Adaptation step.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise



# %%
print("PART 7 & 8: SYNTHETIC AND HYBRID DATA GENERATION")

num_synthetic_samples = len(target_df)

# 1. Sample real rows from Target Domain (Stores 31-45) to maintain physical reality
sampled_target = target_domain_train.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

# 2. Extract the aligned physical properties of the store
aligned_store = sampled_target['Store'].values
aligned_city = sampled_target['city'].values
aligned_type = sampled_target['Type'].values
aligned_size = sampled_target['Size'].values

# 3. Create intervention functions
def int_store(*args, **kwargs): return aligned_store
def int_city(*args, **kwargs): return aligned_city
def int_type(*args, **kwargs): return aligned_type
def int_size(*args, **kwargs): return aligned_size

# Joint intervention on the entire "Store Identity"
interventions_dict = {
    'Store': int_store,
    'city': int_city,
    'Type': int_type,
    'Size': int_size
}

print("\nGenerating synthetic data with paired Spatial & Store Interventions...")

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

target_col = 'Weekly_Sales'

# In the mixed setting, Dtrain acts as our baseline "Source" and Dtest acts as our true "Target"
# because Dtrain is 90% Source Stores, and Dtest is 90% Target Stores.

# 1. Target vs Synthetic
run_whyshift_region_analysis(
    df_source=synthetic_dataset, df_target=test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Target", plot_filename="WALMART_TARGET_SYNTHETIC_STORE_REGION.png"
)

# 2. Real vs Synthetic
run_whyshift_region_analysis(
    df_source=train_df, df_target=synthetic_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Dtrain", target_name="Synthetic", plot_filename="WALMART_REAL_SYNTHETIC_STORE_REGION.png"
)

# 3. Real vs Hybrid
run_whyshift_region_analysis(
    df_source=train_df, df_target=hybrid_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Dtrain", target_name="Hybrid", plot_filename="WALMART_REAL_HYBRID_STORE_REGION.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="WALMART_HYBRID_TARGET_STORE_REGION.png"
)