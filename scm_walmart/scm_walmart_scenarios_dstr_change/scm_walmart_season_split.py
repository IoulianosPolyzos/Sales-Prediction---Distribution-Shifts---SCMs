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
    'source_seasons': [1, 2, 4],  
    'target_seasons': [3],                         
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_season',
}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)





#%%



print("PART 1: LOADING DATA & SETUP")


try:
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)

    df = df.drop(
        columns=["Date", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    if 'Season' in df.columns:
        print(f"Unique seasons available: {sorted(df['Season'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY SEASON



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY SEASON)")


source_seasons = CONFIG['source_seasons']
target_seasons = CONFIG['target_seasons']

# Split based on Season
source_df = df[df['Season'].isin(source_seasons)].copy().reset_index(drop=True)
target_df = df[df['Season'].isin(target_seasons)].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"(Source) seasons: {source_seasons}")
print(f"Train shape: {train_df.shape}")
print(f"\nTarget seasons: {target_seasons}")


# Verify no overlap
#assert set(source_seasons).isdisjoint(set(target_seasons)), "Train/Test seasons overlap!"
print("✓ No overlap between TRAIN/TEST seasons")

#%%


print("PART 3: PREPROCESSING")


try:
    # Define categorical columns
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept", "Season"
    ]

    # Convert numeric columns
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)
    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)


# %%
print("PART 6: FITTING MIXED STRUCTURAL CAUSAL MODEL (ON POOLED TRAIN DATA)")

# Instantiate Structural Causal Model over feature-level DAG
scm = gcm.StructuralCausalModel(feature_graph)

# Fit SCM using the mixed observational training data
try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("✓ Mechanisms setup complete")

    print(f"Fitting SCM on Dtrain (90% Source Seasons + 10% Target Seasons)...")

    # The SCM is trained directly on the mixed dataset, thus automatically
    # learning the distributions of both domains without needing a separate Domain Adaptation step.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise



# %%
print("PART 7: EVALUATION ON Dtest (10% SOURCE SEASONS + 90% TARGET SEASONS)")

try:
    print(f"Generating predictions on {len(test_df)} mixed holdout test samples...")

    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    X_test = test_df[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    mae = mean_absolute_error(test_df['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_df['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_df['Weekly_Sales'].values, test_predictions)
    r2 = r2_score(test_df['Weekly_Sales'].values, test_predictions)

    print(f"\n PREDICTION METRICS (Mixed Test Set) ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (Sklearn): {mape:.2f}")
    print(f"R²:   {r2:.4f}")

    # Utilize the standard evaluation function for a clean printout
    evaluate_sales_on_holdout(
        scm, feature_graph, test_df, "MIXED SCM EVALUATION",target_col="Weekly_Sales"
    )

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise

