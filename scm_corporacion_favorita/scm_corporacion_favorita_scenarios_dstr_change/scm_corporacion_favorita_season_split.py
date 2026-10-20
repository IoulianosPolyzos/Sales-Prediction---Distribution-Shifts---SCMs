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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target



sys.path.append(str(project_root))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis
)

from scm_corporacion_favorita.scm_favorita_graph_utils import build_feature_graph

config_path = project_root / "scm_corporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_seasons': [1, 2, 4],  
    'target_seasons': [3],                         
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_season',
}



# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # We do NOT drop the raw integer 'Season' column here because we need it for the split.
    # It will be dropped AFTER the split to avoid interfering with SCM's Season_sin/Season_cos.
    df = df.drop(
        columns=["date", "id", "item_nbr", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    if 'Season' in df.columns:
        print(f"Unique seasons available: {sorted(df['Season'].unique())}")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: DOMAIN SHIFT SPLIT (MIXED TRAIN/TEST BY SEASON)")

source_seasons = CONFIG['source_seasons']
target_seasons = CONFIG['target_seasons']

# Extract domains based on Season
source_df = df[df['Season'].isin(source_seasons)].copy().reset_index(drop=True)
target_df = df[df['Season'].isin(target_seasons)].copy().reset_index(drop=True)

# 1. Subsample to avoid memory crashes during SCM fit
source_df = source_df.sample(n=min(50000000, len(source_df)), random_state=CONFIG['random_seed'])
target_df = target_df.sample(n=min(50000000, len(target_df)), random_state=CONFIG['random_seed'])
# 2. Split each domain to construct the pooled dataset
# Source: 90% goes to train, 10% goes to test
source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=CONFIG['random_seed']
)

# Target: 10% goes to train, 90% goes to test
target_domain_train, target_domain_test = train_test_split(
    target_df, test_size=0.9, random_state=CONFIG['random_seed']
)

# 3. Create Pooled (Mixed) Train and Test sets
train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=CONFIG[
    'random_seed']).reset_index(drop=True)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=CONFIG[
    'random_seed']).reset_index(drop=True)

print(f"\nDtrain (90% Source Seasons + 10% Target Seasons) shape: {train_df.shape}")
print(f"  - Source instances: {len(source_domain_train)}")
print(f"  - Target instances: {len(target_domain_train)}")

print(f"\nDtest (10% Source Seasons + 90% Target Seasons) shape: {test_df.shape}")
print(f"  - Source instances: {len(source_domain_test)}")
print(f"  - Target instances: {len(target_domain_test)}")

# Verify no overlap in seasons (conceptually, though they are mixed now)
# Drop the raw integer 'Season' column now that the split is complete
train_df = train_df.drop(columns=["Season"], errors="ignore")
test_df = test_df.drop(columns=["Season"], errors="ignore")

# %%
print("PART 3: PREPROCESSING")

try:
    # Define categorical variables that should be excluded from numeric coercion.
    categorical_cols_to_exclude = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

    # Convert numeric columns
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError:
                pass

    # Cast boolean/binary indicators to string representation for classifier-based modeling
    binary_nodes = [
        "IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer",
        "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact"
    ]

    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    # Cast nominal categoricals to string representation
    categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
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

# Since there are source data in train set , there is no need to remove the edge between
# domain split variable or the intervention variable and sales node

# Print all nodes and edges
print("\n--- FINAL FEATURE-LEVEL GRAPH NODES ---")
for node in feature_graph.nodes():
    print(node)

print("\n--- FINAL FEATURE-LEVEL GRAPH EDGES ---")
for source, target in feature_graph.edges():
    print(f"{source} -> {target}")


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

    sales_mechanism = scm.causal_mechanism("unit_sales")
    parents = sorted(list(feature_graph.predecessors("unit_sales")))

    X_test = test_df[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    mae = mean_absolute_error(test_df['unit_sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_df['unit_sales'].values, test_predictions))
    mape = mean_absolute_percentage_error(test_df['unit_sales'].values, test_predictions)
    r2 = r2_score(test_df['unit_sales'].values, test_predictions)

    print(f"\n PREDICTION METRICS (Mixed Test Set) ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (Sklearn): {mape:.2f}")
    print(f"R²:   {r2:.4f}")

    # Utilize the standard evaluation function for a clean printout
    evaluate_sales_on_holdout(
        scm, feature_graph, test_df, "MIXED SCM EVALUATION", target_col="unit_sales"
    )

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise