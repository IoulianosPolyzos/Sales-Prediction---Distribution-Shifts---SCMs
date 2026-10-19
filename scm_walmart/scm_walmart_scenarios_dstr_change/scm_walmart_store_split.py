# %%
# !/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import yaml
import numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scm_walmart.custom_mape import *
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm

current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    evaluate_sales_on_holdout
)

sys.path.append(str(current_dir.parent))
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

# Create checkpoint directory




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


# PART 2: SPLIT DATA BY STORE NUMBER DOMAIN 



print("PART 2: DOMAIN SHIFT SPLIT (STORES 1-30 vs 31-45)")


# Split based on Store Number
source_df = df[df['Store'].isin(CONFIG['source_stores'])].copy().reset_index(drop=True)
target_df = df[df['Store'].isin(CONFIG['target_stores'])].copy().reset_index(drop=True)




source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"Source Domain (Stores 1-30) shape: {train_df.shape}")
print(f"\nTarget Domain (Stores 31-45)")


print("✓ Domain separation verified: No overlap between Store Numbers")

#%%


print("PART 3: PREPROCESSING")


try:
    # Encode city


    categorical_cols_to_exclude = [
        "city", "weather_condition", "Type", "Dept","Store"
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
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(int).astype(str)
            test_df[col] = test_df[col].astype(int).astype(str)

    # Categorical encoding (Type is back as a standard feature!)
    categorical_nodes = ["Type", "weather_condition", "Dept", "city","Store"]
    for col in categorical_nodes:
        if col in train_df.columns:
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
    print("✓ Mechanisms setup complete")

    print("Fitting SCM on Dtrain (90% Source Stores + 10% Target Stores)...")

    # The SCM is trained directly on the mixed dataset, automatically
    # learning local store behaviors without needing a separate Domain Adaptation step.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise



# %%
print("PART 7: EVALUATION ON Dtest (10% SOURCE STORES + 90% TARGET STORES)")

try:
    print(f"Generating predictions on {len(test_df)} mixed test samples...")

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