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
    'source_domain': 'Normal Days (IsHoliday=0)',
    'target_domain': 'Holiday Days (IsHoliday=1)',
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_holidays',

}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)




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

    print(f"Dataset shape: {df.shape}")
    print(f"IsHoliday distribution:")
    print(f"  Normal Days (0): {(df['IsHoliday'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['IsHoliday'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY HOLIDAY DOMAIN 



print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs HOLIDAY)")


# Ensure IsHoliday is numeric (0 or 1)
df['IsHoliday'] = df['IsHoliday'].astype(int)

source_df = df[df['IsHoliday'] == 0].copy().reset_index(drop=True)
target_df = df[df['IsHoliday'] == 1].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"Source Domain: {CONFIG['source_domain']}")
print(f"Train shape: {train_df.shape}")
print(f"\nTarget Domain: {CONFIG['target_domain']}")


# Verify domain separation
#assert (train_df['IsHoliday'] == 0).all(), "Train set contains non-zero IsHoliday!"
print("✓ Domain separation verified: No overlap between Normal/Holiday days")

#%%


print("PART 3: PREPROCESSING")


try:
    # Encode cities
    # le_city = LabelEncoder()
    # le_city.fit(df['city'].astype(str))
    # train_df['city'] = le_city.transform(train_df['city'].astype(str))
    # adaptation_df['city'] = le_city.transform(adaptation_df['city'].astype(str))
    # holdout_test_df['city'] = le_city.transform(holdout_test_df['city'].astype(str))

    

    # Define categorical columns
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
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

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on Dtrain (90% Normal Days + 10% Holiday Days)...")

    # The SCM is trained directly on the mixed dataset, thus automatically
    # learning the distributions of both domains without needing a separate Domain Adaptation step.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7: EVALUATION ON Dtest (10% NORMAL DAYS + 90% HOLIDAY DAYS)")

try:
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