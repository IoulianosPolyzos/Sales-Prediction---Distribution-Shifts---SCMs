#!/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# --- Setup project paths ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from scm_walmart.custom_mape import *

import dowhy.gcm as gcm

# Ignore warnings for cleaner output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parents[1]))

from utils_scm import (
    save_checkpoint,
    load_checkpoint,
    setup_mechanisms,
    evaluate_sales_on_holdout
)

sys.path.append(str(current_dir.parent))
from scm_walmart.scm_walmart_graph_utils import build_feature_graph

# --- Load configurations ---
config_path = project_root / "scm_walmart" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_baseline_walmart',
}

# Create checkpoint directory if it does not exist
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)

    # Drop unnecessary columns
    df = df.drop(
        columns=["Date", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise

# %%
print("PART 2: RANDOM SPLIT 70/30")

# Perform a random split of the data into 70% training and 30% testing
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=CONFIG['random_seed']
)

# Reset indices for both dataframes
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# %%
print("PART 3: PREPROCESSING")

try:
    # Define categorical columns to exclude from numeric conversion
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Convert remaining columns to numeric
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding: Convert binary columns to string categorical representation
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start",
                    "Is_Month_End"]
    for col in binary_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(int).astype(str)
            test_df[col] = test_df[col].astype(int).astype(str)

    # Categorical encoding: Ensure text/categorical nodes are of type string
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the feature-level directed acyclic graph (DAG)
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Initialize the Structural Causal Model with the built graph
scm = gcm.StructuralCausalModel(feature_graph)

try:
    # Assign mechanisms (e.g., Regressors and Classifiers) to the graph nodes
    setup_mechanisms(scm, feature_graph, classifier_nodes, categorical_nodes, lgb_params)
    print("Fitting SCM on training data...")
    gcm.fit(scm, train_df)
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7: EVALUATION ON TEST SET (BASELINE SCM)")

try:
    # Sample from the test set for evaluation
    test_sample_size = min(CONFIG.get('test_sample_size', 50000), len(test_df))
    test_sample = test_df.sample(n=test_sample_size, random_state=CONFIG['random_seed']).reset_index(drop=True)

    # Evaluate predictive performance on target column
    baseline_metrics = evaluate_sales_on_holdout(
        scm, feature_graph, test_sample, "BASELINE SCM", target_col="Weekly_Sales"
    )

    print("\nBASELINE SCM PREDICTION METRICS")
    print(f"MAE:  {baseline_metrics['MAE']:.4f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.4f}")
    print(f"MAPE: {baseline_metrics['MAPE']:.4f}")
    print(f"R²:   {baseline_metrics['R2']:.4f}")

except Exception as e:
    print(f"ERROR: Baseline evaluation failed: {e}")
    raise

# %%
print("PART 8: CAUSAL GRAPH EVALUATION (FALSIFICATION)")

try:
    # Evaluate the structural and mechanism assumptions of the SCM
    print("Running causal model evaluation (this may take some time)...")
    evaluation_metrics = gcm.evaluate_causal_model(
        scm,
        test_df,
        evaluate_causal_mechanisms=True,
        evaluate_causal_structure=True,
        max_num_samples=-1
    )

    # Plot evaluation results and save to file
    gcm.falsify.plot_evaluation_results(evaluation_metrics.graph_falsification)
    plt.savefig("graph_falsification.png", dpi=150, bbox_inches='tight')
    print("Graph falsification plot saved successfully to 'graph_falsification.png'")

except Exception as e:
    print(f"ERROR: Causal evaluation/plotting failed: {e}")