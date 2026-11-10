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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

sys.path.append(str(project_root))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    save_checkpoint,
    load_checkpoint,
    evaluate_sales_on_holdout
)

from scm_corporacion_favorita.scm_favorita_graph_utils import build_feature_graph

config_path = project_root / "scm_corporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'random_seed': 42,
    'test_sample_size': 500000,
    'checkpoint_dir': './checkpoints_baseline_favorita',
}

Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = project_root / "corporacion_favorita_dataset" / "final_data_favorita.parquet"
    df = pd.read_parquet(PARQUET_PATH)

    df = df.drop(columns=["date", "id", "item_nbr", "Season", "DayOfWeek", "Month", "WeekOfYear"], errors='ignore')
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: RANDOM SPLIT 70/30")

train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=CONFIG['random_seed']
)

train_df = train_df.sample(n=min(1000000, len(train_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# %%
print("PART 3: PREPROCESSING")

try:
    categorical_cols_to_exclude = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
            except ValueError:
                pass

    binary_nodes = [
        "IsHoliday", "is_near_holiday", "Is_Summer",
        "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact","Is_Christmas_Season"
    ]

    for col in binary_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(int).astype(str)
            test_df[col] = test_df[col].astype(int).astype(str)

    categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
    for col in categorical_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            test_df[col] = test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f" Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, categorical_nodes, lgb_params)
    print("Fitting SCM on training data...")
    gcm.fit(scm, train_df)
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7: EVALUATION ON TEST SET (BASELINE SCM)")

try:
    test_sample_size = min(CONFIG.get('test_sample_size', 500000), len(test_df))
    test_sample = test_df.sample(n=test_sample_size, random_state=CONFIG['random_seed']).reset_index(drop=True)

    baseline_metrics = evaluate_sales_on_holdout(
        scm, feature_graph, test_sample, "BASELINE SCM", target_col="unit_sales"
    )

    print("\n BASELINE SCM PREDICTION METRICS")
    print(f"MAE:  {baseline_metrics['MAE']:.4f}")
    print(f"RMSE: {baseline_metrics['RMSE']:.4f}")
    print(f"MAPE: {baseline_metrics['MAPE']:.4f}")
    print(f"R²:   {baseline_metrics['R2']:.4f}")

except Exception as e:
    print(f"ERROR: Baseline evaluation failed: {e}")
    raise