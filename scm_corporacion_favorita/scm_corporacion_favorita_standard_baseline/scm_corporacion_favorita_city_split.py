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
    'train_cities': ["Quito", "Guayaquil", "Cuenca", "Santo Domingo", "Machala", "Loja", "Manta"],
    'random_seed': 42,
    'checkpoint_dir': './checkpoints_city_favorita',
}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Drop redundant temporal and identifier columns
    df = df.drop(columns=["date", "id", "item_nbr", "Season", "DayOfWeek", "Month", "WeekOfYear"], errors='ignore')
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Unique cities: {df['city'].nunique()}")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY CITIES)")

train_cities = CONFIG['train_cities']
test_cities = [c for c in df['city'].unique() if c not in train_cities]
CONFIG['test_cities'] = test_cities

# Split data based on city
train_df = df[df['city'].isin(train_cities)].copy().reset_index(drop=True)
test_df_full = df[df['city'].isin(test_cities)].copy().reset_index(drop=True)

# Subsample training data to ensure SCM computational feasibility
train_df = train_df.sample(n=min(1000000, len(train_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)

# Split target domain into Adaptation Set (for refit) and Holdout Set (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)

# Cap adaptation set size for rapid local refitting
adaptation_df = adaptation_df.sample(n=min(500000, len(adaptation_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train cities: {train_cities}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest cities: {test_cities}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify exact domain separation
assert set(train_cities).isdisjoint(set(test_cities)), "Train/Test cities overlap!"
print("✓ No overlap between TRAIN/TEST cities")

# %%
print("PART 3: PREPROCESSING")

try:
    categorical_cols_to_exclude = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

    # Convert continuous features to numeric
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError:
                pass

    # Cast boolean/binary indicators to string representation for classifier-based modeling
    binary_nodes = [
        "IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer",
        "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact"
    ]

    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Cast nominal categoricals to string representation
    categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# SURGICAL REMOVAL FOR CITY SHIFT EXPERIMENT
edges_to_remove = [
    ('city', 'unit_sales'),
    ('state', 'unit_sales')
]
for src, tgt in edges_to_remove:
    if feature_graph.has_edge(src, tgt):
        feature_graph.remove_edge(src, tgt)

# Identify causal descendants impacted by the Location domain shift
affected_groups = get_all_descendants(macro_graph, "Location")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Location: {set(macro_graph.successors('Location'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on training data (Source Cities)...")
    gcm.fit(scm, train_df)

    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {'status': 'initial_fit_complete', 'train_set_size': len(train_df)},
        CONFIG['checkpoint_dir'],
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

ordered_nodes = list(nx.topological_sort(feature_graph))
nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm, feature_graph,
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {'status': 'domain_adaptation_complete'},
        CONFIG['checkpoint_dir'],
        stage_name="domain_adapted"
    )
except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

# %%
print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")

try:
    test_sample = holdout_test_df.sample(n=min(50000, len(holdout_test_df)),
                                         random_state=CONFIG['random_seed']).reset_index(drop=True)

    sales_mechanism = scm.causal_mechanism("unit_sales")
    parents = sorted(list(feature_graph.predecessors("unit_sales")))

    X_test = test_sample[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    mae = mean_absolute_error(test_sample['unit_sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_sample['unit_sales'].values, test_predictions))
    mape = mean_absolute_percentage_error(test_sample['unit_sales'].values, test_predictions)
    r2 = r2_score(test_sample['unit_sales'].values, test_predictions)

    print(f"\n PREDICTION METRICS (Holdout Set - After Adaptation) ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (Sklearn): {mape:.2f}")
    print(f"R²:   {r2:.4f}")

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise

# %%
print("PART 9: FAIR COMPARISON - BASELINE vs ADAPTED ON SAME HOLDOUT SET")

try:
    baseline_scm, _ = load_checkpoint(scm_path_initial, metadata_path_initial)

    # Establish consistent evaluation matrix
    comparison_df = test_sample.copy()

    baseline_metrics = evaluate_sales_on_holdout(
        baseline_scm, feature_graph, comparison_df, "BASELINE SCM", target_col="unit_sales"
    )
    adapted_metrics = evaluate_sales_on_holdout(
        scm, feature_graph, comparison_df, "ADAPTED SCM", target_col="unit_sales"
    )

    print("\nFAIR HOLDOUT COMPARISON (Target Cities)")
    print(f"{'Metric':<10} {'Baseline':<15} {'Adapted':<15} {'Delta':<15}")
    

    mae_delta = baseline_metrics["MAE"] - adapted_metrics["MAE"]
    rmse_delta = baseline_metrics["RMSE"] - adapted_metrics["RMSE"]
    mape_delta = baseline_metrics["MAPE"] - adapted_metrics["MAPE"]
    r2_delta = adapted_metrics["R2"] - baseline_metrics["R2"]

    print(f"{'MAE':<10} {baseline_metrics['MAE']:<15.4f} {adapted_metrics['MAE']:<15.4f} {mae_delta:+.4f}")
    print(f"{'MAPE':<10} {baseline_metrics['MAPE']:<15.4f} {adapted_metrics['MAPE']:<15.4f} {mape_delta:+.4f}")
    print(f"{'RMSE':<10} {baseline_metrics['RMSE']:<15.4f} {adapted_metrics['RMSE']:<15.4f} {rmse_delta:+.4f}")
    print(f"{'R2':<10} {baseline_metrics['R2']:<15.4f} {adapted_metrics['R2']:<15.4f} {r2_delta:+.4f}")

except Exception as e:
    print(f"ERROR: Fair comparison failed: {e}")
    raise