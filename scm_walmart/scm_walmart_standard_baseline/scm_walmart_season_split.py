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
    'test_sample_size': 50000,
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
        columns=["Date", "DayOfWeek", "WeekOfYear","Month"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    if 'Season' in df.columns:
        print(f"Unique seasons available: {sorted(df['Season'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY Season



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY Season)")


source_seasons = CONFIG['source_seasons']
target_seasons = CONFIG['target_seasons']

# Split based on Season
train_df = df[df['Season'].isin(source_seasons)].copy().reset_index(drop=True)
test_df_full = df[df['Season'].isin(target_seasons)].copy().reset_index(drop=True)

# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train Seasons (Source): {source_seasons}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest Seasons (Target): {target_seasons}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify no overlap
assert set(source_seasons).isdisjoint(set(target_seasons)), "Train/Test seasons overlap!"
print(" No overlap between TRAIN/TEST seasons")

#%%
print("PART 3: PREPROCESSING")


try:
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
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
            adaptation_df[col] = adaptation_df[col].astype(str)
            holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f" Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

#%%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# SURGICAL EDGE REMOVAL FOR TEMPORAL CAUSAL VALIDATION
#
# We deliberately remove the direct edges from the root temporal features
# (Month/Week) to the target variable (Weekly_Sales).
#
# WHY: Since our intervention (domain shift) originates at the Month/Week level,
# leaving direct shortcuts to Sales would allow the SCM to lazily bypass the
# true physical mechanisms. By cutting these direct links but intentionally
# KEEPING the 'Season -> Weekly_Sales' edge intact, we force the causal
# information to flow through proper intermediate mediators (Date -> Season -> Sales).
# This ensures we test if the SCM has genuinely learned the structural mechanics
# of the market (i.e., how time shifts the season, and how the season drives sales).

edges_to_remove = [
    ('Month_sin', 'Weekly_Sales'),
    ('Month_cos', 'Weekly_Sales'),
    ('Season_sin', 'Weekly_Sales'),
    ('Season_cos', 'Weekly_Sales'),
    ('WeekOfYear_sin', 'Weekly_Sales'),
    ('WeekOfYear_cos', 'Weekly_Sales'),
    ('Is_Summer', 'Weekly_Sales'),
    ('Year_Progress', 'Weekly_Sales'),
    ('Month_Progress', 'Weekly_Sales')
]

for src, tgt in edges_to_remove:
    if feature_graph.has_edge(src, tgt):
        feature_graph.remove_edge(src, tgt)

# Identify causal descendants impacted by the domain shift
affected_groups = get_all_descendants(macro_graph, "Date_Features")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Date Features: {set(macro_graph.successors('Date_Features'))}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# Print all nodes and edges
print("\n--- FINAL FEATURE-LEVEL GRAPH NODES ---")
for node in feature_graph.nodes():
    print(node)

print("\n--- FINAL FEATURE-LEVEL GRAPH EDGES ---")
for source, target in feature_graph.edges():
    print(f"{source} -> {target}")


# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

# Instantiate Structural Causal Model over feature-level DAG
scm = gcm.StructuralCausalModel(feature_graph)

# Fit SCM using observational training data
try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("✓ Mechanisms setup complete")

    print(f"Fitting SCM on training data (Seasons {source_seasons})...")
    gcm.fit(scm, train_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Model persistence (reproducibility checkpoint)
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df)
        },
        CONFIG['checkpoint_dir'],
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")

# %%
print("PART 7: SUPERVISED DOMAIN ADAPTATION (CONCEPT & COVARIATE SHIFT)")

# Topological ordering ensures causal parents are adapted before their descendants
ordered_nodes = list(nx.topological_sort(feature_graph))

# Selection of nodes affected by distribution shift
nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]
nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to Target Season (learning new sales/weather behavior)...")

# Structural adaptation under seasonal domain shift
try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Concept & Covariate shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count
        },
        CONFIG['checkpoint_dir'],
        stage_name="domain_adapted"
    )

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

# %%
print("PART 8: EVALUATION ON STRICT HOLDOUT SET (AFTER ADAPTATION)")

try:
    test_sample_size = min(CONFIG['test_sample_size'], len(holdout_test_df))
    test_sample = holdout_test_df.sample(n=test_sample_size, random_state=CONFIG['random_seed']).reset_index(drop=True)

    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    X_test = test_sample[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    mae = mean_absolute_error(test_sample['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_sample['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_sample['Weekly_Sales'].values, test_predictions)
    r2 = r2_score(test_sample['Weekly_Sales'].values, test_predictions)

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
    # Load baseline (pre-adaptation) SCM
    baseline_scm, baseline_metadata = load_checkpoint(
        scm_path_initial,
        metadata_path_initial
    )

    # Use identical holdout dataset for fair comparison
    comparison_df = holdout_test_df.reset_index(drop=True)

    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm, feature_graph, comparison_df, "BASELINE SCM", target_col="Weekly_Sales"
    )

    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm,
        feature_graph,
        comparison_df,
        "ADAPTED SCM (after adaptation)",target_col="Weekly_Sales"
    )

    print("\n FAIR HOLDOUT COMPARISON ")
    print(f"{'Metric':<10} {'Baseline':<12} {'Adapted':<12} {'Delta':<12}")
    

    mae_delta = baseline_metrics_holdout["MAE"] - adapted_metrics_holdout["MAE"]
    rmse_delta = baseline_metrics_holdout["RMSE"] - adapted_metrics_holdout["RMSE"]
    mape_delta = baseline_metrics_holdout["MAPE"] - adapted_metrics_holdout["MAPE"]
    r2_delta = adapted_metrics_holdout["R2"] - baseline_metrics_holdout["R2"]

    print(f"{'MAE':<10} {baseline_metrics_holdout['MAE']:<12.4f} {adapted_metrics_holdout['MAE']:<12.4f} {mae_delta:+.4f}")
    print(f"{'RMSE':<10} {baseline_metrics_holdout['RMSE']:<12.4f} {adapted_metrics_holdout['RMSE']:<12.4f} {rmse_delta:+.4f}")
    print(f"{'MAPE':<10} {baseline_metrics_holdout['MAPE']:<15.4f} {adapted_metrics_holdout['MAPE']:<15.4f} {mape_delta:+.4f}")
    print(f"{'R2':<10} {baseline_metrics_holdout['R2']:<12.4f} {adapted_metrics_holdout['R2']:<12.4f} {r2_delta:+.4f}")

    if (adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and
        adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]):
        print("\n✓ Adaptation improved performance on the same holdout set.")
    else:
        print("\nWARNING: Adaptation did not clearly improve performance on the same holdout set.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise