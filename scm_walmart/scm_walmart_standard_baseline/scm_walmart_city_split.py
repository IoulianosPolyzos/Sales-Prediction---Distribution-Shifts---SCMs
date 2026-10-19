# %%
#!/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scm_walmart.custom_mape import *
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    save_checkpoint,
    load_checkpoint,
    get_all_descendants,
    setup_mechanisms,
    evaluate_sales_on_holdout
)

sys.path.append(str(current_dir.parent))
from scm_walmart.scm_walmart_graph_utils import build_feature_graph

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]


CONFIG = {
    'train_cities': ["Houston", "Philadelphia", "Phoenix", "San Jose", "Jacksonville","Austin"],
    'test_cities': ["New York", "Los Angeles", "Chicago"],
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_city',

}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

#%%
print("PART 1: LOADING DATA & SETUP")


try:
    CSV_PATH = r"../../walmart_dataset/final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)  # Keep original imputation strategy as requested

    print(f"Dataset shape: {df.shape}")
    print(f"Unique cities: {df['city'].nunique()}")
    print(f"Cities: {sorted(df['city'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY CITIES)")


train_cities = CONFIG['train_cities']
test_cities = CONFIG['test_cities']

train_df = df[df['city'].isin(train_cities)].copy().reset_index(drop=True)
test_df_full = df[df['city'].isin(test_cities)].copy().reset_index(drop=True)

# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)

adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train cities: {train_cities}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest cities: {test_cities}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify no overlap
assert set(train_cities).isdisjoint(set(test_cities)), "Train/Test cities overlap!"
print(" No overlap between TRAIN/TEST cities")
#%%
print("PART 3: PREPROCESSING")


try:

    # Initial confirmation that city-based feature (city) has been encoded.
    # This assumes prior transformation ensuring consistency across datasets.


    # Define categorical columns that should not be coerced into numeric types.
    # These variables represent discrete entities or labels and must be preserved as categorical.
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Attempt to convert all non-categorical columns to numeric format.
    # This enforces type.sh consistency and ensures compatibility with downstream statistical models.
    for col in train_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                train_df[col] = pd.to_numeric(train_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                # Log conversion failures without interrupting execution,
                # allowing inspection of problematic features.
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Define binary variables.
    # These are treated as categorical (string-encoded) to allow classification-based mechanisms
    # within the Structural Causal Model (SCM).
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]

    # Convert binary variables to string-based categorical representation.
    # This avoids unintended ordinal interpretations and aligns with probabilistic classifiers.
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Define multi-class categorical variables.
    # These variables encode nominal information with no inherent ordering.
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]

    # Ensure consistent string typing across all datasets.
    # This guarantees stable encoding and avoids mismatches during model fitting or inference.
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    # Aggregate all nodes that will be modeled using classification mechanisms in the SCM.
    # This distinction is critical for assigning appropriate structural equations.
    classifier_nodes = binary_nodes + categorical_nodes

    print(f" Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    # Global exception handling to ensure traceability of preprocessing failures.
    print(f"ERROR: Preprocessing failed: {e}")
    raise
#%%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# SURGICAL REMOVAL FOR CITY SHIFT EXPERIMENT
# Remove direct causal shortcut to enforce mediated pathways through Store_Metadata / Store_Activity
if feature_graph.has_edge('city', 'Weekly_Sales'):
    feature_graph.remove_edge('city', 'Weekly_Sales')

# Identify causal descendants impacted by the City domain shift
affected_groups = get_all_descendants(macro_graph, "city")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of City: {set(macro_graph.successors('city'))}")
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
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
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
        sorted(list(feature_graph.predecessors("Weekly_Sales"))),
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
    baseline_scm, _ = load_checkpoint(scm_path_initial, metadata_path_initial)

    # Establish consistent evaluation matrix
    comparison_df = test_sample.copy()

    baseline_metrics = evaluate_sales_on_holdout(
        baseline_scm, feature_graph, comparison_df, "BASELINE SCM",target_col="Weekly_Sales"
    )

    adapted_metrics = evaluate_sales_on_holdout(
        scm, feature_graph, comparison_df, "ADAPTED SCM", target_col="Weekly_Sales"
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