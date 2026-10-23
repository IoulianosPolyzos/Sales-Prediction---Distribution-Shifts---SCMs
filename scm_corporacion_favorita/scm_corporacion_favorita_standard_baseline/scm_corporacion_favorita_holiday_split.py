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
    'train_domain': 'Normal Days (IsHoliday=0)',
    'test_domain': 'Holiday Days (IsHoliday=1)',
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_holidays',
}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # SCMs train a separate ML model for EVERY node in the graph.
    # We must sample the data to make SCM fitting computationally feasible.
    print(f"Original shape: {df.shape}. Sampling 200,000 rows for SCM modeling...")
    df = df.sample(n=200000, random_state=42).reset_index(drop=True)

    # Drop redundant raw identifiers and unencoded temporal variables.
    df = df.drop(
        columns=["date", "id", "item_nbr", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape after preprocessing: {df.shape}")
    print(f"IsHoliday distribution:")
    print(f"  Normal Days (0): {(df['IsHoliday'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['IsHoliday'] == 1).sum()} records")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: DOMAIN SHIFT SPLIT (NORMAL vs HOLIDAY)")

# Ensure IsHoliday is numeric (0 or 1)
df['IsHoliday'] = df['IsHoliday'].astype(int)

train_df = df[df['IsHoliday'] == 0].copy().reset_index(drop=True)
train_df = train_df.sample(n=min(1000000, len(train_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
test_df_full = df[df['IsHoliday'] == 1].copy().reset_index(drop=True)

# Split target domain into Adaptation Set (for refit) and Holdout Set (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain: {CONFIG['train_domain']}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest domain: {CONFIG['test_domain']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify domain separation
assert (train_df['IsHoliday'] == 0).all(), "Train set contains non-zero IsHoliday!"
assert (adaptation_df['IsHoliday'] == 1).all(), "Adaptation set contains non-holiday days!"
assert (holdout_test_df['IsHoliday'] == 1).all(), "Holdout set contains non-holiday days!"
print("✓ Domain separation verified: No overlap between Normal/Holiday days")

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
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Binary encoding
    binary_nodes = [
        "IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer",
        "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact"
    ]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Categorical encoding
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

# SURGICAL REMOVAL FOR HOLIDAY SHIFT EXPERIMENT
# Prevent direct causal shortcut for strict mechanistic modeling
edges_to_remove = [
    ('Month_sin', 'unit_sales'),
    ('Month_cos', 'unit_sales'),
    ('WeekOfYear_sin', 'unit_sales'),
    ('WeekOfYear_cos', 'unit_sales'),
    ('IsHoliday', 'unit_sales'),
    ('Year_Progress', 'unit_sales'),
    ('Month_Progress', 'unit_sales')
]

for src, tgt in edges_to_remove:
    if feature_graph.has_edge(src, tgt):
        feature_graph.remove_edge(src, tgt)

# Identify causal descendants impacted by the Holiday domain shift
affected_groups = get_all_descendants(macro_graph, "Holidays")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Holidays: {set(macro_graph.successors('Holidays'))}")
print(f"All affected groups (transitive): {affected_groups}")
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

    print("Fitting SCM on training data...")
    gcm.fit(scm, train_df)
    print("✓ Initial fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Save trained SCM checkpoint
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df),
            'train_domain': CONFIG['train_domain']
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

nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]
nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Features + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to holiday domain (learning new sales behavior during holidays)...")

# Selective retraining of structural mechanisms using adaptation dataset
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
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,
            'adaptation_domain': CONFIG['test_domain']
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
    test_sample = holdout_test_df.sample(
        n=test_sample_size,
        random_state=CONFIG['random_seed']
    ).reset_index(drop=True)

    if len(test_sample) < CONFIG['test_sample_size']:
        print(
            f"WARNING: Test sample size ({len(test_sample)}) < "
            f"configured size ({CONFIG['test_sample_size']}). "
            f"Results may have higher variance."
        )

    print(f"Generating predictions on {len(test_sample)} holdout test samples...")

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
    # Load baseline (pre-adaptation) SCM
    baseline_scm, baseline_metadata = load_checkpoint(
        scm_path_initial,
        metadata_path_initial
    )

    # Use identical holdout dataset for fair comparison
    comparison_df = holdout_test_df.reset_index(drop=True)

    baseline_metrics_holdout = evaluate_sales_on_holdout(
        baseline_scm, feature_graph, comparison_df, "BASELINE SCM", target_col="unit_sales"
    )
    adapted_metrics_holdout = evaluate_sales_on_holdout(
        scm, feature_graph, comparison_df, "ADAPTED SCM", target_col="unit_sales"
    )

    # Side-by-side comparative analysis
    print("\nFAIR HOLDOUT COMPARISON (Holiday Days)")
    print(f"{'Metric':<10} {'Baseline':<15} {'Adapted':<15} {'Delta':<15}")
    

    mae_delta = baseline_metrics_holdout["MAE"] - adapted_metrics_holdout["MAE"]
    rmse_delta = baseline_metrics_holdout["RMSE"] - adapted_metrics_holdout["RMSE"]
    mape_delta = baseline_metrics_holdout["MAPE"] - adapted_metrics_holdout["MAPE"]
    r2_delta = adapted_metrics_holdout["R2"] - baseline_metrics_holdout["R2"]

    print(f"{'MAE':<10} {baseline_metrics_holdout['MAE']:<15.4f} {adapted_metrics_holdout['MAE']:<15.4f} {mae_delta:+.4f}")
    print(f"{'RMSE':<10} {baseline_metrics_holdout['RMSE']:<15.4f} {adapted_metrics_holdout['RMSE']:<15.4f} {rmse_delta:+.4f}")
    print(f"{'MAPE':<10} {baseline_metrics_holdout['MAPE']:<15.4f} {adapted_metrics_holdout['MAPE']:<15.4f} {mape_delta:+.4f}")
    print(f"{'R2':<10} {baseline_metrics_holdout['R2']:<15.4f} {adapted_metrics_holdout['R2']:<15.4f} {r2_delta:+.4f}")

    if (adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"] and
        adapted_metrics_holdout["RMSE"] < baseline_metrics_holdout["RMSE"]):
        print("\n✓ Adaptation improved performance on holiday days.")
        print("  The model successfully learned shifted sales dynamics under intervention-like conditions.")
    else:
        print("\nWARNING: Adaptation did not clearly improve performance on holiday days.")
        print("  This may indicate:")
        print("  (i) weak distribution shift between domains,")
        print("  (ii) insufficient adaptation sample size, or")
        print("  (iii) stable causal mechanisms across domains.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise