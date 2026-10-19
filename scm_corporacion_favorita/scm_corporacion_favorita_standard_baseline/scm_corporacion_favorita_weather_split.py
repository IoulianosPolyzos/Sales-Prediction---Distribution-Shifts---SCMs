# %%
#!/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

# --- PATH MANAGEMENT FOR EXTERNAL UTILITIES ---
current_dir = Path(__file__).resolve().parent

# 1. utils_scm is located two levels up (root)
sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    save_checkpoint,
    load_checkpoint,
    get_all_descendants,
    setup_mechanisms,
    evaluate_sales_on_holdout
)

# 2. scm_favorita_graph_utils is located one level up
sys.path.append(str(current_dir.parent))
from scm_corporacion_favorita.scm_favorita_graph_utils import build_feature_graph

# --- CONFIGURATION ---
config_path = project_root / "scm_orporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_weather': ["Rain", "Snow", "Clouds"],
    'target_weather': ["Clear"],
    'random_seed': 42,
    'test_sample_size': 50000,
    'checkpoint_dir': './checkpoints_weather_favorita',
}

# Create checkpoint directory
Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True, parents=True)

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Drop redundant columns
    df = df.drop(
        columns=["date", "id", "item_nbr", "DayOfWeek", "Month", "WeekOfYear", "Season"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Unique weather: {df['weather_condition'].nunique()}")
    print(f"Weather: {sorted(df['weather_condition'].unique())}")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY weather_conditions)")

source_weather = CONFIG['source_weather']
target_weather = CONFIG['target_weather']

# Split based on Weather Condition
train_df = df[df['weather_condition'].isin(source_weather)].copy().reset_index(drop=True)
test_df_full = df[df['weather_condition'].isin(target_weather)].copy().reset_index(drop=True)

# Subsample training data to ensure SCM computational feasibility
train_df = train_df.sample(n=min(5000000, len(train_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)

# Split Test Data into ADAPTATION SET (for refit) and HOLDOUT SET (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    test_df_full,
    test_size=0.90,  # 10% adaptation, 90% holdout
    random_state=CONFIG['random_seed']
)

# Cap the adaptation set size to ensure rapid refitting
adaptation_df = adaptation_df.sample(n=min(20000, len(adaptation_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train weather conditions: {source_weather}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest weather conditions: {target_weather}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify no overlap
assert set(source_weather).isdisjoint(set(target_weather)), "Train/Test weather conditions overlap!"
print("✓ No overlap between TRAIN/TEST weather_conditions")

# %%
print("PART 3: PREPROCESSING")

try:
    # Define categorical columns
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
            except ValueError:
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
        if col in train_df.columns:
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

# SURGICAL REMOVAL FOR WEATHER SHIFT EXPERIMENT
# Remove direct edge to enforce mediated pathways
if feature_graph.has_edge('weather_condition', 'unit_sales'):
    feature_graph.remove_edge('weather_condition', 'unit_sales')

# Identify causal descendants impacted by the Weather domain shift
affected_groups = get_all_descendants(macro_graph, "Weather")

affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Weather: {set(macro_graph.successors('Weather'))}")
print(f"All affected groups (transitive): {affected_groups}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

print("\n Causal interpretation:")
print("  Weather differences propagate through the causal structure inducing")
print("  distribution shifts in downstream variables.")

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

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("✓ Mechanisms setup complete")

    print(f"Fitting SCM on training data ({','.join(CONFIG['source_weather'])})...")
    gcm.fit(scm, train_df)
    print("✓ Initial structural fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# Model checkpointing
try:
    scm_path_initial, metadata_path_initial = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {
            'status': 'initial_fit_complete',
            'train_set_size': len(train_df),
            'train_domain': f"{','.join(CONFIG['source_weather'])}"
        },
        CONFIG['checkpoint_dir'],
        stage_name="initial_fit"
    )
except Exception as e:
    print(f"ERROR: Could not save initial checkpoint: {e}")

# %%
print("PART 7: SUPERVISED DOMAIN ADAPTATION FOR NEW WEATHER CONDITIONS")

ordered_nodes = list(nx.topological_sort(feature_graph))

# Selection of nodes affected by structural intervention (Weather descendants)
nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]
nodes_to_refit = list(nodes_to_refit)

print(f"\nNodes to refit (Weather Downstream + TARGET included - {len(nodes_to_refit)} nodes):")
for node in sorted(nodes_to_refit):
    print(f"   {node}")

print(f"\nAdapting mechanisms to new weather conditions (learning new local behaviors)...")

try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            print(f"  Refitting {node}...")
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception as e:
            print(f"WARNING: Failed to refit {node}: {e}")

    print(f"\n Shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

    scm_path_adapted, metadata_path_adapted = save_checkpoint(
        scm,
        feature_graph,
        sorted(list(feature_graph.predecessors("unit_sales"))),
        {
            'status': 'domain_adaptation_complete',
            'nodes_refitted': refit_count,
            'adaptation_domain': f"{','.join(CONFIG['target_weather'])}"
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
    # Load baseline SCM (pre-domain adaptation model)
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

    # Side-by-side comparison table
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

    if adapted_metrics_holdout["R2"] > baseline_metrics_holdout["R2"]:
        print("\n✓ Adaptation successfully captured the Weather Shift.")
    else:
        print("\n⚠ WARNING: Adaptation did not clearly improve performance.")

except Exception as e:
    print(f"ERROR: Fair baseline vs adapted comparison failed: {e}")
    raise