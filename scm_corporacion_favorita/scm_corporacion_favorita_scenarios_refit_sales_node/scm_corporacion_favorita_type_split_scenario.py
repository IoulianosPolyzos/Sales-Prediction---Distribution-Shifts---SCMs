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
import itertools
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
    'source_store_types': ['C', 'B', 'E', 'D'],  # Source Domain
    'target_store_types': ['A'],  # Target Domain
    'random_seed': 42
}



# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Drop redundant temporal and identifier columns
    df = df.drop(
        columns=["date", "id", "item_nbr", "DayOfWeek", "Month", "WeekOfYear", "Season"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Store Type distribution:")
    for store_type in sorted(df['store_type'].astype(str).unique()):
        print(f"  Type {store_type}: {(df['store_type'] == store_type).sum()} records")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("PART 2: DOMAIN SHIFT SPLIT (TYPE C,B,E,D vs TYPE A)")

# Extract Source & Target Domains based on Store Type
source_df = df[df['store_type'].isin(CONFIG['source_store_types'])].copy().reset_index(drop=True)
target_df = df[df['store_type'].isin(CONFIG['target_store_types'])].copy().reset_index(drop=True)

# Subsample source data to ensure SCM computational feasibility
source_df = source_df.sample(n=min(500000, len(source_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
# Split target domain into Adaptation Set (for refit) and Holdout Set (for evaluation)
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,  # 10% adaptation, 90% holdout
    random_state=CONFIG['random_seed']
)


adaptation_df = adaptation_df.sample(n=min(100000, len(adaptation_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source domain (Types {CONFIG['source_store_types']}) shape: {source_df.shape}")
print(f"Target domain (Types {CONFIG['target_store_types']})")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify exact domain separation
assert not source_df['store_type'].isin(CONFIG['target_store_types']).any(), "Train set contains Target Store Types!"
assert adaptation_df['store_type'].isin(CONFIG['target_store_types']).all(), "Adaptation set contains Source Store Types!"
assert holdout_test_df['store_type'].isin(CONFIG['target_store_types']).all(), "Holdout set contains Source Store Types!"
print("✓ Domain separation verified")

# %%
print("PART 3: PREPROCESSING")

try:
    categorical_cols_to_exclude = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

    # Convert continuous features to numeric
    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
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
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
    for col in categorical_nodes:
        if col in source_df.columns:
            source_df[col] = source_df[col].astype(str)
            adaptation_df[col] = adaptation_df[col].astype(str)
            holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(source_df.columns)

# Surgical Removal: Prevent direct causal shortcut for strict mechanistic modeling
edges_to_remove = [
    ('city', 'unit_sales'),
    ('state', 'unit_sales'),
    ('store_type','unit_sales'),
    ('store_nbr','unit_sales'),
    ('cluster', 'unit_sales')
]


for src, tgt in edges_to_remove:
    if feature_graph.has_edge(src, tgt):
        feature_graph.remove_edge(src, tgt)

# Identify causal descendants impacted by the Store Type domain shift
affected_groups = get_all_descendants(macro_graph, "Store_Metadata")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Store_Metadata: {set(macro_graph.successors('Store_Metadata'))}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print(f"Fitting SCM on training data (Store Types {CONFIG['source_store_types']})...")
    gcm.fit(scm, source_df)

    print("✓ Initial fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("PART 7: SUPERVISED DOMAIN ADAPTATION")

ordered_nodes = list(nx.topological_sort(feature_graph))
nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]

print(f"\nAdapting mechanisms to adaptation set ({len(nodes_to_refit)} nodes)...")
try:
    refit_count = 0
    for node in nodes_to_refit:
        try:
            fit_causal_model_of_target(scm, node, adaptation_df)
            refit_count += 1
        except Exception:
            pass
    print(f"✓ Concept shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")

except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

# %%
print("PART 8 & 9: SYNTHETIC AND HYBRID DATA GENERATION")

num_synthetic_samples = min(1000000, len(target_df))

# 1. Sample real rows from Target Domain (Stores 31-45) to maintain physical reality
sampled_target = adaptation_df.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

# 2. Extract the aligned physical properties of the store type
aligned_store = sampled_target['store_nbr'].values
aligned_city = sampled_target['city'].values
aligned_state = sampled_target['state'].values
aligned_type = sampled_target['store_type'].values
aligned_cluster = sampled_target['cluster'].values

iter_store = itertools.cycle(aligned_store)
iter_city = itertools.cycle(aligned_city)
iter_state = itertools.cycle(aligned_state)
iter_type = itertools.cycle(aligned_type)
iter_cluster = itertools.cycle(aligned_cluster)

# 3. Create intervention functions with next()
def int_store(x): return next(iter_store)
def int_city(x): return next(iter_city)
def int_state(x): return next(iter_state)
def int_type(x): return next(iter_type)
def int_cluster(x): return next(iter_cluster)

interventions_dict = {
    'store_nbr': int_store,
    'city': int_city,
    'store_type': int_type,
    'state':int_state,
    'cluster':int_cluster,
}

print("\nGenerating synthetic data with paired Spatial & Store Interventions...")

clip_cols = ['unit_sales']
if 'transactions' in source_df.columns:
    clip_cols.append('transactions')

synthetic_dataset = generate_synthetic_data(
    scm=scm,
    interventions_dict=interventions_dict,
    num_samples=num_synthetic_samples,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='unit_sales',
    clip_cols=clip_cols
)
print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['unit_sales'])
y_syn = synthetic_dataset['unit_sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)

print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['unit_sales'].mean():.4f}")
print(f"  Std:  {source_df['unit_sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['unit_sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['unit_sales'].std():.4f}")
# %%

print("\nEvaluation on Test Set ...")

# Format out-of-distribution Target set features for unbiased evaluation
X_test = holdout_test_df.drop(columns=['unit_sales'])
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['unit_sales']

# Generate predictions utilizing the SCM-synthesized target model
y_pred = ml_model_syn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"Prediction Results on Test Set:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Store Type': holdout_test_df['store_type'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%

print(f"✓ Synthetic Dataset Shape: {synthetic_dataset.shape}")

# 2. Hybrid Data Construction via Utility
hybrid_dataset = create_hybrid_dataset(
    source_df=source_df,
    synthetic_dataset=synthetic_dataset,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='unit_sales'
)
print(f"✓ Hybrid Dataset Shape: {hybrid_dataset.shape}")
print("\nHybrid Dataset (Source + Synthetic Target)...")

train_aligned = source_df[synthetic_dataset.columns].copy()

hybrid_dataset = pd.concat([train_aligned, synthetic_dataset], ignore_index=True)


print(f"Hybrid Dataset Shape: {hybrid_dataset.shape} (Real: {len(train_aligned)}, Synthetic: {len(synthetic_dataset)})")

X_hybrid = hybrid_dataset.drop(columns=['unit_sales'])
y_hybrid = hybrid_dataset['unit_sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

ml_model_hbr.fit(X_hybrid, y_hybrid)
print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['unit_sales'].mean():.4f}")
print(f"  Std:  {source_df['unit_sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['unit_sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['unit_sales'].std():.4f}")
print("\nEvaluation on Test Set ...")

X_test = holdout_test_df.drop(columns=['unit_sales'])
X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['unit_sales']

y_pred = ml_model_hbr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"\nPrediction Results:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Store Type': holdout_test_df['store_type'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)


print("\nFirst 10 predictions:")
print(results_df)


# %%
print("PART 10: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")

target_col = 'unit_sales'

# 1. Target vs Synthetic
run_whyshift_region_analysis(
    df_source=synthetic_dataset, df_target=holdout_test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Target", plot_filename="FAV_TARGET_SYNTHETIC_TYPE_REGION.png"
)

# 2. Real vs Synthetic
run_whyshift_region_analysis(
    df_source=source_df, df_target=synthetic_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Source", target_name="Synthetic", plot_filename="FAV_REAL_SYNTHETIC_TYPE_REGION.png"
)

# 3. Real vs Hybrid
run_whyshift_region_analysis(
    df_source=source_df, df_target=hybrid_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Source", target_name="Hybrid", plot_filename="FAV_REAL_HYBRID_TYPE_REGION.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=holdout_test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="FAV_HYBRID_TARGET_TYPE_REGION.png"
)