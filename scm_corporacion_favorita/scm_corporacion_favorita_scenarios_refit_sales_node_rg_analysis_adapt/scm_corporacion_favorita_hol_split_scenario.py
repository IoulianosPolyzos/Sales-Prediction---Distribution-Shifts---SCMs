# %%
# !/usr/bin/env python3

import os
import warnings
import sys
import re
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

sys.path.append(str(project_root))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis,
    extract_high_shift_rules
)
from whyshift.region_analysis import shared_reweight

from scm_corporacion_favorita.scm_favorita_graph_utils import build_feature_graph

config_path = project_root / "scm_corporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_domain': 'Normal Days (IsHoliday=0)',
    'target_domain': 'Holiday Days (IsHoliday=1)',
    'random_seed': 42
}

# %%
print("PART 1: LOADING DATA & SETUP")

try:
    PARQUET_PATH = project_root / "corporacion_favorita_dataset" / "final_data_favorita.parquet"
    print(f">>> Loading Favorita Dataset from Parquet...")
    df = pd.read_parquet(PARQUET_PATH)

    # Drop redundant columns
    df = df.drop(
        columns=["date", "id", "item_nbr", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"IsHoliday distribution:")
    print(f"  Normal Days (0): {(df['IsHoliday'] == 0).sum()} records")
    print(f"  Holiday Days (1): {(df['IsHoliday'] == 1).sum()} records")

except Exception as e:
    print(f"ERROR: Failed to load dataset! {e}")
    raise

# %%
print("\nPART 2: DOMAIN SHIFT SPLIT (NORMAL vs HOLIDAY)")

# Ensure IsHoliday is numeric (0 or 1)
df['IsHoliday'] = df['IsHoliday'].astype(int)

# Extract Source & Target Domains
source_df = df[df['IsHoliday'] == 0].copy().reset_index(drop=True)
target_df = df[df['IsHoliday'] == 1].copy().reset_index(drop=True)

# Subsample source data to ensure SCM computational feasibility
source_df = source_df.sample(n=min(500000, len(source_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
# Split target domain into Adaptation and Holdout sets (10% / 90%)
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)

adaptation_df = adaptation_df.sample(n=min(100000, len(adaptation_df)), random_state=CONFIG['random_seed']).reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source shape: {source_df.shape}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify exact domain separation
assert (source_df['IsHoliday'] == 0).all(), "Train set contains holiday days!"
assert (adaptation_df['IsHoliday'] == 1).all(), "Adaptation set contains normal days!"
assert (holdout_test_df['IsHoliday'] == 1).all(), "Holdout set contains normal days!"
print("✓ Domain separation verified")

# %%
print("\nPART 3: PREPROCESSING")

try:
    categorical_cols_to_exclude = [
        "city", "state", "store_type", "family", "weather_condition",
        "store_nbr", "cluster", "class"
    ]

    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError:
                pass

    binary_nodes = [
        "IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer",
        "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact"
    ]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
    for col in categorical_nodes:
        source_df[col] = source_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%
print("\nPART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(source_df.columns)

# Surgical Removal for Holiday Shift
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

# Identify causal descendants impacted by the intervention
affected_groups = get_all_descendants(macro_graph, "Holidays")
affected_feature_nodes = []
for group in affected_groups:
    affected_feature_nodes.extend(causal_groups[group])

print(f"Direct children of Holidays: {set(macro_graph.successors('Holidays'))}")
print(f"Feature-level graph: {feature_graph.number_of_nodes()} nodes, {feature_graph.number_of_edges()} edges")

# %%
print("\nPART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on training data...")
    gcm.fit(scm, source_df)
    print("✓ Initial fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("\nPART 7: SUPERVISED DOMAIN ADAPTATION")

ordered_nodes = list(nx.topological_sort(feature_graph))
nodes_to_refit = [n for n in ordered_nodes if n in affected_feature_nodes]

print(f"Adapting mechanisms to adaptation set ({len(nodes_to_refit)} nodes)...")
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
print("\nPART 7.5: TARGETED ADAPTATION REGION ANALYSIS & RULE EXTRACTION")

# 1. Ordinal Encoding for Tree (Crucial so rule names match SCM nodes exactly)
X_src_ord = source_df.drop(columns=['unit_sales']).copy()
X_adapt_ord = adaptation_df.drop(columns=['unit_sales']).copy()

cat_cols_present = [c for c in classifier_nodes if c in X_src_ord.columns]

if cat_cols_present:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([X_src_ord[cat_cols_present], X_adapt_ord[cat_cols_present]]))

    X_src_ord[cat_cols_present] = oe.transform(X_src_ord[cat_cols_present])
    X_adapt_ord[cat_cols_present] = oe.transform(X_adapt_ord[cat_cols_present])

y_src = source_df['unit_sales'].values
y_adapt = adaptation_df['unit_sales'].values

# 2. Shared Reweighting & Shift Calculation
wA, wB, _, _ = shared_reweight(X_src_ord.values, X_adapt_ord.values, K=8)

src_model_ra = LGBMRegressor(**lgb_params).fit(X_src_ord.values, y_src, sample_weight=wA)
adapt_model_ra = LGBMRegressor(**lgb_params).fit(X_adapt_ord.values, y_adapt, sample_weight=wB)

pred_src_adapt = src_model_ra.predict(X_adapt_ord.values)
pred_tgt_adapt = adapt_model_ra.predict(X_adapt_ord.values)
shift_y_x = np.abs(pred_src_adapt - pred_tgt_adapt)

# 3. Fit Risk Region Tree on Ordinal Data
feature_names_ord = X_adapt_ord.columns.tolist()
region_tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=CONFIG['random_seed'])
region_tree.fit(X_adapt_ord.values, shift_y_x, sample_weight=wB)

# 4. Extract Rules and Filter Adaptation Data
high_shift_regions = extract_high_shift_rules(region_tree, feature_names_ord, threshold_percentile=80)
rule_features = set()

if high_shift_regions and len(high_shift_regions[0][0]) > 0:
    top_rule_conditions = high_shift_regions[0][0]
    query_str = " & ".join(top_rule_conditions)
    print(f"\nTop High-Shift Rule identified: {query_str}")

    # Extract the exact feature names used in the rule
    rule_features = set(re.findall(r"\`([^\`]+)\`", query_str))

    # Use the ordinal dataframe to evaluate the query and get the indices
    targeted_indices = X_adapt_ord.query(query_str).index
    targeted_adaptation_df = adaptation_df.iloc[targeted_indices].copy().reset_index(drop=True)
else:
    print("\nNo specific high-shift rule found. Using full adaptation_df.")
    targeted_adaptation_df = adaptation_df.copy()

if len(targeted_adaptation_df) == 0:
    print("Warning: Rule returned empty subset. Falling back to full adaptation_df.")
    targeted_adaptation_df = adaptation_df.copy()

# %%
print("\nPART 8 & 9: SYNTHETIC AND HYBRID DATA GENERATION")

num_synthetic_samples = min(500000, len(target_df))

# Sample FROM THE TARGETED REGION instead of the uniform adaptation set
sampled_target = targeted_adaptation_df.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values
aligned_week_sin = sampled_target['WeekOfYear_sin'].values if 'WeekOfYear_sin' in target_df.columns else None
aligned_week_cos = sampled_target['WeekOfYear_cos'].values if 'WeekOfYear_cos' in target_df.columns else None
aligned_holidays = sampled_target['IsHoliday'].values
aligned_year_progress = sampled_target['Year_Progress'].values
aligned_month_progress = sampled_target['Month_Progress'].values


def create_intervention_func(values):
    """Helper function to properly scope the iterator for DoWhy interventions"""
    iterator = itertools.cycle(values)
    return lambda x: next(iterator)

interventions_dict = {
    'Month_sin': create_intervention_func(aligned_month_sin),
    'Month_cos': create_intervention_func(aligned_month_cos),
    'WeekOfYear_sin': create_intervention_func(aligned_week_sin) if aligned_week_sin is not None else None,
    'WeekOfYear_cos': create_intervention_func(aligned_week_cos) if aligned_week_cos is not None else None,
    'IsHoliday': create_intervention_func(aligned_holidays),
    'Year_Progress': create_intervention_func(aligned_year_progress),
    'Month_Progress': create_intervention_func(aligned_month_progress)
}

# Remove None values in case WeekOfYear was not present
interventions_dict = {k: v for k, v in interventions_dict.items() if v is not None}

# --- DYNAMIC RULE-BASED SCM INTERVENTIONS ---
print("\nApplying Direct SCM Interventions from Region Analysis Rules:")
for feature in rule_features:
    if feature in feature_graph.nodes and feature not in interventions_dict:
        feature_vals = sampled_target[feature].values
        interventions_dict[feature] = create_intervention_func(feature_vals)
        print(f"  -> Added intervention node: do({feature} = [Rule-based distribution])")

print("\nGenerating synthetic data with paired Temporal & Holiday Interventions...")

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
ml_model_syn.fit(X_syn, y_syn, categorical_feature=categorical_nodes)

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
    'Holidays': holdout_test_df['IsHoliday'].values,
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
X_hybrid = hybrid_dataset.drop(columns=['unit_sales'])
y_hybrid = hybrid_dataset['unit_sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

ml_model_hbr.fit(X_hybrid, y_hybrid, categorical_feature=categorical_nodes)
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
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_hybrid[col].cat.categories)
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
    'Holidays': holdout_test_df['IsHoliday'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)

# %%
print("\nPART 10: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")


target_col = 'unit_sales'
n_whyshift = 100000
synth_sample = synthetic_dataset.sample(n=min(n_whyshift, len(synthetic_dataset)), random_state=CONFIG['random_seed'])
target_sample = holdout_test_df.sample(n=min(n_whyshift, len(holdout_test_df)), random_state=CONFIG['random_seed'])
source_sample = source_df.sample(n=min(n_whyshift, len(source_df)), random_state=CONFIG['random_seed'])
hybrid_sample = hybrid_dataset.sample(n=min(n_whyshift, len(hybrid_dataset)), random_state=CONFIG['random_seed'])
# 1. Target vs Synthetic
run_whyshift_region_analysis(
    df_source=synth_sample, df_target=target_sample,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Target", plot_filename="FAV_TARGET_SYNTHETIC_HOLIDAY_REGION.png"
)

# 2. Source vs Synthetic
run_whyshift_region_analysis(
    df_source=source_sample, df_target=synth_sample,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Source", target_name="Synthetic", plot_filename="FAV_SOURCE_SYNTHETIC_HOLIDAY_REGION.png"
)

# 3. Source vs Hybrid
run_whyshift_region_analysis(
    df_source=source_sample, df_target=hybrid_sample,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Source", target_name="Hybrid", plot_filename="FAV_SOURCE_HYBRID_HOLIDAY_REGION.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_sample, df_target=target_sample,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="FAV_HYBRID_TARGET_HOLIDAY_REGION.png"
)