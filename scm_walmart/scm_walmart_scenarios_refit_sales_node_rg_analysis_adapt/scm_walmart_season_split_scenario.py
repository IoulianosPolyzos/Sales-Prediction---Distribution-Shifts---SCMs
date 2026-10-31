#!/usr/bin/env python3

import os
import warnings
import sys
import re
from pathlib import Path
import numpy as np
import itertools
import pandas as pd
import yaml
import networkx as nx
from lightgbm import LGBMRegressor, LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

# --- PATH RESOLUTION ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- ENVIRONMENT SETTINGS ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# --- CUSTOM UTILITIES ---
from scm_walmart.custom_mape import custom_retail_mape
from whyshift.region_analysis import shared_reweight

sys.path.append(str(project_root))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis,
    extract_high_shift_rules
)

from scm_walmart.scm_walmart_graph_utils import build_feature_graph

# --- CONFIGURATION ---
config_path = project_root / "scm_walmart" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_seasons': [1, 2, 4],
    'target_seasons': [3],
    'random_seed': 42
}

# %%

# PART 1: DATA INGESTION & INITIAL FILTERING

print("PART 1: LOADING DATA & SETUP")

try:
    # Load observational data and eliminate redundant/collinear temporal variables
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    # Mean/Zero imputation for missing mechanisms
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")

except FileNotFoundError:
    print("ERROR: Data file final_data_walmart.csv not found!")
    raise


# PART 2: DOMAIN STRATIFICATION (TEMPORAL SHIFT FORMULATION)

print("\nPART 2: DOMAIN SHIFT SPLIT")

source_seasons = CONFIG['source_seasons']
target_seasons = CONFIG['target_seasons']

# Split based on Season
source_df = df[df['Season'].isin(source_seasons)].copy().reset_index(drop=True)
target_df = df[df['Season'].isin(target_seasons)].copy().reset_index(drop=True)

# Partition the Target Domain into an Adaptation set (for structural updating)
# and a Holdout test set (for rigorous OOD evaluation)
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,  # 10% adaptation
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain: {CONFIG['source_seasons']}")
print(f"Train shape: {source_df.shape}")
print(f"\nTest domain: {CONFIG['target_seasons']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Invariant assertion: Verify strict temporal isolation between domains
assert (source_df['Season'].isin([1, 2, 4])).all(), "Train set contains Season = 3!"
assert (adaptation_df['Season'] == 3).all(), "Adaptation set contains Season = 1 2 4!"
assert (holdout_test_df['Season'] == 3).all(), "Holdout set contains Season = 1 2 4!"

# %%

# PART 3: FEATURE TRANSFORMATION AND DATA TYPE ENFORCEMENT

print("\nPART 3: PREPROCESSING")

try:
    # Isolate categorical nodes to ensure correct functional mappings in SCM
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    # Enforce continuous spaces for Additive Noise Models (ANM)
    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    # Explicitly string-encode binary covariates to enforce classifier assignment
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start",
                    "Is_Month_End"]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    # Standardize string representations for multi-class categorical features
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        source_df[col] = source_df[col].astype(str)
        adaptation_df[col] = adaptation_df[col].astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(str)

    # Compile all classification targets for graph formulation
    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

# %%

print("\nPART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(source_df.columns)

# SURGICAL EDGE REMOVAL FOR TEMPORAL CAUSAL VALIDATION
# We deliberately remove the direct edges from the root temporal features
# (Month/Week) to the target variable (Weekly_Sales).
# WHY: Since our intervention (domain shift) originates at the Month/Week level,
# leaving direct shortcuts to Sales would allow the SCM to lazily bypass the
# true physical mechanisms. By cutting these direct links but intentionally
# KEEPING the 'Season -> Weekly_Sales' edge intact, we force the causal
# information to flow through proper intermediate mediators (Date -> Season -> Sales).

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

# %%

print("\nPART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print(f"Fitting SCM on training data (Seasons {CONFIG['source_seasons']})...")
    gcm.fit(scm, source_df)
    print("✓ Initial fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%

print("\nPART 7: SUPERVISED DOMAIN ADAPTATION")

ordered_nodes = list(nx.topological_sort(feature_graph))
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes
]

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
X_src_ord = source_df.drop(columns=['Weekly_Sales']).copy()
X_adapt_ord = adaptation_df.drop(columns=['Weekly_Sales']).copy()

cat_cols_present = [c for c in categorical_nodes if c in X_src_ord.columns]

if cat_cols_present:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([X_src_ord[cat_cols_present], X_adapt_ord[cat_cols_present]]))

    X_src_ord[cat_cols_present] = oe.transform(X_src_ord[cat_cols_present])
    X_adapt_ord[cat_cols_present] = oe.transform(X_adapt_ord[cat_cols_present])

y_src = source_df['Weekly_Sales'].values
y_adapt = adaptation_df['Weekly_Sales'].values

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

# 1. Why use Paired/Aligned Arrays instead of independent random sampling?
#    Cyclical features (sin/cos) are geometrically coupled. By bootstrapping authentic,
#    complete rows from the Target Domain, we inject perfectly synchronized time-coordinates.
num_synthetic_samples = len(target_df)

# Sample FROM THE TARGETED REGION instead of the uniform adaptation set
sampled_target = targeted_adaptation_df.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values

aligned_week_sin = sampled_target['WeekOfYear_sin'].values if 'WeekOfYear_sin' in target_df else None
aligned_week_cos = sampled_target['WeekOfYear_cos'].values if 'WeekOfYear_cos' in target_df else None

aligned_season_sin = sampled_target['Season_sin'].values
aligned_season_cos = sampled_target['Season_cos'].values
aligned_year_progress = sampled_target['Year_Progress'].values
aligned_month_progress = sampled_target['Month_Progress'].values
aligned_is_summer = sampled_target['Is_Summer'].values

def create_intervention_func(values):
    """Helper function to properly scope the iterator for DoWhy interventions"""
    iterator = itertools.cycle(values)
    return lambda x: next(iterator)

interventions_dict = {
    'Month_sin': create_intervention_func(aligned_month_sin),
    'Month_cos': create_intervention_func(aligned_month_cos),
    'WeekOfYear_sin': create_intervention_func(aligned_week_sin) if aligned_week_sin is not None else None,
    'WeekOfYear_cos': create_intervention_func(aligned_week_cos) if aligned_week_cos is not None else None,
    'Season_sin': create_intervention_func(aligned_season_sin),
    'Season_cos': create_intervention_func(aligned_season_cos),
    'Is_Summer': create_intervention_func(aligned_is_summer),
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

print("\nGenerating synthetic data with paired Temporal & Season(sin/cos) Interventions...")
clip_cols = ['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

synthetic_dataset = generate_synthetic_data(
    scm=scm,
    interventions_dict=interventions_dict,
    num_samples=num_synthetic_samples,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales',
    clip_cols=clip_cols
)

print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn)

print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {source_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['Weekly_Sales'].std():.4f}")

# %%

print("\nEvaluation on Test Set ...")

# Format out-of-distribution Target set features for unbiased evaluation
X_test = holdout_test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

# Generate predictions utilizing the SCM-synthesized target model
y_pred = ml_model_syn.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)

print(f"Prediction Results on Test Set:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Season': holdout_test_df['Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)

# %%

# 2. Hybrid Data Construction via Utility
hybrid_dataset = create_hybrid_dataset(
    source_df=source_df,
    synthetic_dataset=synthetic_dataset,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales'
)

X_hybrid = hybrid_dataset.drop(columns=['Weekly_Sales'])
y_hybrid = hybrid_dataset['Weekly_Sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

ml_model_hbr.fit(X_hybrid, y_hybrid)
print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {source_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {source_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {holdout_test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {holdout_test_df['Weekly_Sales'].std():.4f}")
print("\nEvaluation on Test Set ...")

X_test = holdout_test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_hybrid[col].cat.categories)
y_test = holdout_test_df['Weekly_Sales']

y_pred = ml_model_hbr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = custom_retail_mape(y_test, y_pred)
print(f"\nPrediction Results:")
print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
print(f"  - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.2f}")
print(f"  - R²: {r2:.2f}")

results_df = pd.DataFrame({
    'Season': holdout_test_df['Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)

# %%
print("\nPART 10: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")

target_col = 'Weekly_Sales'

# 1. Target vs Synthetic
run_whyshift_region_analysis(
    df_source=synthetic_dataset, df_target=holdout_test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Target", plot_filename="WALMART_TARGET_SYNTHETIC_SEASON_REGION.png"
)

# 2. Real vs Synthetic
run_whyshift_region_analysis(
    df_source=source_df, df_target=synthetic_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Source", target_name="Synthetic", plot_filename="WALMART_REAL_SYNTHETIC_SEASON_REGION.png"
)

# 3. Real vs Hybrid
run_whyshift_region_analysis(
    df_source=source_df, df_target=hybrid_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Real Source", target_name="Hybrid", plot_filename="WALMART_REAL_HYBRID_SEASON_REGION.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=holdout_test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="WALMART_HYBRID_TARGET_SEASON_REGION.png"
)