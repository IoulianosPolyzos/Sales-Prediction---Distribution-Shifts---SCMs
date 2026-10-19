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
    'checkpoint_dir': './checkpoints_season',
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
    print(
        "ERROR: Data file final_data_walmart.csv'not found!")
    raise


# PART 2: DOMAIN STRATIFICATION (TEMPORAL SHIFT FORMULATION)


print("PART 2: DOMAIN SHIFT SPLIT")

# Ensure intervention target (Season) is explicitly discrete
df['Season'] = df['Season'].astype(int)


source_df = df[df['Season'].isin([1, 2, 4])].copy()
target_df = df[df['Season'].isin([3])].copy()

# Partition the Target Domain into an Adaptation set (for structural updating)
# and a Holdout test set (for rigorous OOD evaluation)
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.20,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Train domain: {CONFIG['source_domain']}")
print(f"Train shape: {source_df.shape}")
print(f"\nTest domain: {CONFIG['target_domain']}")
print(f"Adaptation set shape (for refit): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Invariant assertion: Verify strict temporal isolation between domains
assert (source_df['Season'].isin([1, 2, 4])).all(), "Train set contains Season = 3!"
assert (adaptation_df['Season'] == 3).all(), "Adaptation set contains Season = 1 2 4!"
assert (holdout_test_df['Season'] == 3).all(), "Holdout set contains Season = 1 2 4!"

# %%


# PART 3: FEATURE TRANSFORMATION AND DATA TYPE ENFORCEMENT


print("PART 3: PREPROCESSING")

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

print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(source_df.columns)


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

# %%
print("PART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

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
print("PART 7: SUPERVISED DOMAIN ADAPTATION")

ordered_nodes = list(nx.topological_sort(feature_graph))
nodes_to_refit = [
    n for n in ordered_nodes
    if n in affected_feature_nodes and n != "Weekly_Sales"
]

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
print("PART 7 & 8: SYNTHETIC AND HYBRID DATA GENERATION")


# 1. Why use Paired/Aligned Arrays instead of independent random sampling?
#    Cyclical features (sin/cos) are geometrically coupled. Independent sampling
#    could fatally mix the sine of June with the cosine of December, creating
#    non-existent temporal paradoxes. By bootstrapping authentic, complete rows
#    from the Target Domain, we inject perfectly synchronized time-coordinates.
#    The SCM is thus forced to generate counterfactuals around 100% realistic,
#    chronologically coherent dates.

num_synthetic_samples = len(target_df)


sampled_target = adaptation_df.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])


aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values

aligned_week_sin = sampled_target['WeekOfYear_sin'].values if 'WeekOfYear_sin' in target_df else None
aligned_week_cos = sampled_target['WeekOfYear_cos'].values if 'WeekOfYear_cos' in target_df else None

aligned_season_sin = sampled_target['Season_sin'].values
aligned_season_cos = sampled_target['Season_cos'].values
aligned_year_progress = sampled_target['Year_Progress'].values
aligned_month_progress = sampled_target['Month_Progress'].values
aligned_is_summer = sampled_target['Is_Summer'].values

def intervention_month_sin(*args, **kwargs): return aligned_month_sin
def intervention_month_cos(*args, **kwargs): return aligned_month_cos
def intervention_week_sin(*args, **kwargs):  return aligned_week_sin
def intervention_week_cos(*args, **kwargs):  return aligned_week_cos
def intervention_is_summer(*args, **kwargs):  return aligned_is_summer
def intervention_season_sin(*args, **kwargs): return aligned_season_sin
def intervention_season_cos(*args, **kwargs): return aligned_season_cos
def intervention_year_progress(*args, **kwargs): return aligned_year_progress
def intervention_month_progress(*args, **kwargs): return aligned_month_progress

interventions_dict = {
    'Month_sin': intervention_month_sin,
    'Month_cos': intervention_month_cos,
    'WeekOfYear_sin': intervention_week_sin,
    'WeekOfYear_cos': intervention_week_cos,
    'Season_sin': intervention_season_sin,
    'Season_cos': intervention_season_cos,
    'Is_Summer': intervention_is_summer,
    'Year_Progress': intervention_year_progress,
    'Month_Progress': intervention_month_progress
}
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
print("Synthetic Season distribution:")
print(synthetic_dataset['Season'].value_counts())


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
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
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
print("PART 10: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")

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
