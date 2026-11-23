# %%
# !/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path

# --- PATH RESOLUTION ---
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- STANDARD IMPORTS ---
import numpy as np
import yaml
import itertools
import pandas as pd
import networkx as nx
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree, DecisionTreeRegressor

# --- SCM & CAUSAL IMPORTS ---
import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

# --- CUSTOM UTILITIES ---
from scm_walmart.custom_mape import custom_retail_mape
from whyshift.region_analysis import shared_reweight

# --- PATH MANAGEMENT FOR EXTERNAL UTILITIES ---
current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis,
    extract_high_shift_rules
)

sys.path.append(str(current_dir.parent))
from scm_walmart.scm_walmart_graph_utils import build_feature_graph

# --- ENVIRONMENT SETTINGS ---
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION ---
config_path = project_root / "scm_walmart" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

CONFIG = {
    'source_domain': 'Normal Days (IsChristmas_Season=0)',
    'target_domain': 'Holiday Days (Is_Christmas_Season=1)',
    'random_seed': 42,
}
# %%

print("PART 1: LOADING DATA & SETUP")

try:
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date", "Season", "DayOfWeek", "Month", "WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Is_Christmas_Season distribution:")
    print(f"  Normal Days (0): {(df['Is_Christmas_Season'] == 0).sum()} records")
    print(f"  Christmas Season Days (1): {(df['Is_Christmas_Season'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise

print("\nPART 2: DOMAIN SHIFT SPLIT (NORMAL vs CHRISTMAS SEASON)")

# Ensure Is_Christmas_Season is numeric (0 or 1)
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

source_df = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)
target_df = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)

# 10% for Adaptation, 90% for Holdout Testing
adaptation_df, holdout_test_df = train_test_split(
    target_df,
    test_size=0.90,
    random_state=CONFIG['random_seed']
)
adaptation_df = adaptation_df.reset_index(drop=True)
holdout_test_df = holdout_test_df.reset_index(drop=True)

print(f"Source domain: {CONFIG['source_domain']}")
print(f"Source shape: {source_df.shape}")
print(f"\nTarget domain: {CONFIG['target_domain']}")
print(f"Adaptation set shape (for refit/analysis): {adaptation_df.shape}")
print(f"Holdout Test shape (for evaluation): {holdout_test_df.shape}")

# Verify domain separation
assert (source_df['Is_Christmas_Season'] == 0).all(), "Train set contains non-zero Is_Christmas_Season!"
assert (adaptation_df['Is_Christmas_Season'] == 1).all(), "Adaptation set contains non-CHRISTMAS days!"
assert (holdout_test_df['Is_Christmas_Season'] == 1).all(), "Holdout set contains non-CHRISTMAS days!"
print("✓ Domain separation verified: No overlap between Normal/CHRISTMAS days")

# %%

print("\nPART 3: PREPROCESSING")

try:
    categorical_cols_to_exclude = [
        "city", "Type", "weather_condition",
        "Store", "Dept"
    ]

    for col in source_df.columns:
        if col not in categorical_cols_to_exclude:
            try:
                source_df[col] = pd.to_numeric(source_df[col], errors='raise')
                adaptation_df[col] = pd.to_numeric(adaptation_df[col], errors='raise')
                holdout_test_df[col] = pd.to_numeric(holdout_test_df[col], errors='raise')
            except ValueError as e:
                pass

    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start",
                    "Is_Month_End"]
    for col in binary_nodes:
        source_df[col] = source_df[col].astype(int).astype(str)
        adaptation_df[col] = adaptation_df[col].astype(int).astype(str)
        holdout_test_df[col] = holdout_test_df[col].astype(int).astype(str)

    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
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

# Surgical Removal for Christmas Shift
edges_to_remove = [
    ('Month_sin', 'Weekly_Sales'),
    ('Month_cos', 'Weekly_Sales'),
    ('WeekOfYear_sin', 'Weekly_Sales'),
    ('WeekOfYear_cos', 'Weekly_Sales'),
    ('Is_Christmas_Season', 'Weekly_Sales'),
    ('Year_Progress', 'Weekly_Sales'),
    ('Month_Progress', 'Weekly_Sales')
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

print("\nPART 6: FITTING STRUCTURAL CAUSAL MODEL ON TRAINING DATA")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, categorical_nodes, lgb_params)
    print("Fitting SCM on training data (Normal Days)...")
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
        except Exception as e:
            print(f"Failed refit on {node}: {e}")
    print(f"✓ Concept shift adaptation complete ({refit_count}/{len(nodes_to_refit)} nodes)")
except Exception as e:
    print(f"ERROR: Domain adaptation failed: {e}")
    raise

# %%
import re
from sklearn.preprocessing import OrdinalEncoder

print("\nPART 7.5: TARGETED ADAPTATION REGION ANALYSIS & RULE EXTRACTION")

# 1. Ordinal Encoding for Tree (Crucial so rule names match SCM nodes exactly, avoiding dummy cols)
X_src_ord = source_df.drop(columns=['Weekly_Sales']).copy()
X_adapt_ord = adaptation_df.drop(columns=['Weekly_Sales']).copy()

cat_cols_present = [c for c in classifier_nodes if c in X_src_ord.columns]

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
high_shift_regions = extract_high_shift_rules(region_tree, feature_names_ord, threshold_percentile=50)
rule_features = set()

valid_rules = ["(" + " & ".join(r[0]) + ")" for r in high_shift_regions if len(r[0]) > 0]
if valid_rules:
    query_str = " | ".join(valid_rules)
    print(f"\nCombined High-Shift Rules ({len(valid_rules)} regions): {query_str}")
    rule_features = set(re.findall(r"\`([^\`]+)\`", query_str))
    targeted_indices = X_adapt_ord.query(query_str).index
    targeted_adaptation_df = adaptation_df.iloc[targeted_indices].copy().reset_index(drop=True)
else:
    targeted_adaptation_df = adaptation_df.copy()

if len(targeted_adaptation_df) == 0:
    print("Warning: Rule returned empty subset. Falling back to full adaptation_df.")
    targeted_adaptation_df = adaptation_df.copy()

# %%

print("\nPART 8 & 9: SYNTHETIC DATA GENERATION (SCM INTERVENTIONS)")

num_synthetic_samples = len(target_df)

# Sample base temporal data from the targeted subset
sampled_target = targeted_adaptation_df.sample(n=num_synthetic_samples, replace=True,
                                               random_state=CONFIG['random_seed'])

aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values
aligned_week_sin = sampled_target['WeekOfYear_sin'].values
aligned_week_cos = sampled_target['WeekOfYear_cos'].values
aligned_christmas = sampled_target['Is_Christmas_Season'].values
aligned_year_progress = sampled_target['Year_Progress'].values
aligned_month_progress = sampled_target['Month_Progress'].values
aligned_holiday = sampled_target['IsHoliday'].values
aligned_near_holiday = sampled_target['is_near_holiday'].values
aligned_holiday_prox = sampled_target['holiday_proximity'].values
aligned_hol_weight = sampled_target['holiday_weight'].values

iter_month_sin = itertools.cycle(aligned_month_sin)
iter_month_cos = itertools.cycle(aligned_month_cos)
iter_week_sin = itertools.cycle(aligned_week_sin)
iter_week_cos = itertools.cycle(aligned_week_cos)
iter_christmas = itertools.cycle(aligned_christmas)
iter_year_progress = itertools.cycle(aligned_year_progress)
iter_month_progress = itertools.cycle(aligned_month_progress)
iter_holiday = itertools.cycle(aligned_holiday)
iter_near_holiday = itertools.cycle(aligned_near_holiday)
iter_hol_prox = itertools.cycle(aligned_holiday_prox)
iter_hol_weight = itertools.cycle(aligned_hol_weight)

def int_month_sin(x): return next(iter_month_sin)
def int_month_cos(x): return next(iter_month_cos)
def int_week_sin(x): return next(iter_week_sin)
def int_week_cos(x): return next(iter_week_cos)
def int_christmas(x): return next(iter_christmas)
def int_year_progress(x): return next(iter_year_progress)
def int_month_progress(x): return next(iter_month_progress)
def int_holiday(x): return next(iter_holiday)
def int_near_holiday(x): return next(iter_near_holiday)
def int_hol_prox(x): return next(iter_hol_prox)
def int_hol_weight(x): return next(iter_hol_weight)


interventions_dict = {
    'Month_sin': int_month_sin,
    'Month_cos': int_month_cos,
    'WeekOfYear_sin': int_week_sin,
    'WeekOfYear_cos': int_week_cos,
    'Is_Christmas_Season': int_christmas,
    'Year_Progress': int_year_progress,
    'Month_Progress': int_month_progress,
    'IsHoliday': int_holiday,
    'is_near_holiday': int_near_holiday,
    'holiday_proximity': int_hol_prox,
    'holiday_weight': int_hol_weight,
}

def create_intervention_func(values):
    iterator = itertools.cycle(values)
    return lambda x: next(iterator)


# --- DYNAMIC RULE-BASED SCM INTERVENTIONS ---
print("\nApplying Direct SCM Interventions from Region Analysis Rules:")
for feature in rule_features:
    if feature in feature_graph.nodes and feature not in interventions_dict:
        # Intervene directly on the SCM node by forcing it to sample from the rule's distribution
        feature_vals = sampled_target[feature].values
        interventions_dict[feature] = create_intervention_func(feature_vals)
        print(f"  -> Added intervention node: do({feature} = [Rule-based distribution])")



clip_cols = ['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']

print("\nGenerating synthetic data...")
synthetic_dataset = generate_synthetic_data(
    scm=scm,
    interventions_dict=interventions_dict,
    num_samples=num_synthetic_samples,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales',
    clip_cols=clip_cols
)
sales_parents = sorted(list(feature_graph.predecessors('Weekly_Sales')))
sales_mech = scm.causal_mechanism('Weekly_Sales')
X_syn_parents = synthetic_dataset[sales_parents].to_numpy()
clean_sales = sales_mech.prediction_model.predict(X_syn_parents).flatten()
synthetic_dataset['Weekly_Sales'] = np.clip(clean_sales, a_min=0, a_max=None)
print(f"Denoised Synthetic Weekly_Sales Mean: {synthetic_dataset['Weekly_Sales'].mean():.2f}")
print("\nTrain ML model on Synthetic Dataset...")
# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn, categorical_feature=categorical_nodes)

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
    'Christmas Season': holdout_test_df['Is_Christmas_Season'].values,
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

weights = np.concatenate([
    np.ones(len(source_df)),
    np.full(len(synthetic_dataset), 0.05)
])

ml_model_hbr.fit(X_hybrid, y_hybrid, sample_weight=weights, categorical_feature=categorical_nodes)
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
    'Christmas Season': holdout_test_df['Is_Christmas_Season'].values,
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
    source_name="Synthetic", target_name="Target", plot_filename="WALMART_TARGET_SYNTHETIC_CHRISTMAS_REGION_test.png"
)

# 2. Source vs Synthetic
run_whyshift_region_analysis(
    df_source=source_df, df_target=synthetic_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Source", target_name="Synthetic", plot_filename="WALMART_SOURCE_SYNTHETIC_CHRISTMAS_REGION_test.png"
)

# 3. Source vs Hybrid
run_whyshift_region_analysis(
    df_source=source_df, df_target=hybrid_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Source", target_name="Hybrid", plot_filename="WALMART_SOURCE_HYBRID_CHRISTMAS_REGION_test.png"
)

# 4. Hybrid vs Target
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=holdout_test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Target", plot_filename="WALMART_HYBRID_TARGET_CHRISTMAS_REGION_test.png"
)