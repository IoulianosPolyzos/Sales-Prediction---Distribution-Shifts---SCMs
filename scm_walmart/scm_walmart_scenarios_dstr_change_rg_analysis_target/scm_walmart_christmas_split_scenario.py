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

import yaml
import numpy as np
import itertools
import pandas as pd
import networkx as nx
from lightgbm import LGBMRegressor
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

from scm_walmart.custom_mape import *
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


import dowhy.gcm as gcm
from dowhy.gcm.fitting_sampling import fit_causal_model_of_target

# --- PATH MANAGEMENT FOR EXTERNAL UTILITIES ---
current_dir = Path(__file__).resolve().parent


sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
    get_all_descendants,
    setup_mechanisms,
    evaluate_sales_on_holdout,
    generate_synthetic_data,
    create_hybrid_dataset,
    run_whyshift_region_analysis,
    extract_high_shift_rules
)
from whyshift.region_analysis import shared_reweight


sys.path.append(str(current_dir.parent))
from scm_walmart.scm_walmart_graph_utils import build_feature_graph

# --- CONFIGURATION ---
try:
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    lgb_params = config["lgb_params"]
except FileNotFoundError:
    print("Config not found, using default LGBM parameters.")
    lgb_params = {'n_estimators': 100, 'random_state': 42}


CONFIG = {
    'source_domain': 'Normal Days (IsChristmasSeasons=0)',
    'target_domain': 'Christmas Days (Is_Christmas_Season=1)',
    'random_seed': 42
}

#%%

print("PART 1: LOADING DATA & SETUP")


try:
    CSV_PATH = project_root / "walmart_dataset" / "final_data_walmart.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.drop(
        columns=["Date","Season","DayOfWeek","Month","WeekOfYear"],
        errors='ignore'
    )
    df = df.fillna(0)

    print(f"Dataset shape: {df.shape}")
    print(f"Is_Christmas_Season distribution:")
    print(f"  Normal Days (0): {(df['Is_Christmas_Season'] == 0).sum()} records")
    print(f"  Christmas Days (1): {(df['Is_Christmas_Season'] == 1).sum()} records")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY Christmas DOMAIN


print("\nPART 2: DOMAIN SHIFT SPLIT (NORMAL vs Christmas)")


# Ensure Is_Christmas_Season is numeric (0 or 1)
df['Is_Christmas_Season'] = df['Is_Christmas_Season'].astype(int)

source_df = df[df['Is_Christmas_Season'] == 0].copy().reset_index(drop=True)
target_df = df[df['Is_Christmas_Season'] == 1].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=CONFIG['random_seed']
)

source_domain_train = source_domain_train.reset_index(drop=True)
source_domain_test = source_domain_test.reset_index(drop=True)

# Target: 10% goes to train, 90% goes to test
target_domain_train, target_domain_test = train_test_split(
    target_df, test_size=0.9, random_state=CONFIG['random_seed']
)

target_domain_train = target_domain_train.reset_index(drop=True)
target_domain_test = target_domain_test.reset_index(drop=True)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=CONFIG[
    'random_seed']).reset_index(drop=True)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=CONFIG[
    'random_seed']).reset_index(drop=True)


print(f"Train domain: {CONFIG['source_domain']}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest domain: {CONFIG['target_domain']}")


# Verify domain separation
#assert (train_df['Is_Christmas_Season'] == 0).all(), "Train set contains non-zero Is_Christmas_Season!"

print("✓ Domain separation verified: No overlap between Normal/Christmas days")

#%%

print("\nPART 3: PREPROCESSING")


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
                test_df[col] = pd.to_numeric(test_df[col], errors='raise')
                source_domain_train[col] = pd.to_numeric(source_domain_train[col], errors='raise')
                target_domain_train[col] = pd.to_numeric(target_domain_train[col], errors='raise')
                source_domain_test[col] = pd.to_numeric(source_domain_test[col], errors='raise')
                target_domain_test[col] = pd.to_numeric(target_domain_test[col], errors='raise')
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)
        source_domain_train[col] = source_domain_train[col].astype(int).astype(str)
        target_domain_train[col] = target_domain_train[col].astype(int).astype(str)
        source_domain_test[col] = source_domain_test[col].astype(int).astype(str)
        target_domain_test[col] = target_domain_test[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept", "city"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        source_domain_train[col] = source_domain_train[col].astype(str)
        target_domain_train[col] = target_domain_train[col].astype(str)
        source_domain_test[col] = source_domain_test[col].astype(str)
        target_domain_test[col] = target_domain_test[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise

#%%
print("\nPART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# Since there are source data in train set , there is no need to remove the edge between
# domain split variable or the intervention variable and sales node

# %%
print("\nPART 6: FITTING MIXED STRUCTURAL CAUSAL MODEL (ON POOLED TRAIN DATA)")

scm = gcm.StructuralCausalModel(feature_graph)

try:
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("Fitting SCM on Dtrain (90% Normal + 10% Christmas)...")

    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")
except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise

# %%
print("\nPART 7: EVALUATION ON Dtest (10% NORMAL + 90% CHRISTMAS)")

try:
    sales_mechanism = scm.causal_mechanism("Weekly_Sales")
    parents = sorted(list(feature_graph.predecessors("Weekly_Sales")))

    X_test = test_df[parents].to_numpy()
    test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

    mae = mean_absolute_error(test_df['Weekly_Sales'].values, test_predictions)
    rmse = np.sqrt(mean_squared_error(test_df['Weekly_Sales'].values, test_predictions))
    mape = custom_retail_mape(test_df['Weekly_Sales'].values, test_predictions)
    r2 = r2_score(test_df['Weekly_Sales'].values, test_predictions)

    print(f"\n PREDICTION METRICS (Mixed Test Set) ")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE (Sklearn): {mape:.2f}")
    print(f"R²:   {r2:.4f}")

    evaluate_sales_on_holdout(
        scm, feature_graph, test_df, "MIXED SCM EVALUATION",target_col="Weekly_Sales"
    )

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise

# %%
print("\nPART 7.5: TARGETED REGION ANALYSIS & RULE EXTRACTION")

# 1. Ordinal Encoding for Tree (Crucial so rule names match SCM nodes exactly)
# Using source_domain_train as the source and target_domain_train as the adaptation/target set
X_src_ord = source_domain_train.drop(columns=['Weekly_Sales']).copy()
X_adapt_ord = target_domain_train.drop(columns=['Weekly_Sales']).copy()

cat_cols_present = [c for c in classifier_nodes if c in X_src_ord.columns]

if cat_cols_present:
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    oe.fit(pd.concat([X_src_ord[cat_cols_present], X_adapt_ord[cat_cols_present]]))

    X_src_ord[cat_cols_present] = oe.transform(X_src_ord[cat_cols_present])
    X_adapt_ord[cat_cols_present] = oe.transform(X_adapt_ord[cat_cols_present])

y_src = source_domain_train['Weekly_Sales'].values
y_adapt = target_domain_train['Weekly_Sales'].values

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

# 4. Extract Rules and Filter Adaptation Data (Target Domain Train)
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
    targeted_adaptation_df = target_domain_train.iloc[targeted_indices].copy().reset_index(drop=True)
else:
    print("\nNo specific high-shift rule found. Using full target_domain_train.")
    targeted_adaptation_df = target_domain_train.copy()

if len(targeted_adaptation_df) == 0:
    print("Warning: Rule returned empty subset. Falling back to full target_domain_train.")
    targeted_adaptation_df = target_domain_train.copy()

# %%
print("\nPART 8 & 9: SYNTHETIC AND HYBRID DATA GENERATION")

num_synthetic_samples = len(target_df)

# Sample FROM THE TARGETED REGION instead of the uniform target domain train set
sampled_target = targeted_adaptation_df.sample(n=num_synthetic_samples, replace=True, random_state=CONFIG['random_seed'])

aligned_month_sin = sampled_target['Month_sin'].values
aligned_month_cos = sampled_target['Month_cos'].values
aligned_week_sin = sampled_target['WeekOfYear_sin'].values if 'WeekOfYear_sin' in target_df.columns else None
aligned_week_cos = sampled_target['WeekOfYear_cos'].values if 'WeekOfYear_cos' in target_df.columns else None
aligned_christmas = sampled_target['Is_Christmas_Season'].values
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
    'Is_Christmas_Season': create_intervention_func(aligned_christmas),
    'Year_Progress': create_intervention_func(aligned_year_progress),
    'Month_Progress': create_intervention_func(aligned_month_progress)
}

# Remove None values in case WeekOfYear was not present
interventions_dict = {k: v for k, v in interventions_dict.items() if v is not None}

# --- DYNAMIC RULE-BASED SCM INTERVENTIONS ---
print("\nApplying Direct SCM Interventions from Region Analysis Rules:")
for feature in rule_features:
    if feature in feature_graph.nodes and feature not in interventions_dict:
        # Intervene directly on the SCM node by forcing it to sample from the rule's distribution
        feature_vals = sampled_target[feature].values
        interventions_dict[feature] = create_intervention_func(feature_vals)
        print(f"  -> Added intervention node: do({feature} = [Rule-based distribution])")


clip_cols = [
    'Weekly_Sales',
    'MarkDown1',
    'MarkDown2',
    'MarkDown3',
    'MarkDown4',
    'MarkDown5'
]

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

print("\nTrain ML model on Synthetic Dataset...")

# Extrapolate functional mapping strictly via the synthetically generated Interventional distribution
X_syn = synthetic_dataset.drop(columns=['Weekly_Sales'])
y_syn = synthetic_dataset['Weekly_Sales']

ml_model_syn = LGBMRegressor(**lgb_params)
ml_model_syn.fit(X_syn, y_syn, categorical_feature=categorical_nodes)

print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {train_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {train_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {test_df['Weekly_Sales'].std():.4f}")
# %%

print("\nEvaluation on Test Set ...")

# Format out-of-distribution Target set features for unbiased evaluation
X_test = test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_syn.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_syn[col].cat.categories)
y_test = test_df['Weekly_Sales']

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
    'Christmas Season': test_df['Is_Christmas_Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)
# %%


# 2. Hybrid Data Construction via Utility
hybrid_dataset = create_hybrid_dataset(
    source_df=train_df,
    synthetic_dataset=synthetic_dataset,
    original_df=df,
    classifier_nodes=classifier_nodes,
    target_col='Weekly_Sales'
)

X_hybrid = hybrid_dataset.drop(columns=['Weekly_Sales'])
y_hybrid = hybrid_dataset['Weekly_Sales']

print("\nTraining ML Model on Hybrid Dataset...")
ml_model_hbr = LGBMRegressor(**lgb_params)

ml_model_hbr.fit(X_hybrid, y_hybrid, categorical_feature=categorical_nodes)
print(f"\nTRAIN DATA STATS :")
print(f"  Mean: {train_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {train_df['Weekly_Sales'].std():.4f}")

print(f"\nTESTDATA STATS :")
print(f"  Mean: {test_df['Weekly_Sales'].mean():.4f}")
print(f"  Std:  {test_df['Weekly_Sales'].std():.4f}")
print("\nEvaluation on Test Set ...")

X_test = test_df.drop(columns=['Weekly_Sales'])
X_test = X_test[X_hybrid.columns]
for col in classifier_nodes:
    if col in X_test.columns:
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=X_hybrid[col].cat.categories)
y_test = test_df['Weekly_Sales']

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
    'Christmas Season': test_df['Is_Christmas_Season'].values,
    'Actual_Sales': y_test.values,
    'Predicted_Sales': y_pred
}).head(10)

print("\nFirst 10 predictions:")
print(results_df)

# %%
print("\nPART 10: WHYSHIFT REGION ANALYSIS (DEGRADATION & RISK REGIONS)")

target_col = 'Weekly_Sales'

# 1. Test vs Synthetic
run_whyshift_region_analysis(
    df_source=synthetic_dataset, df_target=test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Synthetic", target_name="Test", plot_filename="WALMART_TEST_SYNTHETIC_CHRISTMAS_MIXED_REGION.png"
)

# 2. Dtrain vs Synthetic
run_whyshift_region_analysis(
    df_source=train_df, df_target=synthetic_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Dtrain", target_name="Synthetic", plot_filename="WALMART_TRAIN_SYNTHETIC_CHRISTMAS_MIXED_REGION.png"
)

# 3. Dtrain vs Hybrid
run_whyshift_region_analysis(
    df_source=train_df, df_target=hybrid_dataset,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Dtrain", target_name="Hybrid", plot_filename="WALMART_TRAIN_HYBRID_CHRISTMAS_MIXED_REGION.png"
)

# 4. Hybrid vs Test
run_whyshift_region_analysis(
    df_source=hybrid_dataset, df_target=test_df,
    target_col=target_col, categorical_nodes=categorical_nodes, lgb_params=lgb_params,
    source_name="Hybrid", target_name="Test", plot_filename="WALMART_HYBRID_TEST_CHRISTMAS_MIXED_REGION.png"
)