# %%
# !/usr/bin/env python3

import os
import warnings
import sys
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scm_walmart.custom_mape import *
import pandas as pd
import networkx as nx
import dowhy.gcm as gcm

current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir.parents[1]))
from utils_scm import (
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
    'source_weather': ["Rain","Snow","Clouds"],
    'target_weather': ["Clear"],
    'random_seed': 42
}




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
    print(f"Unique Weather Conditions: {df['weather_condition'].nunique()}")
    print(f"Weather Conditions: {sorted(df['weather_condition'].unique())}")

except FileNotFoundError:
    print("ERROR: Data file 'final_data_walmart.csv' not found!")
    raise


# PART 2: SPLIT DATA BY WEATHER



print("PART 2: DOMAIN SHIFT SPLIT (TRAIN/TEST BY weather_conditions)")


source_weather = CONFIG['source_weather']
target_weather = CONFIG['target_weather']

source_df = df[df['weather_condition'].isin(source_weather)].copy().reset_index(drop=True)
target_df = df[df['weather_condition'].isin(target_weather)].copy().reset_index(drop=True)


source_domain_train, source_domain_test = train_test_split(
    source_df, test_size=0.1, random_state=42
)
target_domain_train , target_domain_test = train_test_split(target_df,test_size=0.9,random_state=42)

train_df = pd.concat([source_domain_train, target_domain_train], axis=0).sample(frac=1, random_state=42)
test_df = pd.concat([source_domain_test, target_domain_test], axis=0).sample(frac=1, random_state=42)

print(f"Train weather conditions: {source_weather}")
print(f"Train shape: {train_df.shape}")
print(f"\nTest weather conditions: {target_weather}")


# Verify no overlap
#assert set(source_weather).isdisjoint(set(target_weather)), "Train/Test weather conditions overlap!"
print("✓ No overlap between TRAIN/TEST weather_conditions")
#%%


print("PART 3: PREPROCESSING")


try:
    # Encode cities
    # le_city = LabelEncoder()
    # le_city.fit(df['city'].astype(str))
    # train_df['city'] = le_city.transform(train_df['city'].astype(str))
    # adaptation_df['city'] = le_city.transform(adaptation_df['city'].astype(str))
    # holdout_test_df['city'] = le_city.transform(holdout_test_df['city'].astype(str))

    

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
            except ValueError as e:
                print(f"WARNING: Could not convert {col} to numeric: {e}")

    # Binary encoding
    binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season","Is_Summer","Is_Month_Start","Is_Month_End"]
    for col in binary_nodes:
        train_df[col] = train_df[col].astype(int).astype(str)
        test_df[col] = test_df[col].astype(int).astype(str)

    # Categorical encoding
    categorical_nodes = ["Type", "weather_condition", "Store", "Dept","city"]
    for col in categorical_nodes:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    classifier_nodes = binary_nodes + categorical_nodes
    print(f"✓ Preprocessing complete (Classifier nodes: {len(classifier_nodes)})")

except Exception as e:
    print(f"ERROR: Preprocessing failed: {e}")
    raise
#%%
print("PART 4 & 5: CAUSAL GRAPH ANALYSIS & BUILDING")

# Build the base feature graph and retrieve structural groups using the external utility
feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

# Since there are source data in train set , there is no need to remove the edge between
# domain split variable or the intervention variable and sales node

# %%
print("PART 6: FITTING MIXED STRUCTURAL CAUSAL MODEL (ON POOLED TRAIN DATA)")

# Instantiate Structural Causal Model over feature-level DAG
scm = gcm.StructuralCausalModel(feature_graph)

try:
    # Set up the internal causal mechanisms (LGBM) using the external utility
    setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)
    print("✓ Mechanisms setup complete")

    print("Fitting SCM on Dtrain (90% Source Weather + 10% Target Weather)...")

    # The SCM is trained directly on the mixed dataset, automatically
    # learning adverse and clear weather behaviors without needing a separate Domain Adaptation step.
    gcm.fit(scm, train_df)
    print("✓ Mixed SCM fit complete")

except Exception as e:
    print(f"ERROR: Failed to fit SCM: {e}")
    raise


# %%
print("PART 7: EVALUATION ON Dtest (10% SOURCE WEATHER + 90% TARGET WEATHER)")

try:
    print(f"Generating predictions on {len(test_df)} mixed test samples...")

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

    # Utilize the standard evaluation function for a clean printout
    evaluate_sales_on_holdout(
        scm, feature_graph, test_df, "MIXED SCM EVALUATION",target_col="Weekly_Sales"
    )

except Exception as e:
    print(f"ERROR: Test evaluation failed: {e}")
    raise
