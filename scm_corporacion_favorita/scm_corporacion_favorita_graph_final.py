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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import dowhy.gcm as gcm

current_dir = Path(__file__).resolve().parent

sys.path.append(str(current_dir.parents[1]))
from utils_scm import setup_mechanisms

sys.path.append(str(current_dir.parent))
from scm_favorita_graph_utils import get_favorita_definitions, build_feature_graph


config_path = project_root / "scm_orporacion_favorita" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"
print(f">>> Loading Favorita Dataset from Parquet...")
df = pd.read_parquet(PARQUET_PATH)

df = df.sample(n=200000, random_state=42).reset_index(drop=True)
df = df.drop(columns=["date", "id", "item_nbr", "Season", "DayOfWeek", "Month", "WeekOfYear"], errors='ignore')
df = df.fillna(0)

print(f"Dataset shape after preprocessing: {df.shape}")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

categorical_cols_to_exclude = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]

for col in train_df.columns:
    if col not in categorical_cols_to_exclude:
        try:
            train_df[col] = pd.to_numeric(train_df[col], errors='raise')
            test_df[col] = pd.to_numeric(test_df[col], errors='raise')
        except ValueError as e:
            pass

binary_nodes = ["IsHoliday", "is_near_holiday", "Is_Christmas_Season", "Is_Summer", "Is_Month_Start", "Is_Month_End", "onpromotion", "perishable", "earthquake_impact"]
for col in binary_nodes:
    train_df[col] = train_df[col].astype(int).astype(str)
    test_df[col] = test_df[col].astype(int).astype(str)

categorical_nodes = ["city", "state", "store_type", "family", "weather_condition", "store_nbr", "cluster", "class"]
for col in categorical_nodes:
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)

classifier_nodes = binary_nodes + categorical_nodes

feature_graph, causal_groups, macro_graph = build_feature_graph(train_df.columns)

print(f"\nNodes: {feature_graph.number_of_nodes()}")
print(f"Edges: {feature_graph.number_of_edges()}")

# --- SCM SPECIFICATION & FITTING ---
scm = gcm.StructuralCausalModel(feature_graph)

setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params)

print("\nFitting SCM on training data...")
gcm.fit(scm, train_df)

# --- EVALUATION ---
sales_mechanism = scm.causal_mechanism("unit_sales")
parents = sorted(list(feature_graph.predecessors("unit_sales")))
X_test = test_df[parents].to_numpy()

test_predictions = sales_mechanism.prediction_model.predict(X_test).flatten()

mae = mean_absolute_error(test_df['unit_sales'].values, test_predictions)
rmse = np.sqrt(mean_squared_error(test_df['unit_sales'].values, test_predictions))
mape = mean_absolute_percentage_error(test_df['unit_sales'].values, test_predictions)
r2 = r2_score(test_df['unit_sales'].values, test_predictions)

print("\n--- Downstream Performance of SCM on unit_sales ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"R2:   {r2:.4f}")