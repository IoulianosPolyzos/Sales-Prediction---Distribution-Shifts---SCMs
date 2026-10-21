import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, plot_tree
import lightgbm as lgbm

import sys
import os
from whyshift_reg.disde_reg import degradation_decomp_regression

PARQUET_PATH = r"../../corporacion_favorita_dataset/final_data_favorita.parquet"

print(">>> Loading Favorita Dataset from Parquet...")
df_raw = pd.read_parquet(PARQUET_PATH)

# Define domains based on weather condition
source_weather = ['Clouds', 'Rain', 'Snow']
target_weather = ['Clear']

print(f">>> Assigning domains (Target: Clear, Source: Clouds/Rain/Snow)...")
df_raw['domain'] = np.where(df_raw['weather_condition'].isin(target_weather), 'Target',
                   np.where(df_raw['weather_condition'].isin(source_weather), 'Source', 'Other'))

# Drop any rows that don't fall into our defined Source or Target
df_raw = df_raw[df_raw['domain'] != 'Other']

print(">>> Performing Stratified Sampling for Weather...")
# STRATIFIED SAMPLING: Take up to 1,500,000 rows from each domain
df_raw = df_raw.groupby('domain', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 1500000), random_state=42)
).reset_index(drop=True)

print(f"Sampled shape: {df_raw.shape}")
print("Rows per domain:\n", df_raw['domain'].value_counts())

weather_source = df_raw[df_raw['domain'] == 'Source']['unit_sales']
weather_target = df_raw[df_raw['domain'] == 'Target']['unit_sales']

print(f"\nSource (Clouds/Rain/Snow) -> mean: {weather_source.mean():.4f}, std: {weather_source.std():.4f}")
print(f"Target (Clear) -> mean: {weather_target.mean():.4f}, std: {weather_target.std():.4f}")

# OHE: We keep store/location metadata because buildings exist in all weather.
# We DO NOT include 'weather_condition' here since it's our splitting variable!
cols_to_encode = ['city', 'state', 'store_type', 'cluster', 'family']
print("\n>>> One-Hot Encoding categorical variables...")
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)

# Split Source and Target DataFrames
source_df = df_encoded[df_encoded['domain'] == 'Source'].copy()
target_df = df_encoded[df_encoded['domain'] == 'Target'].copy()

# DROP: Remove 'weather_condition', 'domain', and temporal features
# to prevent the model from using time of year as a proxy for weather.
drop_cols = [
    'unit_sales', 'date', 'id', 'domain', 'weather_condition',
    'IsHoliday', 'holiday_proximity', 'is_near_holiday', 'holiday_weight', 'Is_Christmas_Season',
    'Month', 'WeekOfYear', 'Quarter', 'Season', 'Year_Progress', 'Month_Progress',
    'WeekOfYear_sin', 'WeekOfYear_cos', 'Month_sin', 'Month_cos', 'Season_sin', 'Season_cos',
    'Is_Month_Start', 'Is_Month_End', 'DayOfWeek', 'DayOfWeek_sin', 'DayOfWeek_cos'
]

feature_names = [c for c in source_df.columns if c not in drop_cols]

X_source_full = source_df[feature_names].values
y_source_full = source_df['unit_sales'].values

X_target = target_df[feature_names].values
y_target = target_df['unit_sales'].values

# Split Source (70% Train, 30% Test)
np.random.seed(42)
perm_source = np.random.permutation(len(X_source_full))
train_idx = perm_source[:int(0.7 * len(X_source_full))]
id_test_idx = perm_source[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]
X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"\nFeatures used: {len(feature_names)}")

# Split Target (70% Train, 30% Test)
np.random.seed(42)
perm_target = np.random.permutation(len(X_target))
train_target = perm_target[:int(0.7 * len(X_target))]
id_test_target = perm_target[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]
X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]

print(f"Source test size: {len(X_source_test)} | Target test size: {len(X_target_test)}")

print("\n>>> Training LGBM Regressor...")
model = lgbm.LGBMRegressor()
model.fit(X_source_train, y_source_train)

print("\n>>> Running Regression DISDE...")
reg_domain_classifier = lgbm.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    class_weight='balanced',
    eval_metric='logloss'
)

p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
    model, data_sum=100000, domain_classifier=reg_domain_classifier, K=8,
    draw_calibration=True, save_calibration_png='favorita_weather_regression_calibration.png'
)

total_degradation = q2q - p2p
yx_shift = s2q - p2s
x_shift_P = p2s - p2p
x_shift_Q = q2q - s2q
total_x_shift = x_shift_P + x_shift_Q


print(f"Source MAE (p2p): {p2p:.4f}")
print(f"Target MAE (q2q): {q2q:.4f}")
print(f"p2s: expected Source Error on Sx (Reweighted Source Error): {p2s:.4f}")
print(f"s2q: expected Target Error on Sx (Reweighted Target Error): {s2q:.4f}")
print(f"Total Degradation: {total_degradation:.4f}")
print(f"Y|X Shift: {yx_shift:.4f}")
print(f"X Shift (P side): {x_shift_P:.4f}")
print(f"X Shift (Q side): {x_shift_Q:.4f}")
print(f"Total X Shift : {total_x_shift:.4f}")

if total_degradation != 0:
    print(f"X Shift P side ratio: {x_shift_P/total_degradation:.4f}")
    print(f"X Shift Q side ratio: {x_shift_Q/total_degradation:.4f}")
    print(f"X Shift Ratio : {total_x_shift/total_degradation:.4f}")
    print(f"Y|X Shift Ratio: {yx_shift/total_degradation:.4f}")


from whyshift.region_analysis import shared_reweight

wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

# Define LGBM parameters
lgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'num_leaves': 62,
    'max_depth': 20,
    'verbose': -1,
    'n_jobs': -1
}

# Train separate regressors for region disagreement analysis
source_model = lgbm.LGBMRegressor(**lgb_params)
target_model = lgbm.LGBMRegressor(**lgb_params)

source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)

# Disagreement in predictions
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)
new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")
print(f"Source model MAE on ID test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Source model MAE on OOD test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")
print(f"Target model MAE on ID test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on OOD test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

region_tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, min_weight_fraction_leaf=0.05)
region_tree.fit(new_X, new_Y, sample_weight=new_weights)

# Plot 1: Overall Degradation Bar
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(["Source MAE", "Target MAE"], [p2p, q2q], alpha=0.6)
ax.set_title("Favorita Sales: MAE Degradation (Weather Split)")
plt.tight_layout()
plt.savefig("favorita_regression_weather_decomposition.png")
plt.close()

# Plot 2: Risk Tree
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10, precision=2)
plt.title("Risk Regions (Prediction Disagreement) - Weather Split")
plt.savefig("favorita_regression_risk_weather_tree.png", bbox_inches='tight')
plt.close()

# Plot 3: Detailed Shift Decomposition
fig, ax = plt.subplots(figsize=(10, 6))

base_val = min(p2p, q2q)
max_val = max(p2p, q2q)
gap = max_val - base_val

shift_y = abs(p2s - s2q)
shift_p = abs(p2p - p2s)
shift_q = abs(s2q - q2q)
total_shift = shift_y + shift_p + shift_q + 1e-9

norm_y = (shift_y / total_shift) * gap
norm_p = (shift_p / total_shift) * gap
norm_q = (shift_q / total_shift) * gap

ax.bar(["Source", "Target"], [p2p, q2q], color='blue', alpha=0.5)

ax.bar(["Decomposition"], [norm_y], bottom=base_val, color='orange', label="Concept Shift (Y|X)")
ax.bar(["Decomposition"], [norm_p], bottom=base_val + norm_y, color='#AA0000', label="Covariate Shift (X) (P)")
ax.bar(["Decomposition"], [norm_q], bottom=base_val + norm_y + norm_p, color='red', label="Covariate Shift (X) (Q)")

ax.set_title("Favorita Sales: Weather Decomposition (Regression DISDE)")
ax.set_ylabel("MAE")
ax.legend()
plt.tight_layout()
plt.savefig('favorita_regression_weather_decomposition_bar.png')
plt.close()