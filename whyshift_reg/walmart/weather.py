from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import lightgbm as lgbm
from whyshift_reg.disde_reg import *

import yaml
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

config_path = project_root / "scm_walmart" / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

lgb_params = config["lgb_params"]

# Path to your specific Walmart CSV file
CSV_PATH = r"../../walmart_dataset/final_data_walmart.csv"



print(">>> Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)


df_raw.columns = df_raw.columns.str.strip()



source_weather = ['Clouds','Rain','Snow']
target_weather = ['Clear']


print(">>> Applying Manual One-Hot Encoding...")
cols_to_encode = ['Quarter']
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)
df_encoded = df_encoded.fillna(0)

source_df = df_encoded[df_encoded['weather_condition'].isin(source_weather)].copy()
target_df = df_encoded[df_encoded['weather_condition'].isin(target_weather)].copy()

drop_cols = [
    'Weekly_Sales', 'Date','Type', 'Season','Month','Quarter','city','Month_sin','Month_cos'
    ,'Is_Summer','WeekOfYear','WeekOfYear_sin','WeekOfYear_cos','Temperature','weather_condition'
    ,'Year_Progress','Season_sin','Season_cos', 'Is_Christmas_Season'
    ,'wind_speed','humidity','precipitation','temperature_max','temperature_min'
    ,'Store','MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5','CPI', 'Unemployment', 'Fuel_Price','Dept','Size'
    'latitude', 'longitude',
]

feature_names = [c for c in source_df.columns if c not in drop_cols]

X_source_full = source_df[feature_names].values
y_source_full = source_df['Weekly_Sales'].values

X_target = target_df[feature_names].values
y_target = target_df['Weekly_Sales'].values

np.random.seed(42)
perm = np.random.permutation(len(X_source_full))
train_idx = perm[:int(0.7 * len(X_source_full))]
id_test_idx = perm[int(0.7 * len(X_source_full)):]

X_source_train = X_source_full[train_idx]
y_source_train = y_source_full[train_idx]

X_source_test = X_source_full[id_test_idx]
y_source_test = y_source_full[id_test_idx]

print(f"Features: {len(feature_names)}")
np.random.seed(42)
perm = np.random.permutation(len(X_target))
train_target = perm[:int(0.7 * len(X_target))]
id_test_target = perm[int(0.7 * len(X_target)):]

X_train_target = X_target[train_target]
y_train_target = y_target[train_target]

X_target_test = X_target[id_test_target]
y_target_test = y_target[id_test_target]
print(f"Source test: {len(X_source_test)} | Target: {len(X_target_test)}")

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
    use_label_encoder=False,
    eval_metric='logloss'
)
p2p, q2q, p2s, s2q = degradation_decomp_regression(
    X_source_test, y_source_test,
    X_target_test, y_target_test,
    model, data_sum=20000, domain_classifier = reg_domain_classifier, K=8,draw_calibration=True, save_calibration_png='weather_regression_calibration.png'
)



total_degradation = q2q - p2p
yx_shift = s2q - p2s
x_shift_P = p2s - p2p
x_shift_Q = q2q - s2q
total_x_shift = x_shift_P + x_shift_Q


print("-"*50)
print(f"Source MAE (p2p): {p2p:.4f}")
print(f"Target MAE (q2q): {q2q:.4f}")
print(f"p2s: expected Source Error on Sx (Reweighted Source Error)  : {p2s:.4f}")
print(f"s2q: expected Target Error on Sx (Reweighted Target Error)  : {s2q:.4f}")
print(f"Total Degradation: {total_degradation:.4f}")
print(f"Y|X Shift: {yx_shift:.4f}")
print(f"X Shift (P side): {x_shift_P:.4f}")
print(f"X Shift (Q side): {x_shift_Q:.4f}")
print(f"Total X Shift : {total_x_shift:.4f}")
print(f"X Shift P side ratio: {x_shift_P/total_degradation:.4f}")
print(f"X Shift Q side ratio: {x_shift_Q/total_degradation:.4f}")
print(f"X Shift Ratio : {total_x_shift/total_degradation:.4f}")
print(f"Y|X Shift Ratio: {yx_shift/total_degradation:.4f}")
print("-"*50)

from whyshift.region_analysis import shared_reweight

wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

# Train separate regressors
source_model = lgbm.LGBMRegressor(**lgb_params)
target_model = lgbm.LGBMRegressor(**lgb_params)

source_model.fit(X_source_train, y_source_train, sample_weight=wA)
target_model.fit(X_train_target, y_train_target, sample_weight=wB)

# Disagreement in predictions
pred_src = source_model.predict(new_X)
pred_tgt = target_model.predict(new_X)

new_Y = np.abs(pred_src - pred_tgt)

print("\n>>> Model MAE Comparison:")

# Source model
print(f"Source model MAE on ID test:  {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
print(f"Source model MAE on OOD test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")

# Target model
print(f"Target model MAE on ID test:  {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
print(f"Target model MAE on OOD test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")
region_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=100,
    min_weight_fraction_leaf=0.05
)

region_tree.fit(new_X, new_Y, sample_weight=new_weights)



fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(["Source MAE", "Target MAE"], [p2p, q2q], alpha=0.6)
ax.set_title("Walmart Sales: MAE Degradation (Weather)")
plt.tight_layout()
plt.savefig("walmart_regression_weather_decomposition.png")
plt.show()

# Risk Tree
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("walmart_regression_risk_weather_tree.png", bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

base_val = min(p2p, q2q)
max_val = max(p2p, q2q)
gap = max_val - base_val

shift_y = abs(p2s - s2q)
shift_p = abs(p2p - p2s)
shift_q = abs(s2q - q2q)
total_shift = shift_y + shift_p + shift_q + 1e-9 #

norm_y = (shift_y / total_shift) * gap
norm_p = (shift_p / total_shift) * gap
norm_q = (shift_q / total_shift) * gap

ax.bar(["Source", "Target"], [p2p, q2q], color='blue', alpha=0.5)

#ax.bar(["Decomposition"], [base_val], color='blue', alpha=0.5)

ax.bar(["Decomposition"], [norm_y], bottom=base_val, color='orange', label="Concept Shift (Y|X)")
ax.bar(["Decomposition"], [norm_p], bottom=base_val + norm_y, color='#AA0000', label="Covariate Shift (X) (P)")
ax.bar(["Decomposition"], [norm_q], bottom=base_val + norm_y + norm_p, color='red', label="Covariate Shift (X) (Q)")

ax.set_title("Walmart Sales: Weather Decomposition (Regression DISDE)")
ax.set_ylabel("MAE")
ax.legend()
plt.tight_layout()
plt.savefig('walmart_regression_weather_decomposition_bar.png')
plt.show()