from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import lightgbm as lgbm
from whyshift_reg.disde_reg import *
import sys
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

CSV_PATH = r"../../walmart_dataset/final_data_walmart.csv"

print(">>> Loading Walmart Dataset from CSV...")
df_raw = pd.read_csv(CSV_PATH)


df_raw.columns = df_raw.columns.str.strip()
means = df_raw.groupby("Is_Christmas_Season")["Weekly_Sales"].mean()
std = df_raw.groupby("Is_Christmas_Season")["Weekly_Sales"].std()
print(means)
print(std)

cols_to_encode = ['Store', 'weather_condition', 'city', 'Dept', 'Type']
df_encoded = pd.get_dummies(df_raw, columns=cols_to_encode)

# Source Domain: Normal Days (In-Distribution)
# Target Domain: Holidays (Out-of-Distribution)
source_df = df_encoded[df_encoded['Is_Christmas_Season'] == 0].copy()
target_df = df_encoded[df_encoded['Is_Christmas_Season'] == 1].copy()

# Define feature columns (exclude target, proxies, and the splitting variable)
drop_cols = [
    'Weekly_Sales', 'Date',
    'IsHoliday','holiday_proximity','is_near_holiday','holiday_weight','Is_Christmas_Season','Month','WeekOfYear',
    'Quarter','Season','Year_Progress','Month_Progress','WeekOfYear_sin','WeekOfYear_cos','Month_sin','Month_cos','Season_sin','Season_cos','Is_Month_Start',
    'Is_Month_End','DayOfWeek','DayOfWeek_sin','DayOfWeek_cos','MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'
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



# def degradation_decomp_regression(
#     source_X,
#     source_y,
#     target_X_raw,
#     target_y_raw,
#     model,
#     data_sum=20000,
#     K=8,
#     domain_classifier=None,draw_calibration=False, save_calibration_png='calibration.png'
# ):
#
#     perm1 = np.random.permutation(target_X_raw.shape[0])
#     target_X = target_X_raw[perm1[:data_sum], :]
#     target_y = target_y_raw[perm1[:data_sum]]
#
#
#     piA = np.zeros(source_X.shape[0])
#     piB = np.zeros(target_X.shape[0])
#
#     permA = np.random.permutation(source_X.shape[0])
#     permB = np.random.permutation(target_X.shape[0])
#
#     kf = KFold(n_splits=K, shuffle=False)
#
#     A_train_index_list, A_test_index_list = [], []
#     B_train_index_list, B_test_index_list = [], []
#
#     for train_idx, test_idx in kf.split(source_X):
#         A_train_index_list.append(train_idx)
#         A_test_index_list.append(test_idx)
#
#     for train_idx, test_idx in kf.split(target_X):
#         B_train_index_list.append(train_idx)
#         B_test_index_list.append(test_idx)
#
#     for i in range(K):
#         trainX = np.concatenate([
#             source_X[permA[A_train_index_list[i]]],
#             target_X[permB[B_train_index_list[i]]]
#         ], axis=0)
#
#         trainT = np.zeros(trainX.shape[0])
#         trainT[len(A_train_index_list[i]):] = 1.0
#
#         if domain_classifier is None:
#             clf = XGBClassifier(random_state=0)
#         else:
#             clf = domain_classifier
#
#         clf.fit(trainX, trainT)
#
#         piA[permA[A_test_index_list[i]]] = clf.predict_proba(
#             source_X[permA[A_test_index_list[i]]]
#         )[:, 1]
#
#         piB[permB[B_test_index_list[i]]] = clf.predict_proba(
#             target_X[permB[B_test_index_list[i]]]
#         )[:, 1]
#
#     if draw_calibration:
#         plot_calibration(piA, piB, save_dir=save_calibration_png)
#     alpha = target_X.shape[0] / (source_X.shape[0] + target_X.shape[0])
#
#     wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
#     wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))
#
#     wA /= np.sum(wA)
#     wB /= np.sum(wB)
#
#
#     pred_source = model.predict(source_X)
#     pred_target = model.predict(target_X)
#
#     loss_source = np.abs(pred_source - source_y)
#     loss_target = np.abs(pred_target - target_y)
#
#     # empirical risks
#     errorA = np.mean(loss_source)
#     errorB = np.mean(loss_target)
#
#     # weighted risks
#     sx_A = np.dot(wA, loss_source)
#     sx_B = np.dot(wB, loss_target)
#
#     return errorA,errorB,sx_A,sx_B
#
# def plot_calibration(prop_p, prop_q, nbins=20, p_weights=None, q_weights=None,
#                      nanmask_threshold=0.01, name='Prop Score',
#                      save_dir='.', balance=False):
#
#     fig,ax=plt.subplots(1,3,figsize=(10,4))
#     for i in range(3):
#         ax[i].set_box_aspect(1)
#         ax[i].set_xlim(0,1)
#     fig.suptitle("Calibration: {}".format(name), fontsize="x-large")
#
#     if p_weights is None: p_weights = np.ones_like(prop_p)
#     if q_weights is None: q_weights = np.ones_like(prop_q)
#
#     p_sample_weights = p_weights.copy()
#     q_sample_weights = q_weights.copy()
#     if balance:
#         p_sample_weights = p_sample_weights / p_sample_weights.sum()
#         q_sample_weights = q_sample_weights / q_sample_weights.sum()
#
#     conf_scores, bin_edges = np.histogram(np.concatenate([1-prop_p, prop_q]),bins=nbins, density=True,
#                                           weights=np.concatenate([p_sample_weights,
#                                                                  q_sample_weights]),
#                                          range=(0,1))
#     bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
#
#     nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
#     # print(bin_mids)
#     # print(nanmask * conf_scores / (conf_scores + conf_scores[::-1]))
#     ax[0].plot(bin_mids, nanmask * conf_scores / (conf_scores + conf_scores[::-1]), color='green')
#     ax[0].set_ylim(0,1)
#     ax[0].set_ylabel('Proportion correct')
#     ax[0].set_xlabel('Predicted probability')
#     ax[0].set_title('Prop calibration: combined')
#
#     conf_scores, bin_edges = np.histogram(np.concatenate([prop_p]),bins=nbins, weights=p_sample_weights,
#                                          range=(0,1))
#     bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
#     nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
#
#     ax[1].plot(bin_mids, conf_scores, color='green')
#     ax[1].set_title('Density: P')
#     ax[1].set_xlabel('Predicted probability of Q')
#     ax[1].set_ylim(bottom=0)
#
#     conf_scores, bin_edges = np.histogram(np.concatenate([prop_q]),bins=nbins, weights=q_sample_weights,
#                                          range=(0,1))
#     bin_mids = (bin_edges[1:]+bin_edges[:-1])/2
#     nanmask = np.where(conf_scores < nanmask_threshold, np.nan, 1)
#
#     ax[2].plot(bin_mids, conf_scores, color='green')
#     ax[2].set_title('Density: Q')
#     ax[2].set_xlabel('Predicted probability of Q')
#     ax[2].set_ylim(bottom=0)
#
#     fig.tight_layout()
#     if save_dir is not None:
#         plt.savefig(save_dir)

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
    model, data_sum=20000, domain_classifier = reg_domain_classifier, K=8,draw_calibration=True, save_calibration_png='Is_Christmas_Season_regression_calibration.png'
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
ax.set_title("Walmart Sales: MAE Degradation (Normal → Holiday)")
plt.tight_layout()
plt.savefig("walmart_regression_Is_Christmas_Season_decomposition.png")
plt.show()

# Risk Tree
plt.figure(figsize=(25, 12))
plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
plt.title("Risk Regions (Prediction Disagreement)")
plt.savefig("walmart_regression_risk_Is_Christmas_Season_tree.png", bbox_inches='tight')
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

ax.set_title("Walmart Sales: Christmas Decomposition (Regression DISDE)")
ax.set_ylabel("MAE")
ax.legend()
plt.tight_layout()
plt.savefig('walmart_regression_Is_Christmas_Season_decomposition_bar.png')
plt.show()
