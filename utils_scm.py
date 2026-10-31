import os
import pickle
from pathlib import Path
from datetime import datetime
import networkx as nx
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import dowhy.gcm as gcm
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree, DecisionTreeRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from whyshift.region_analysis import shared_reweight
from whyshift_reg.disde_reg import degradation_decomp_regression


def save_checkpoint(scm, feature_graph, parents, metrics, checkpoint_dir, stage_name="checkpoint"):
    try:
        Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scm_path = Path(checkpoint_dir) / f"scm_{stage_name}_{timestamp}.pkl"
        with open(scm_path, "wb") as f:
            pickle.dump(scm, f)

        metadata = {
            "timestamp": timestamp,
            "stage": stage_name,
            "parents": parents,
            "metrics": metrics,
            "feature_count": len(parents),
            "nodes": list(feature_graph.nodes()),
            "edges": list(feature_graph.edges())
        }
        metadata_path = Path(checkpoint_dir) / f"metadata_{stage_name}_{timestamp}.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        print(f" Checkpoint saved: {scm_path.name}")
        return str(scm_path), str(metadata_path)
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        raise


def load_checkpoint(scm_path, metadata_path):
    try:
        with open(scm_path, "rb") as f:
            scm = pickle.load(f)
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        print(f" Checkpoint loaded from {Path(scm_path).name}")
        return scm, metadata
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        raise


def get_all_descendants(graph, source_node):
    descendants = set()
    to_visit = [source_node]
    visited = set()
    while to_visit:
        node = to_visit.pop(0)
        if node in visited:
            continue
        visited.add(node)
        children = list(graph.successors(node))
        descendants.update(children)
        to_visit.extend(children)
    return descendants


def setup_mechanisms(scm, feature_graph, classifier_nodes, lgb_params):
    for node in feature_graph.nodes:
        parents = list(feature_graph.predecessors(node))
        if len(parents) == 0:
            scm.set_causal_mechanism(node, gcm.EmpiricalDistribution())
        elif node in classifier_nodes:
            scm.set_causal_mechanism(
                node, gcm.ClassifierFCM(gcm.ml.SklearnClassificationModel(LGBMClassifier(**lgb_params)))
            )
        else:
            scm.set_causal_mechanism(
                node, gcm.AdditiveNoiseModel(gcm.ml.SklearnRegressionModel(LGBMRegressor(**lgb_params)))
            )


def print_target_stats(df, target_col, title):
    """Prints summary statistics of the target column for any dataset."""
    print(f"\n>>> {title} :")
    print(f"{target_col} stats:")
    print(f"  Mean: {df[target_col].mean():.4f}")
    print(f"  Std:  {df[target_col].std():.4f}")
    print(f"  Min:  {df[target_col].min():.4f}")
    print(f"  Max:  {df[target_col].max():.4f}")
    print(f"  % negative: {(df[target_col] < 0).mean() * 100:.2f}%")


def evaluate_sales_on_holdout(model_scm, feature_graph, eval_df, label, target_col):
    parents = sorted(list(feature_graph.predecessors(target_col)))
    sales_mechanism = model_scm.causal_mechanism(target_col)

    X_eval = eval_df[parents].to_numpy()
    y_true = eval_df[target_col].to_numpy()
    y_pred = sales_mechanism.prediction_model.predict(X_eval).flatten()

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

    print(f"\n{label}")
    print(f"Target: {target_col}")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAPE: {metrics['MAPE']:.2f}")
    print(f"R²:   {metrics['R2']:.4f}")

    return metrics


def generate_synthetic_data(scm, interventions_dict, num_samples, original_df, classifier_nodes,
                            target_col, clip_cols=None):
    """Generates synthetic interventional data and restores appropriate formatting."""
    print(f"Generating {num_samples} synthetic samples...")
    synthetic_dataset = gcm.interventional_samples(
        scm,
        interventions=interventions_dict,
        num_samples_to_draw=num_samples
    )

    # Clip continuous variables that cannot be negative
    if clip_cols:
        synthetic_dataset[clip_cols] = synthetic_dataset[clip_cols].clip(lower=0)

    # Re-align categorical factor levels matching the global vocabulary
    for col in classifier_nodes:
        if col in original_df.columns:
            all_categories = original_df[col].astype(str).unique()
            synthetic_dataset[col] = pd.Categorical(synthetic_dataset[col].astype(str), categories=all_categories)

    print(f"✓ Synthetic Dataset Shape: {synthetic_dataset.shape}")
    print_target_stats(synthetic_dataset, target_col, "RAW SYNTHETIC DATA")

    return synthetic_dataset


def create_hybrid_dataset(source_df, synthetic_dataset, original_df, classifier_nodes, target_col):
    """Fuses observational source sequences with synthetic interventional sequences."""
    train_aligned = source_df[synthetic_dataset.columns].copy()
    hybrid_dataset = pd.concat([train_aligned, synthetic_dataset], ignore_index=True)

    # Maintain factor integrity across concatenated feature space
    for col in classifier_nodes:
        if col in original_df.columns:
            all_categories = original_df[col].astype(str).unique()
            hybrid_dataset[col] = pd.Categorical(hybrid_dataset[col].astype(str), categories=all_categories)

    print(f"✓ Hybrid Dataset Shape: {hybrid_dataset.shape}")
    print_target_stats(hybrid_dataset, target_col, "RAW HYBRID DATA")

    return hybrid_dataset


def run_whyshift_region_analysis(df_source, df_target, target_col, categorical_nodes, lgb_params,
                                 source_name, target_name, plot_filename, random_seed=42):
    """
    Executes the full Whyshift Degradation Decomposition and Risk Region Analysis pipeline.
    """
    print(f"\n{'=' * 60}")
    print(f"REGION ANALYSIS: {source_name} vs {target_name}")
    print(f"{'=' * 60}")

    X_src = df_source.drop(columns=[target_col])
    X_tgt = df_target.drop(columns=[target_col])

    # ONE HOT ENCODING (CRITICAL)
    combined = pd.concat([X_src, X_tgt], axis=0)
    combined_encoded = pd.get_dummies(combined, columns=[c for c in categorical_nodes if c in combined.columns])

    X_enc = combined_encoded.iloc[:len(X_src)]
    X_target_enc = combined_encoded.iloc[len(X_src):]

    feature_names = X_enc.columns.tolist()

    X_source_full = X_enc.values
    X_target_full = X_target_enc.values

    y_source_full = df_source[target_col].values
    y_target_full = df_target[target_col].values

    # Train/Test splits for both domains (70/30)
    np.random.seed(random_seed)
    perm_src = np.random.permutation(len(X_source_full))
    train_idx = perm_src[:int(0.7 * len(X_source_full))]
    test_idx = perm_src[int(0.7 * len(X_source_full)):]

    X_source_train, y_source_train = X_source_full[train_idx], y_source_full[train_idx]
    X_source_test, y_source_test = X_source_full[test_idx], y_source_full[test_idx]

    np.random.seed(random_seed)
    perm_tgt = np.random.permutation(len(X_target_full))
    train_tgt_idx = perm_tgt[:int(0.7 * len(X_target_full))]
    test_tgt_idx = perm_tgt[int(0.7 * len(X_target_full)):]

    X_train_target, y_train_target = X_target_full[train_tgt_idx], y_target_full[train_tgt_idx]
    X_target_test, y_target_test = X_target_full[test_tgt_idx], y_target_full[test_tgt_idx]

    # Baseline Model for Degradation
    model_disde = LGBMRegressor(**lgb_params)
    model_disde.fit(X_source_train, y_source_train)

    reg_domain_classifier = LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        random_state=random_seed, class_weight='balanced',
        use_label_encoder=False, eval_metric='logloss'
    )

    print(f"\n>>> Running Degradation Decomposition ({source_name} -> {target_name})...")
    p2p, q2q, p2s, s2q = degradation_decomp_regression(
        X_source_full, y_source_full, X_target_full, y_target_full,
        model_disde, data_sum=min(20000, len(X_source_full), len(X_target_full)),
        K=8, domain_classifier=reg_domain_classifier, draw_calibration=False
    )

    print(f"Total Performance Degradation (p2p - q2q): {p2p - q2q:.4f}")
    print(f"Proportion of Y|X-shift:                  {(p2s - s2q) / (p2p - q2q):.4f}")
    print(f"Proportion of X-shift:                    {(p2p - p2s + s2q - q2q) / (p2p - q2q):.4f}")

    # Importance Sampling Reweighting
    wA, wB, new_X, new_weights = shared_reweight(X_source_train, X_train_target, K=8)

    # Train Reweighted Models
    source_model = LGBMRegressor(**lgb_params).fit(X_source_train, y_source_train, sample_weight=wA)
    target_model = LGBMRegressor(**lgb_params).fit(X_train_target, y_train_target, sample_weight=wB)

    pred_src, pred_tgt = source_model.predict(new_X), target_model.predict(new_X)
    new_Y = np.abs(pred_src - pred_tgt)

    print("\n>>> Model MAE Comparison:")
    print(
        f"[{source_name} Model] MAE on {source_name} (ID) Test: {mean_absolute_error(y_source_test, source_model.predict(X_source_test)):.4f}")
    print(
        f"[{source_name} Model] MAE on {target_name} (OOD) Test: {mean_absolute_error(y_target_test, source_model.predict(X_target_test)):.4f}")
    print(
        f"[{target_name} Model] MAE on {source_name} (ID) Test: {mean_absolute_error(y_source_test, target_model.predict(X_source_test)):.4f}")
    print(
        f"[{target_name} Model] MAE on {target_name} (OOD) Test: {mean_absolute_error(y_target_test, target_model.predict(X_target_test)):.4f}")

    # Risk Region Tree
    region_tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, min_weight_fraction_leaf=0.05)
    region_tree.fit(new_X, new_Y, sample_weight=new_weights)

    plt.figure(figsize=(25, 12))
    plot_tree(region_tree, filled=True, feature_names=feature_names, fontsize=10)
    plt.title(f"Risk Regions (Prediction Disagreement: {source_name} vs {target_name})")
    plt.savefig(plot_filename)
    plt.close()  # Free memory
    print(f"Saved region analysis plot to: {plot_filename}")


from sklearn.tree import _tree
import numpy as np


def extract_high_shift_rules(tree, feature_names, threshold_percentile=80):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            path.append(f"`{name}` <= {threshold:.4f}")
            recurse(tree_.children_left[node], path)
            path.pop()

            path.append(f"`{name}` > {threshold:.4f}")
            recurse(tree_.children_right[node], path)
            path.pop()
        else:
            paths.append((path[:], tree_.value[node][0][0], tree_.n_node_samples[node]))

    recurse(0, [])

    values = [p[1] for p in paths]
    threshold_val = np.percentile(values, threshold_percentile)

    high_shift_rules = [p for p in paths if p[1] >= threshold_val]
    high_shift_rules.sort(key=lambda x: x[1], reverse=True)

    return high_shift_rules