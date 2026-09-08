"""
(1) build_clusters -- groups flagged near-duplicate pairs from
    sequence_leakage_audit.py into clusters via connected components,
    at a chosen identity threshold (80% default).

(2) grouped_loo_squared_errors -- leave-cluster-out cross-validation:
    when evaluating any cluster member, every other member of that
    cluster is also excluded from the training fold.

Usage:
    python cluster_and_grouped_cv.py \
        --leakage-audit leakage_audit_results.json \
        --results dispersion_error_results_deduped.json \
        --threshold 0.80 \
        --output grouped_cv_results.json
"""
import argparse
import json

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


def build_clusters(flagged_pairs, all_targets):
    parent = {t: t for t in all_targets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for a, b, *_ in flagged_pairs:
        if a in parent and b in parent:
            union(a, b)

    clusters = {}
    for t in all_targets:
        root = find(t)
        clusters.setdefault(root, []).append(t)
    return list(clusters.values())


def grouped_loo_squared_errors(log_length, log_dispersion, log_error, cluster_id, spline_df=4):
    n = len(log_length)
    se_0 = np.zeros(n)
    se_1 = np.zeros(n)
    full_data = pd.DataFrame({"log_length": log_length})
    basis_info = patsy.dmatrix(f"cr(log_length, df={spline_df}) - 1", full_data,
                                 return_type="dataframe").design_info

    for i in range(n):
        my_cluster = cluster_id[i]
        train_idx = [j for j in range(n) if cluster_id[j] != my_cluster]

        train_data = pd.DataFrame({"log_length": log_length[train_idx]})
        X_train = patsy.build_design_matrices([basis_info], train_data, return_type="dataframe")[0]
        X_train["log_dispersion"] = log_dispersion[train_idx]
        model_1 = sm.OLS(log_error[train_idx], X_train).fit()

        X_train_0 = X_train.drop(columns=["log_dispersion"])
        model_0 = sm.OLS(log_error[train_idx], X_train_0).fit()

        test_data = pd.DataFrame({"log_length": log_length[[i]]})
        X_test = patsy.build_design_matrices([basis_info], test_data, return_type="dataframe")[0]
        X_test["log_dispersion"] = log_dispersion[[i]]
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        X_test_0 = X_test.drop(columns=["log_dispersion"])
        X_test_0 = X_test_0.reindex(columns=X_train_0.columns, fill_value=0.0)

        pred_1 = model_1.predict(X_test).values[0]
        pred_0 = model_0.predict(X_test_0).values[0]
        se_0[i] = (pred_0 - log_error[i]) ** 2
        se_1[i] = (pred_1 - log_error[i]) ** 2
    return se_0, se_1


def run_grouped_analysis(length, dispersion, error, targets, flagged_pairs, threshold,
                           n_boot=10000, seed=0):
    clusters = build_clusters(flagged_pairs, targets)
    n_clusters = len(clusters)
    n_multi = sum(1 for c in clusters if len(c) > 1)
    largest = max(len(c) for c in clusters)

    target_to_cluster = {}
    for cid, members in enumerate(clusters):
        for m in members:
            target_to_cluster[m] = cid
    cluster_id = np.array([target_to_cluster[t] for t in targets])

    log_length = np.log(length)
    log_dispersion = np.log(dispersion)
    log_error = np.log(error)

    se_0, se_1 = grouped_loo_squared_errors(log_length, log_dispersion, log_error, cluster_id)
    d = se_0 - se_1

    rng = np.random.default_rng(seed)
    n = len(d)
    boot_means = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    ci = np.percentile(boot_means, [2.5, 97.5])
    signs = rng.choice([-1, 1], size=(n_boot, n))
    perm_means = (signs * d).mean(axis=1)
    perm_p = np.mean(np.abs(perm_means) >= np.abs(d.mean()))

    return {
        "n_clusters": n_clusters,
        "n_multi_member_clusters": n_multi,
        "largest_cluster_size": largest,
        "rmse_length_only_grouped": float(np.sqrt(se_0.mean())),
        "rmse_length_plus_dispersion_grouped": float(np.sqrt(se_1.mean())),
        "mean_squared_error_diff": float(d.mean()),
        "bootstrap_95ci_diff": [float(ci[0]), float(ci[1])],
        "paired_permutation_p": float(perm_p),
        "clusters": clusters,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leakage-audit", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output", default="grouped_cv_results.json")
    args = parser.parse_args()

    with open(args.leakage_audit) as f:
        audit = json.load(f)
    with open(args.results) as f:
        data = json.load(f)

    threshold_key = str(args.threshold)
    flagged_pairs = None
    for k in audit["within_cohort_flagged"]:
        if abs(float(k) - args.threshold) < 1e-6:
            flagged_pairs = audit["within_cohort_flagged"][k]
            break
    if flagged_pairs is None:
        raise ValueError(f"No flagged-pairs entry found for threshold {args.threshold} "
                          f"in {args.leakage_audit}")

    targets = [d["target"] for d in data]
    length = np.array([d["length"] for d in data])
    dispersion = np.array([d["dispersion"] for d in data])
    error = np.array([d["error"] for d in data])

    results = run_grouped_analysis(length, dispersion, error, targets, flagged_pairs, args.threshold)

    print(f"Threshold: {args.threshold}")
    print(f"Clusters: {results['n_clusters']} total, "
          f"{results['n_multi_member_clusters']} with >1 member, "
          f"largest has {results['largest_cluster_size']} members")
    print(f"RMSE length-only (grouped): {results['rmse_length_only_grouped']:.4f}")
    print(f"RMSE length+dispersion (grouped): {results['rmse_length_plus_dispersion_grouped']:.4f}")
    print(f"Mean squared-error diff: {results['mean_squared_error_diff']:.5f}")
    print(f"Bootstrap 95% CI: {results['bootstrap_95ci_diff']}")
    print(f"Permutation p: {results['paired_permutation_p']:.4f}")

    with open(args.output, "w") as f:
        json.dump(results, f)
    print(f"\nSaved to {args.output}")
