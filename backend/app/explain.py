"""Exact SHAP values in plain numpy, so the served API needs neither the shap library nor pandas.

- Linear model, independent masker (what shap.LinearExplainer does with maskers.Independent):
      phi_j = coef_j * (x_j - mean_j(background)), in the scaler's standardised space, log-odds scale.
- Random forest, path-dependent TreeSHAP (Lundberg et al. 2018, Algorithm 2; what shap.TreeExplainer does
  with no background data), class-1 probability scale, averaged over trees.

tests/test_explain.py checks both against the shap library on every preset and on random rows.
"""
from __future__ import annotations

import numpy as np


def linear_shap(scaler, lr, background: np.ndarray):
    mean_bg = scaler.transform(background).mean(axis=0)
    coef = lr.coef_[0]

    def run(x: np.ndarray) -> np.ndarray:
        return coef * (scaler.transform(x.reshape(1, -1))[0] - mean_bg)
    return run


def _tree_arrays(est):
    t = est.tree_
    v = t.value[:, 0, :]
    v = v / v.sum(axis=1, keepdims=True)  # class fractions (a no-op on sklearn >= 1.4)
    return (t.children_left.tolist(), t.children_right.tolist(), t.feature.tolist(), t.threshold.tolist(),
            t.weighted_n_node_samples.tolist(), v[:, 1].tolist())


def _tree_shap(x: list[float], left, right, feat, thr, cover, val, phi: list[float]) -> None:
    # Path entries: [feature, zero_fraction, one_fraction, weight]
    def extend(path, z, o, i):
        path = [p[:] for p in path] + [[i, z, o, 1.0 if not path else 0.0]]
        n = len(path) - 1
        for k in range(n - 1, -1, -1):
            path[k + 1][3] += o * path[k][3] * (k + 1) / (n + 1)
            path[k][3] = z * path[k][3] * (n - k) / (n + 1)
        return path

    def unwind(path, idx):
        path = [p[:] for p in path]
        n = len(path) - 1
        o, z = path[idx][2], path[idx][1]
        nxt = path[n][3]
        for k in range(n - 1, -1, -1):
            if o != 0:
                tmp = path[k][3]
                path[k][3] = nxt * (n + 1) / ((k + 1) * o)
                nxt = tmp - path[k][3] * z * (n - k) / (n + 1)
            else:
                path[k][3] = path[k][3] * (n + 1) / (z * (n - k))
        for k in range(idx, n):
            path[k][0], path[k][1], path[k][2] = path[k + 1][0], path[k + 1][1], path[k + 1][2]
        return path[:-1]

    def unwound_sum(path, idx):
        n = len(path) - 1
        o, z = path[idx][2], path[idx][1]
        total, nxt = 0.0, path[n][3]
        for k in range(n - 1, -1, -1):
            if o != 0:
                tmp = nxt * (n + 1) / ((k + 1) * o)
                total += tmp
                nxt = path[k][3] - tmp * z * (n - k) / (n + 1)
            else:
                total += path[k][3] / z * (n + 1) / (n - k)
        return total

    def recurse(j, path, z, o, i):
        path = extend(path, z, o, i)
        if left[j] == -1:
            for k in range(1, len(path)):
                w = unwound_sum(path, k)
                phi[path[k][0]] += w * (path[k][2] - path[k][1]) * val[j]
            return
        f = feat[j]
        hot, cold = (left[j], right[j]) if x[f] <= thr[j] else (right[j], left[j])
        iz = io = 1.0
        for k in range(1, len(path)):
            if path[k][0] == f:
                iz, io = path[k][1], path[k][2]
                path = unwind(path, k)
                break
        recurse(hot, path, iz * cover[hot] / cover[j], io, f)
        recurse(cold, path, iz * cover[cold] / cover[j], 0.0, f)

    recurse(0, [], 1.0, 1.0, -1)


def forest_shap(forest):
    trees = [_tree_arrays(e) for e in forest.estimators_]
    n_features = forest.n_features_in_

    def run(x: np.ndarray) -> np.ndarray:
        xs = np.asarray(x, dtype=np.float32).astype(float).tolist()  # sklearn compares float32 thresholds
        phi = [0.0] * n_features
        for tr in trees:
            _tree_shap(xs, *tr, phi)
        return np.asarray(phi) / len(trees)
    return run
