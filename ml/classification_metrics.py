"""Threshold-based classification metrics (accuracy, precision, recall, specificity, F1) plus ROC-AUC.

Uses the repeat-averaged out-of-fold probabilities written by reproduce.py (ml/oof_predictions.csv), so these
numbers come from the same paired cross-validation as ml/metrics.json. Threshold fixed at 0.5 before looking at
results (no threshold tuning on the evaluation data). 95% CIs: 2,000 patient-level bootstrap resamples, seed 2026,
as in reproduce.py.

Run:  python ml/classification_metrics.py   ->  ml/classification_metrics.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5
SEED, N_BOOT = 2026, 2000
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
FAMILIES = ["LR", "RF"]


def specificity(y, yhat):
    neg = y == 0
    return float((yhat[neg] == 0).mean()) if neg.any() else float("nan")


METRICS = {
    "accuracy": lambda y, p, yh: accuracy_score(y, yh),
    "precision": lambda y, p, yh: precision_score(y, yh, zero_division=0),
    "recall": lambda y, p, yh: recall_score(y, yh, zero_division=0),
    "specificity": lambda y, p, yh: specificity(y, yh),
    "f1": lambda y, p, yh: f1_score(y, yh, zero_division=0),
    "roc_auc": lambda y, p, yh: roc_auc_score(y, p),
}


def evaluate(y: np.ndarray, p: np.ndarray, boot: list[np.ndarray]) -> dict:
    yh = (p >= THRESHOLD).astype(int)
    out = {}
    for name, fn in METRICS.items():
        point = float(fn(y, p, yh))
        vals = [fn(y[b], p[b], yh[b]) for b in boot if len(set(y[b])) == 2]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = {"value": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)]}
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    fn_ = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    out["confusion"] = {"tp": tp, "fp": fp, "fn": fn_, "tn": tn}
    out["prevalence"] = round(float(y.mean()), 3)
    return out


def main() -> None:
    df = pd.read_csv(ROOT / "ml" / "oof_predictions.csv")
    n = len(df)
    rng = np.random.default_rng(SEED)
    boot = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    res = {"threshold": THRESHOLD, "source": "ml/oof_predictions.csv (reproduce.py, all inputs, repeat-averaged OOF)",
           "bootstrap": N_BOOT, "seed": SEED, "n": n}
    for fam in FAMILIES:
        for t in TARGETS:
            res.setdefault(fam, {})[t] = evaluate(df[f"y_{t}"].to_numpy(), df[f"{fam}_all_{t}"].to_numpy(), boot)
    (ROOT / "ml" / "classification_metrics.json").write_text(json.dumps(res, indent=1))
    for fam in FAMILIES:
        for t in TARGETS:
            r = res[fam][t]
            print(fam, t, {k: r[k]["value"] for k in METRICS})


if __name__ == "__main__":
    main()
