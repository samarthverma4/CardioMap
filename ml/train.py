"""Fit the served models and write everything the app reads into artifacts/.

Run:  python ml/train.py
Out:  artifacts/models.joblib, feature_schema.json, metrics.json, anatomy_map.json, presets.json

Design:
- Candidate families are the two prespecified in reproduce.py (same hyperparameters).
- Per target, LR vs RF is chosen by a separate 5-fold CV on the training data (all 303 rows, since the final
  model is fit on all of them). The reported numbers in metrics.json are never used for selection.
- Final served model: CalibratedClassifierCV(chosen, method="sigmoid", cv=5) fit on all 303 rows.
- Variability band: 5 calibrated models, one per training fold of the selection split; the UI shows min-max.
- SHAP base: the uncalibrated chosen model refit on all 303 rows (explanations are of this, not the calibrated wrapper).
- ESC baseline: calibrated LR on the 12 ESC features, CAD only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ROOT = F.ROOT
ART = ROOT / "artifacts"
SEED = 2026
CANDIDATES = {
    "LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
}
EVIDENCE_THRESHOLDS = {"weak": "< 0.60", "limited": "0.60-0.70", "moderate": "> 0.70"}


def evidence_badge(auc: float) -> str:
    if auc < 0.60:
        return "weak"
    if auc <= 0.70:
        return "limited"
    return "moderate"


def calibrated(base):
    return CalibratedClassifierCV(clone(base), method="sigmoid", cv=5)


def anatomy_map() -> dict:
    return {
        "dominance_assumption": "assumes right-dominant circulation",
        "note": "Generic textbook territories. The 3D heart is a generic model, not this patient's anatomy.",
        "aha_17_segments": {
            "LAD": [1, 2, 7, 8, 13, 14, 17],
            "RCA": [3, 4, 9, 10, 15],
            "LCX": [5, 6, 11, 12, 16],
        },
        "vessels": {
            "LAD": {"name": "Left anterior descending artery",
                    "course": "Anterior interventricular groove to the apex",
                    "territory": "Anterior wall, anterior septum, apex"},
            "LCX": {"name": "Left circumflex artery",
                    "course": "Left atrioventricular groove towards the back",
                    "territory": "Lateral and posterolateral wall"},
            "RCA": {"name": "Right coronary artery",
                    "course": "Right atrioventricular groove to the inferior wall",
                    "territory": "Inferior wall, inferior septum"},
        },
    }


def pick_presets(df: pd.DataFrame, Y: dict, exclude=(93,)) -> list[dict]:
    """First dataset row (by index) matching each case definition. Rule fixed in code, not chosen by model output."""
    nves = Y["LAD"] + Y["LCX"] + Y["RCA"]
    ok = ~np.isin(np.arange(len(df)), exclude)
    cases = [
        ("no_cad", "Sample A", "No CAD on angiography", (Y["CAD"] == 0) & (nves == 0)),
        ("single_lad", "Sample B", "Single-vessel disease (LAD)",
         (Y["CAD"] == 1) & (Y["LAD"] == 1) & (nves == 1)),
        ("multi_vessel", "Sample C", "Three-vessel disease", (Y["CAD"] == 1) & (nves == 3)),
        ("lcx_rca", "Sample D", "LCX/RCA disease without LAD",
         (Y["CAD"] == 1) & (Y["LAD"] == 0) & ((Y["LCX"] == 1) | (Y["RCA"] == 1))),
    ]
    C = F.clinical_frame(df)
    out = []
    for pid, name, desc, m in cases:
        row = int(np.where(m & ok)[0][0])
        feats = {k: (v if isinstance(v, str) else (int(v) if float(v).is_integer() else round(float(v), 4)))
                 for k, v in C.iloc[row].items()}
        out.append({
            "id": pid, "name": name, "dataset_row": row, "features": feats,
            "hidden_labels": {"description": desc, "CAD": int(Y["CAD"][row]),
                              **{v: int(Y[v][row]) for v in F.VESSELS}},
            "in_training_data": True,
        })
    return out


def main() -> None:
    ART.mkdir(exist_ok=True)
    df = F.load_raw()
    X = F.encode(df)
    Y = F.targets(df)
    cols = list(X.columns)
    F.assert_no_labels(cols)
    nves = Y["LAD"] + Y["LCX"] + Y["RCA"]
    split = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, nves))

    models, selection = {}, {}
    for t in F.TARGETS:
        y = Y[t]
        cv_auc = {}
        for name, base in CANDIDATES.items():
            oof = np.zeros(len(X))
            for tr, te in split:
                oof[te] = clone(base).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            cv_auc[name] = round(float(roc_auc_score(y, oof)), 4)
        chosen = max(cv_auc, key=cv_auc.get)
        base = CANDIDATES[chosen]
        final = calibrated(base).fit(X, y)
        shap_base = clone(base).fit(X, y)
        folds = [calibrated(base).fit(X.iloc[tr], y[tr]) for tr, _ in split]
        models[t] = {"family": chosen, "features": cols, "model": final, "shap_base": shap_base,
                     "fold_models": folds, "selection_cv_auc": cv_auc}
        selection[t] = {"chosen": chosen, "inner_cv_auc": cv_auc}
        print(f"{t}: {cv_auc} -> {chosen}", flush=True)

    esc = calibrated(CANDIDATES["LR"]).fit(X[F.ESC], Y["CAD"])
    esc_folds = [calibrated(CANDIDATES["LR"]).fit(X.iloc[tr][F.ESC], Y["CAD"][tr]) for tr, _ in split]
    bundle = {
        "targets": models,
        "esc_baseline": {"family": "LR", "features": list(F.ESC), "model": esc, "fold_models": esc_folds},
        "columns": cols,
        "background": X.to_numpy(),  # columns = "columns"; for the linear SHAP baseline
        "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__},
        "dataset_sha256": F.DATA_SHA256,
    }
    for t, m in bundle["targets"].items():
        F.assert_no_labels(m["features"])
    F.assert_no_labels(bundle["esc_baseline"]["features"])
    joblib.dump(bundle, ART / "models.joblib", compress=3)

    schema = F.build_schema(df)
    (ART / "feature_schema.json").write_text(json.dumps(schema, indent=1, ensure_ascii=False))

    src = ROOT / "ml" / "metrics.json"
    metrics = json.loads(src.read_text())
    evidence = {}
    for v in F.VESSELS:
        aucs = {mn: metrics[mn]["vessels"][v]["within_CAD"]["dedicated_model_scores"]["auc"] for mn in ("LR", "RF")}
        mean = round((aucs["LR"] + aucs["RF"]) / 2, 3)
        evidence[v] = {"within_cad_auc": mean, "by_model": aucs, "badge": evidence_badge(mean)}
    metrics["derived"] = {
        "source": "ml/metrics.json from ml/reproduce.py, copied unchanged above this key",
        "evidence": evidence,
        "evidence_rule": "mean of LR and RF within-CAD AUC of dedicated model scores; "
                         + ", ".join(f"{k} {v}" for k, v in EVIDENCE_THRESHOLDS.items()),
        "served_model_selection": selection,
        "classification_metrics": json.loads((ROOT / "ml" / "classification_metrics.json").read_text()),
        "served_versions": bundle["versions"],
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART / "anatomy_map.json").write_text(json.dumps(anatomy_map(), indent=1))
    presets = pick_presets(df, Y)
    (ART / "presets.json").write_text(json.dumps(
        {"note": "Real dataset rows. These patients were in the training data, so predictions for them are optimistic.",
         "presets": presets}, indent=1, ensure_ascii=False))
    print("artifacts written to", ART)


if __name__ == "__main__":
    main()
