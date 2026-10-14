"""Track A — reproducible experiments on the UCI 'Extension of Z-Alizadeh Sani' dataset.

Run:  python reproduce.py  (expects the UCI xlsx in the same folder)
Out:  metrics.json, oof_predictions.csv, results.md

Design (fixed before looking at results of this run):
- One set of CV splits for every target and model: RepeatedStratifiedKFold(5 folds x 10 repeats,
  random_state=2026), stratified on the number of stenotic vessels (0-3), so all comparisons are paired.
- Two prespecified model families, reported separately (no best-of selection):
  LR = StandardScaler + L2 logistic regression (C=0.1); RF = random forest (200 trees, min_samples_leaf=3).
- Each patient's out-of-fold (OOF) probability is averaged over the 10 repeats; AUCs and 95% CIs come from
  2,000 patient-level bootstrap resamples of those averaged OOF scores. Differences are paired bootstraps.
- Label columns LAD, LCX, RCA, Cath are never inputs (asserted). 'Cath' is the CAD target.
"""
import hashlib, json, sys
import numpy as np, pandas as pd, sklearn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.base import clone

FILE = "extention of Z-Alizadeh sani dataset.xlsx"
SEED, N_REP, N_BOOT = 2026, 10, 2000
sha = hashlib.sha256(open(FILE, "rb").read()).hexdigest()

df = pd.read_excel(FILE)
LABELS = ["LAD", "LCX", "RCA", "Cath"]
Y = {"CAD": df.Cath.eq("CAD").astype(int).values}
for v in ["LAD", "LCX", "RCA"]:
    Y[v] = df[v].eq("Stenotic").astype(int).values
nves = sum(Y[v] for v in ["LAD", "LCX", "RCA"])

X = df.drop(columns=LABELS + ["Exertional CP"]).copy()  # Exertional CP is constant ('N')
X["Sex"] = X.Sex.eq("Male").astype(int)
X["VHD"] = X.VHD.map({"N": 0, "mild": 1, "Moderate": 2, "Severe": 3})
X = pd.get_dummies(X, columns=["BBB"], drop_first=True)
for c in X.columns:
    if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
        X[c] = X[c].map({"Y": 1, "N": 0})
X = X.astype(float)
assert not set(LABELS) & set(X.columns), "label leakage"
assert X.notna().all().all()

ESC = ["Age", "Sex", "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
       "FH", "Current Smoker", "EX-Smoker", "DLP", "HTN", "DM"]
ECG = ["Tinversion", "St Depression", "St Elevation", "Q Wave", "LVH", "Poor R Progression"]
ECHO = ["EF-TTE", "Region RWMA", "VHD"]
SETS = {"esc_features": ESC, "esc+ecg": ESC + ECG, "esc+ecg+echo": ESC + ECG + ECHO, "all": list(X.columns)}
MODELS = {"LR": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
          "RF": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=-1)}

splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED).split(X, nves))
n = len(X)

def oof(model, cols, y, train_mask=None):
    """Repeat-averaged OOF probabilities. train_mask restricts training rows (e.g. CAD-positive only);
    predictions are still made for every test row."""
    P = np.zeros((N_REP, n))
    for k, (tr, te) in enumerate(splits):
        tr2 = tr if train_mask is None else tr[train_mask[tr]]
        m = clone(model).fit(X.iloc[tr2][cols], y[tr2])
        P[k // 5, te] = m.predict_proba(X.iloc[te][cols])[:, 1]
    return P

rng = np.random.default_rng(SEED)
BOOT = [rng.integers(0, n, n) for _ in range(N_BOOT)]

def auc_ci(y, s, mask=None):
    idx = np.arange(n) if mask is None else np.where(mask)[0]
    point = roc_auc_score(y[idx], s[idx])
    vals = []
    for b in BOOT:
        bi = b if mask is None else rng.choice(idx, len(idx))
        if len(set(y[bi])) == 2:
            vals.append(roc_auc_score(y[bi], s[bi]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auc": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)]}

def diff_ci(y, s1, s2, mask=None):
    idx = np.arange(n) if mask is None else np.where(mask)[0]
    point = roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx])
    vals = []
    for b in BOOT:
        bi = b if mask is None else rng.choice(idx, len(idx))
        if len(set(y[bi])) == 2:
            vals.append(roc_auc_score(y[bi], s1[bi]) - roc_auc_score(y[bi], s2[bi]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"diff": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)], "ci_excludes_0": bool(lo > 0 or hi < 0)}

def rep_spread(P, y, mask=None):
    idx = np.arange(n) if mask is None else np.where(mask)[0]
    a = [roc_auc_score(y[idx], P[r, idx]) for r in range(N_REP)]
    return {"mean_of_repeats": round(float(np.mean(a)), 3), "sd_of_repeats": round(float(np.std(a)), 3)}

R = {"dataset_sha256": sha, "n": int(n), "python": sys.version.split()[0], "sklearn": sklearn.__version__,
     "design": {"cv": "RepeatedStratifiedKFold 5x10, stratified on #stenotic vessels", "seed": SEED, "bootstrap": N_BOOT},
     "counts": {"CAD": int(Y["CAD"].sum()), "LAD": int(Y["LAD"].sum()), "LCX": int(Y["LCX"].sum()), "RCA": int(Y["RCA"].sum()),
                "n_vessels": {int(k): int(v) for k, v in pd.Series(nves).value_counts().sort_index().items()},
                "rows_vessel_stenotic_but_cath_normal": [int(i) for i in np.where((nves > 0) & (Y["CAD"] == 0))[0]]}}
cad = Y["CAD"] == 1
OUT = {}
for mn, model in MODELS.items():
    r = {}
    # per-target models, all features, plus ablation
    S = {}
    for setname, cols in SETS.items():
        for t in ["CAD", "LAD", "LCX", "RCA"]:
            P = oof(model, cols, Y[t]); s = P.mean(0); S[(setname, t)] = s
            OUT[f"{mn}_{setname}_{t}"] = s
            r.setdefault("ablation", {}).setdefault(setname, {})[t] = {**auc_ci(Y[t], s), **rep_spread(P, Y[t])}
    # ablation paired differences (each step vs previous; and all vs esc)
    steps = [("esc+ecg", "esc_features"), ("esc+ecg+echo", "esc+ecg"), ("esc+ecg+echo", "esc_features"), ("all", "esc_features")]
    for t in ["CAD", "LAD", "LCX", "RCA"]:
        for a, b in steps:
            r.setdefault("ablation_diffs", {}).setdefault(t, {})[f"{a} minus {b}"] = diff_ci(Y[t], S[(a, t)], S[(b, t)])
    cadp = S[("all", "CAD")]
    for v in ["LAD", "LCX", "RCA"]:
        ded = S[("all", v)]
        # conditional model trained on CAD-positive training rows only
        Pc = oof(model, SETS["all"], Y[v], train_mask=cad); cond = Pc.mean(0); OUT[f"{mn}_conditional_{v}"] = cond
        hier = cadp * cond; OUT[f"{mn}_hierarchical_{v}"] = hier
        r.setdefault("vessels", {})[v] = {
            "dedicated_all_patients": auc_ci(Y[v], ded),
            "cad_probability_as_vessel_score": auc_ci(Y[v], cadp),
            "dedicated_minus_cadprob": diff_ci(Y[v], ded, cadp),
            "hierarchical_P(CAD)xP(V|CAD)": auc_ci(Y[v], hier),
            "hierarchical_minus_dedicated": diff_ci(Y[v], hier, ded),
            "within_CAD": {
                "dedicated_model_scores": auc_ci(Y[v], ded, cad),
                "conditional_model_trained_on_CAD_only": {**auc_ci(Y[v], cond, cad), **rep_spread(Pc, Y[v], cad)},
                "cad_probability": auc_ci(Y[v], cadp, cad),
                "conditional_minus_dedicated": diff_ci(Y[v], cond, ded, cad)}}
    # calibration of the CAD model by sex
    male = X.Sex.values == 1
    for lab, msk in [("all", np.ones(n, bool)), ("male", male), ("female", ~male)]:
        p = np.clip(cadp[msk], 1e-6, 1 - 1e-6); yy = Y["CAD"][msk]
        lr = LogisticRegression(C=1e6, max_iter=5000).fit(np.log(p / (1 - p)).reshape(-1, 1), yy)
        r.setdefault("cad_calibration_by_sex", {})[lab] = {
            "n": int(msk.sum()), "prevalence": round(float(yy.mean()), 3), "mean_predicted": round(float(p.mean()), 3),
            "brier": round(float(brier_score_loss(yy, p)), 3), "calibration_slope": round(float(lr.coef_[0][0]), 2),
            "auc": round(float(roc_auc_score(yy, p)), 3)}
    R[mn] = r
    print("done", mn, flush=True)

json.dump(R, open("metrics.json", "w"), indent=2)
pd.DataFrame({"row": np.arange(n), "n_stenotic_vessels": nves, **{f"y_{k}": v for k, v in Y.items()},
              **{k: np.round(v, 5) for k, v in OUT.items()}}).to_csv("oof_predictions.csv", index=False)
print("sha256", sha)
