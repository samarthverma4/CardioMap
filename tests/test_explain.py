"""The served API computes SHAP in numpy (backend/app/explain.py). It must equal the shap library exactly."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "app"))
import explain  # noqa: E402

shap = pytest.importorskip("shap")
BUNDLE = joblib.load(ROOT / "artifacts" / "models.joblib")
IDX = {c: i for i, c in enumerate(BUNDLE["columns"])}


@pytest.mark.parametrize("target", ["CAD", "LAD", "LCX", "RCA"])
def test_numpy_shap_equals_shap_library(target):
    m = BUNDLE["targets"][target]
    X = BUNDLE["background"][:, [IDX[f] for f in m["features"]]]
    rows = np.random.default_rng(7).choice(len(X), 25, replace=False)
    base = m["shap_base"]
    if m["family"] == "LR":
        sc, lr = base.steps[0][1], base.steps[-1][1]
        ref = shap.LinearExplainer(lr, shap.maskers.Independent(sc.transform(X), max_samples=len(X)))
        mine = explain.linear_shap(sc, lr, X)
        expected = np.asarray(ref.shap_values(sc.transform(X[rows])))
    else:
        mine = explain.forest_shap(base)
        sv = np.asarray(shap.TreeExplainer(base).shap_values(X[rows]))
        expected = sv[:, :, 1] if sv.ndim == 3 else sv
    got = np.array([mine(X[i]) for i in rows])
    np.testing.assert_allclose(got, expected, atol=1e-9)
