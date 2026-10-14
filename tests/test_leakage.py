"""No label column (LAD, LCX, RCA, Cath) may reach any model, and the API must reject label fields."""
import json
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT))
import features as F  # noqa: E402

LABELS = {"LAD", "LCX", "RCA", "Cath"}
BUNDLE = joblib.load(ROOT / "artifacts" / "models.joblib")


def _all_models():
    for t, m in BUNDLE["targets"].items():
        yield f"{t}/final", m["features"], m["model"]
        yield f"{t}/shap_base", m["features"], m["shap_base"]
        for i, fm in enumerate(m["fold_models"]):
            yield f"{t}/fold{i}", m["features"], fm
    esc = BUNDLE["esc_baseline"]
    yield "esc/final", esc["features"], esc["model"]
    for i, fm in enumerate(esc["fold_models"]):
        yield f"esc/fold{i}", esc["features"], fm


@pytest.mark.parametrize("name,feats,model", list(_all_models()), ids=lambda x: x if isinstance(x, str) else "")
def test_no_label_in_model_features(name, feats, model):
    assert not LABELS & set(feats)
    seen = getattr(model, "feature_names_in_", None)
    assert seen is not None, f"{name} was not fit on a named frame"
    assert not LABELS & set(seen)
    assert list(seen) == list(feats)


def test_schema_has_no_label_inputs():
    schema = json.loads((ROOT / "artifacts" / "feature_schema.json").read_text())
    assert not LABELS & {i["name"] for i in schema["inputs"]}


def test_encoding_matches_reproduce():
    """Same encoding as ml/reproduce.py (snippet copied verbatim)."""
    import pandas as pd
    df = F.load_raw()
    X = df.drop(columns=["LAD", "LCX", "RCA", "Cath"] + ["Exertional CP"]).copy()
    X["Sex"] = X.Sex.eq("Male").astype(int)
    X["VHD"] = X.VHD.map({"N": 0, "mild": 1, "Moderate": 2, "Severe": 3})
    X = pd.get_dummies(X, columns=["BBB"], drop_first=True)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
            X[c] = X[c].map({"Y": 1, "N": 0})
    X = X.astype(float)
    pd.testing.assert_frame_equal(F.encode(df), X)
    rows = F.clinical_frame(df).to_dict("records")
    pd.testing.assert_frame_equal(F.encode_rows(rows, list(X.columns)), X)


@pytest.mark.parametrize("label", sorted(LABELS))
def test_validation_rejects_label(label):
    schema = json.loads((ROOT / "artifacts" / "feature_schema.json").read_text())
    with pytest.raises(F.InputError):
        F.complete_inputs({label: "Stenotic"}, schema)


@pytest.fixture(scope="module")
def client():
    main = pytest.importorskip("backend.app.main", reason="API not built yet (Milestone 2)")
    from fastapi.testclient import TestClient
    return TestClient(main.app)


@pytest.mark.parametrize("label", sorted(LABELS))
@pytest.mark.parametrize("value", ["Stenotic", "CAD", 1, 0])
def test_api_rejects_label_fields(client, label, value):
    r = client.post("/api/predict", json={"features": {"Age": 60, label: value}})
    assert r.status_code == 400
    assert label in r.text


def test_api_rejects_unknown_field(client):
    r = client.post("/api/predict", json={"features": {"Cholesterol": 200}})
    assert r.status_code == 400


def test_api_accepts_empty_payload(client):
    r = client.post("/api/predict", json={"features": {}})
    assert r.status_code == 200
