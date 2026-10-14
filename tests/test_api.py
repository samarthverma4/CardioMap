"""API contract tests (Milestone 2)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import main

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(main.app)
PRESETS = json.loads((ROOT / "artifacts" / "presets.json").read_text())["presets"]


def test_prob_to_color_endpoints():
    assert main.prob_to_color(0) == "#2e9e5b"
    assert main.prob_to_color(0.5) == "#d9822b"
    assert main.prob_to_color(1) == "#c8382c"


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_predict_contract(preset):
    r = client.post("/api/predict", json={"features": preset["features"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["population_note"] == "Among patients referred for angiography"
    assert 0 <= d["cad"]["p"] <= 1 and 0 <= d["cad"]["esc_baseline_p"] <= 1
    assert len(d["cad"]["shap"]) == 8
    assert d["defaults_used"] == []
    for v in ["LAD", "LCX", "RCA"]:
        ves = d["vessels"][v]
        assert ves["band"][0] <= ves["band"][1]
        assert ves["cad_p"] == d["cad"]["p"]
        assert ves["evidence"] in {"weak", "limited", "moderate"}
        assert d["scene_state"][v]["color"] == main.prob_to_color(ves["p"])
        assert d["scene_state"][v]["evidence"] == ves["evidence"]
        assert len(ves["shap"]) == 8


def test_bbb_contribution_is_aggregated():
    d = client.post("/api/predict", json={"features": {"BBB": "LBBB"}}).json()
    names = [s["feature"] for v in d["vessels"].values() for s in v["shap"]] + [s["feature"] for s in d["cad"]["shap"]]
    assert not any(n.startswith("BBB_") for n in names)


@pytest.mark.parametrize("bad", [{"Age": 200}, {"Sex": 2}, {"BBB": "XYZ"}, {"Age": "old"}, {"Weight": 200, "Length": 120}])
def test_invalid_values_400(bad):
    assert client.post("/api/predict", json={"features": bad}).status_code == 400


def test_bmi_derived_from_weight_and_height():
    d = client.post("/api/predict", json={"features": {"Weight": 80, "Length": 170}}).json()
    assert abs(d["inputs"]["BMI"] - 80 / 1.7 ** 2) < 1e-9


@pytest.mark.parametrize("path", ["/api/schema", "/api/metrics", "/api/presets", "/api/anatomy", "/api/meta"])
def test_get_endpoints(path):
    assert client.get(path).status_code == 200
