"""CardioMap API. Serves /api/* from artifacts/ and the built frontend from frontend/dist."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain  # noqa: E402
import features as F  # noqa: E402

# Models were fitted on named DataFrames; the API passes plain arrays in the same column order.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ART = ROOT / "artifacts"
DIST = ROOT / "frontend" / "dist"
POPULATION_NOTE = "Among patients referred for angiography"
DISCLAIMER = ("Decision support / education only. Not a diagnosis and not a substitute for angiography or "
              "CT angiography. The 3D heart is a generic model, not this patient's anatomy.")
TOP_K = 8

BUNDLE = joblib.load(ART / "models.joblib")
SCHEMA = json.loads((ART / "feature_schema.json").read_text())
METRICS = json.loads((ART / "metrics.json").read_text())
PRESETS = json.loads((ART / "presets.json").read_text())
ANATOMY = json.loads((ART / "anatomy_map.json").read_text())
SPEC = {i["name"]: i for i in SCHEMA["inputs"]}
COLUMNS = BUNDLE["columns"]
COL_IDX = {c: i for i, c in enumerate(COLUMNS)}
EVIDENCE = METRICS["derived"]["evidence"]

for _t, _m in BUNDLE["targets"].items():
    F.assert_no_labels(_m["features"])
F.assert_no_labels(BUNDLE["esc_baseline"]["features"])

# --- colour: the ONLY place probabilities become colours -------------------------------------------
_STOPS = [(0.0, (0x2e, 0x9e, 0x5b)), (0.5, (0xd9, 0x82, 0x2b)), (1.0, (0xc8, 0x38, 0x2c))]


def prob_to_color(p: float) -> str:
    """Green (0) -> amber (0.5) -> red (1), linear in RGB."""
    p = float(np.clip(p, 0.0, 1.0))
    for (p0, c0), (p1, c1) in zip(_STOPS, _STOPS[1:]):
        if p <= p1:
            f = (p - p0) / (p1 - p0)
            return "#" + "".join(f"{round(a + (b - a) * f):02x}" for a, b in zip(c0, c1))
    return "#%02x%02x%02x" % _STOPS[-1][1]


# --- SHAP on the uncalibrated base model (exact, numpy; see explain.py) ------------------------------
def _cols(X: np.ndarray, features: list[str]) -> np.ndarray:
    return X[:, [COL_IDX[f] for f in features]]


def _make_explainer(m: dict):
    base = m["shap_base"]
    if m["family"] == "LR":
        scaler, lr = base.steps[0][1], base.steps[-1][1]
        return explain.linear_shap(scaler, lr, _cols(BUNDLE["background"], m["features"])), "log-odds"
    return explain.forest_shap(base), "probability"


EXPLAINERS = {t: _make_explainer(m) for t, m in BUNDLE["targets"].items()}


def _fmt_value(name: str, v: Any) -> str:
    it = SPEC[name]
    if "options" in it:
        for o in it["options"]:
            if o["value"] == v:
                return o["label"]
    s = f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
    return f"{s} {it['unit']}".strip()


def _shap(target: str, X, inputs: dict) -> tuple[list[dict], str]:
    run, scale = EXPLAINERS[target]
    sv = run(_cols(X, BUNDLE["targets"][target]["features"])[0])
    contrib: dict[str, float] = {}
    for col, val in zip(BUNDLE["targets"][target]["features"], sv):
        name = "BBB" if col.startswith("BBB_") else col
        contrib[name] = contrib.get(name, 0.0) + float(val)
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:TOP_K]
    return [{"feature": n, "label": SPEC[n]["label"], "value": inputs[n], "display_value": _fmt_value(n, inputs[n]),
             "unit": SPEC[n]["unit"], "contribution": round(c, 4)} for n, c in top], scale


def _sentence(target: str, shap_items: list[dict]) -> str:
    up = [s for s in shap_items if s["contribution"] > 0][:2]
    down = [s for s in shap_items if s["contribution"] < 0][:1]
    what = "overall CAD probability" if target == "CAD" else f"{target} score"
    parts = []
    if up:
        parts.append("raised most by " + " and ".join(f"{s['label'].lower()} = {s['display_value']}" for s in up))
    if down:
        parts.append("lowered most by " + f"{down[0]['label'].lower()} = {down[0]['display_value']}")
    text = f"The {what} is " + ("; ".join(parts) if parts else "close to the dataset average") + "."
    if target != "CAD":
        text += " These are mostly general CAD risk factors, not signs of where a lesion is."
    return text


def _predict_target(m: dict, X) -> tuple[float, list[float]]:
    Xm = _cols(X, m["features"])
    p = float(m["model"].predict_proba(Xm)[0, 1])
    folds = [float(fm.predict_proba(Xm)[0, 1]) for fm in m["fold_models"]]
    return p, [round(min(folds), 4), round(max(folds), 4)]


def predict(features: dict) -> dict:
    inputs = F.complete_inputs(features, SCHEMA)
    X = F.encode_rows([inputs], COLUMNS)
    out: dict[str, Any] = {}

    cad_m = BUNDLE["targets"]["CAD"]
    p_cad, band = _predict_target(cad_m, X)
    esc_p, esc_band = _predict_target(BUNDLE["esc_baseline"], X)
    shap_cad, scale = _shap("CAD", X, inputs)
    out["cad"] = {"p": round(p_cad, 4), "band": band, "esc_baseline_p": round(esc_p, 4), "esc_baseline_band": esc_band,
                  "model": cad_m["family"], "shap": shap_cad, "shap_scale": scale, "explanation": _sentence("CAD", shap_cad)}
    out["vessels"], out["scene_state"] = {}, {}
    for v in F.VESSELS:
        m = BUNDLE["targets"][v]
        p, band = _predict_target(m, X)
        sh, scale = _shap(v, X, inputs)
        ev = EVIDENCE[v]
        out["vessels"][v] = {"p": round(p, 4), "band": band, "cad_p": round(p_cad, 4), "evidence": ev["badge"],
                             "within_cad_auc": ev["within_cad_auc"], "model": m["family"], "shap": sh,
                             "shap_scale": scale, "explanation": _sentence(v, sh)}
        out["scene_state"][v] = {"color": prob_to_color(p), "evidence": ev["badge"]}
    out["population_note"] = POPULATION_NOTE
    derived_bmi = "BMI" not in features and "Weight" in features and "Length" in features
    out["defaults_used"] = sorted(k for k in inputs if k not in features and not (k == "BMI" and derived_bmi))
    out["inputs"] = inputs
    return out


# --- HTTP ------------------------------------------------------------------------------------------
app = FastAPI(title="CardioMap", description=DISCLAIMER)


class PredictRequest(BaseModel):
    features: dict[str, Any] = {}


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    try:
        return predict(req.features)
    except F.InputError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/schema")
def api_schema():
    return SCHEMA


@app.get("/api/metrics")
def api_metrics():
    return METRICS


@app.get("/api/presets")
def api_presets():
    return PRESETS


@app.get("/api/anatomy")
def api_anatomy():
    return ANATOMY


@app.get("/api/meta")
def api_meta():
    return {"disclaimer": DISCLAIMER, "population_note": POPULATION_NOTE,
            "color_scale": [{"p": p, "color": prob_to_color(p)} for p in (0, 0.25, 0.5, 0.75, 1)],
            "versions": BUNDLE["versions"], "dataset_sha256": BUNDLE["dataset_sha256"]}


# Hashed bundles (/assets/*) never change, so they may be cached forever; everything else (index.html,
# heart.glb) must be revalidated, or a browser keeps showing the previous build after a rebuild.
NO_CACHE = {"Cache-Control": "no-cache"}


class ImmutableStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return r


if DIST.exists():
    app.mount("/assets", ImmutableStatic(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f, headers=NO_CACHE)
        return FileResponse(DIST / "index.html", headers=NO_CACHE)
