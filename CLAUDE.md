# CardioMap: Track A "Honest Map" (Multimodal AI Hackathon 2026, IIT Mandi)

## Hard constraints
- **Deadline: Devpost cutoff 15 Oct 2026, 12:15 AM IST.** Build in milestone order; if time runs short, cut from the bottom.
- Web app: interactive 3D heart + clinical dashboard + ML backend. Must run on a laptop without a dedicated GPU.
- Never use `LAD`, `LCX`, `RCA` or `Cath` as model inputs. `Cath` is the CAD target. Assert this in code AND reject those fields in the API.
- A visible disclaimer banner on every screen: "Decision support / education only. Not a diagnosis and not a substitute for angiography or CT angiography. The 3D heart is a generic model, not this patient's anatomy."
- Every percentage shown is labelled "among patients referred for angiography" (the dataset's population).
- AI disclosure is mandatory: README section "AI assistance" listing tools (Claude Code, Claude) and which parts were AI-assisted.

## Core idea (what makes this entry different)
Vessel predictions from bedside data mostly re-encode overall CAD risk. Our reproducible analysis (`ml/reproduce.py`, `ml/metrics.json`) shows:
- CAD probability used as the vessel score ≈ dedicated vessel models (AUC LAD ~0.85, LCX ~0.72, RCA ~0.72; differences not significant).
- Within CAD patients, vessel AUC is only 0.52–0.67.
- The hierarchical P(CAD)×P(vessel|CAD) model gave NO gain. Use independent per-vessel models.
- ESC-feature logistic baseline (12 inputs) CAD AUC 0.910; + ECG + echo → 0.939 (+0.029, CI 0.013–0.048).
So the UI must show each vessel's probability **beside** the overall CAD probability, with an **evidence badge** per vessel, and must present the ESC-feature baseline and the modality ablation. Never imply the model knows where a lesion is.

## Stack
- `ml/`: Python 3.11+, pandas, scikit-learn, shap, joblib.
- `backend/`: FastAPI + uvicorn, serves `/api/*` and the built frontend.
- `frontend/`: Vite + React + TypeScript + @react-three/fiber + @react-three/drei. No external CDN at runtime.
- One `Dockerfile` that builds frontend and runs backend on port 8000.

## Repo layout
```
data/extention of Z-Alizadeh sani dataset.xlsx   # SHA-256 739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b
ml/features.py      # load + encode (copy logic from reproduce.py), feature groups, schema export
ml/reproduce.py     # evaluation script (already written; do not change its design)
ml/train.py         # fits final served models -> artifacts/
artifacts/          # models.joblib, feature_schema.json, metrics.json, anatomy_map.json, presets.json
backend/app/main.py
frontend/src/...
tests/
docs/               # 6-page writeup source, figures
```

## Milestone 1: ML artefacts (target ~1 h)
1. `ml/features.py`: encoding exactly as in `reproduce.py` (Sex Male=1; VHD N/mild/Moderate/Severe → 0–3; BBB one-hot; Y/N → 1/0; drop `Exertional CP`). Export `feature_schema.json`: for each input its name, display label, unit, type (binary / ordinal / continuous), allowed range, default (dataset median or mode), and group (`demographic`, `history`, `symptoms`, `exam`, `ecg`, `lab`, `echo`). ESC-feature set = Age, Sex, Typical Chest Pain, Atypical, Nonanginal, Dyspnea, FH, Current Smoker, EX-Smoker, DLP, HTN, DM.
2. `ml/train.py`: for each target (CAD, LAD, LCX, RCA) choose LR vs RF by **inner 5-fold CV on the training data only** (no selection on reported results); fit final model on all 303 rows, wrapped in `CalibratedClassifierCV(method="sigmoid", cv=5)`. Also keep the 5 fold models from one 5-fold split for a "model variability" band (min–max of their predictions). Also fit an ESC-feature-only LR for CAD (the baseline shown in the UI).
3. Copy `metrics.json` from `reproduce.py` into `artifacts/`; derive per-vessel evidence badge from within-CAD AUC (dedicated model scores, mean of LR and RF): `< 0.60 weak`, `0.60–0.70 limited`, `> 0.70 moderate`.
4. `presets.json`: 4 real dataset rows as sample patients (no-CAD, single-vessel LAD, multi-vessel, LCX/RCA case), with labels hidden in the UI until "reveal".
5. Test: `tests/test_leakage.py` asserts no label column in any model's features and that the API returns 400 for label fields.

## Milestone 2: API (target ~30 min)
`POST /api/predict` body `{ "features": { name: value, ... } }` (missing → schema default; unknown or label field → 400). Response:
```json
{
  "cad": {"p": 0.82, "band": [0.76, 0.88], "esc_baseline_p": 0.71, "shap": [{"feature":"Typical Chest Pain","value":1,"unit":"","contribution":0.21}]},
  "vessels": {
    "LAD": {"p": 0.74, "band": [0.66, 0.80], "evidence": "limited", "within_cad_auc": 0.64, "shap": [...]},
    "LCX": {...}, "RCA": {...}
  },
  "scene_state": {"LAD": {"color": "#d9822b", "evidence": "limited"}, "LCX": {...}, "RCA": {...}},
  "population_note": "Among patients referred for angiography"
}
```
`GET /api/schema`, `GET /api/metrics`, `GET /api/presets`. SHAP: `shap.TreeExplainer` for RF, `shap.LinearExplainer` for LR (on the uncalibrated base model; say so in docs). Top 8 contributions, in original units.
Colour: ONE function `prob_to_color(p)` (green → amber → red) defined in the backend; the frontend uses `scene_state` colours only, never recomputes them.

## Milestone 3: frontend (target ~2.5 h)
Layout: left = patient form; centre = 3D canvas; right = results panel.
- **3D heart (procedural, no external mesh needed):** an ellipsoid-ish heart from merged spheres or a lathe geometry; three coronary arteries as `TubeGeometry` along `CatmullRomCurve3` paths: LAD down the anterior interventricular groove to the apex, LCX around the left AV groove to the back, RCA around the right AV groove to the inferior wall. Mesh names exactly `LAD`, `LCX`, `RCA`. OrbitControls rotate/zoom; click an artery to select it; hover tooltip. `frameloop="demand"`, low segment counts, `dpr={[1, 1.5]}`.
- Artery colour = `scene_state` colour. Evidence encoding: `moderate` solid; `limited` 70% opacity; `weak` 45% opacity + dashed outline/pulsing ring. Legend explains both channels.
- Optional territory overlay: tint the myocardium region of the selected vessel; add a small SVG AHA 17-segment bullseye (LAD 1,2,7,8,13,14,17; RCA 3,4,9,10,15; LCX 5,6,11,12,16) with label "assumes right-dominant circulation".
- **Results panel:** CAD probability (large) with band and the ESC-feature baseline next to it ("Guideline inputs alone: 71% → with your ECG + echo: 82%"); three vessel rows each showing p, band, evidence badge and P(CAD) beside it; selected vessel → SHAP bar chart in clinical units plus one plain-language sentence.
- **Model card tab:** metrics table with CIs from `metrics.json`; the vessel AUC chart (dedicated vs CAD-prob-as-score vs within-CAD); modality ablation chart; calibration note; data-quality note (row 93, constant column); limitations.
- Form is generated from `feature_schema.json` grouped by modality; "Load sample patient" uses presets.

## Milestone 4: polish + submission (target ~1 h)
- Dockerfile; `README.md` with run steps, architecture, results table, licences (UCI dataset CC BY 4.0; BodyParts3D CC BY 4.0 only if used), **AI assistance** section.
- Playwright or simple script test: rendered artery colour == API colour for a preset.
- Screenshots to `docs/figures/` for the 6-page doc and video.

## Cut list if behind (in this order)
bullseye → territory overlay → model-variability band → Playwright test → Docker (keep README run steps).

## Coding rules
- Read numbers for the UI from `artifacts/`, never hard-code them in the frontend.
- Keep feature definitions in one place (`feature_schema.json`).
- Commit after each milestone with a clear message.
