# CardioMap: an honest map of coronary risk

Track A entry, Multimodal AI Hackathon 2026, IIT Mandi.

> **Decision support / education only.** Not a diagnosis and not a substitute for angiography or CT angiography. The 3D heart is a generic model, not this patient's anatomy. Every probability is **among patients referred for angiography**, the population of the dataset.

CardioMap takes bedside data (history, symptoms, exam, ECG, labs, echo), estimates the probability of coronary artery disease (CAD) and paints a per-vessel score onto a generic 3D heart. The vessel scores sit beside the overall CAD probability, each with an evidence badge, because the main finding of the analysis is that **vessel predictions from bedside data mostly re-encode overall CAD risk**. The app is built so that nobody reads a red LAD as "the lesion is in the LAD".

![Patient view](docs/figures/02_sample_c_lad_selected.png)

## Run it

### Docker (one command)

```bash
docker build -t cardiomap . && docker run --rm -p 8000:8000 cardiomap
```

Open http://localhost:8000. The image builds the frontend, retrains the served models (about 10 s) and runs the tests during the build.

### Local

Requirements: Python 3.11+ (tested on 3.12), Node 20+.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python ml/train.py                 # writes artifacts/ (models.joblib is not in git)
(cd frontend && npm ci && npm run build)
.venv/bin/uvicorn backend.app.main:app --port 8000
```

Frontend development with hot reload: run the backend as above, then `cd frontend && npm run dev` (proxies `/api` to port 8000).

### Tests

```bash
.venv/bin/python -m pytest -q tests
```

- `tests/test_leakage.py`: no label column (`LAD`, `LCX`, `RCA`, `Cath`) in any model's features, including fold and baseline models; encoding identical to `ml/reproduce.py`; the API returns 400 for any label field.
- `tests/test_api.py`: response contract, colour function, input validation, BMI derivation.
- `tests/test_scene_color.py` (needs `pip install -r requirements-dev.txt`): opens the built app in headless Chromium (Playwright), loads each sample patient and checks that the colour rendered on each artery mesh equals the API's `scene_state` colour. Uses Playwright's Chromium if installed, otherwise a local Chrome, Brave or Chromium (or `$CARDIOMAP_BROWSER`).

Write-up: `docs/writeup.md` (source) and `docs/writeup.pdf` (6 pages, rebuilt with `python docs/build_writeup.py`). Screenshots: `python docs/capture_screenshots.py` with the backend running.

The full evaluation is `ml/reproduce.py` (about 20 min on one core). Its outputs are `ml/metrics.json` and `ml/oof_predictions.csv`; see `ml/README_reproduce.md`.

## Architecture

```
data/*.xlsx ──► ml/features.py ──► ml/train.py ──► artifacts/
                (encoding, schema,   (model choice,    models.joblib, feature_schema.json,
                 validation)          calibration)     metrics.json, presets.json, anatomy_map.json
                                                            │
ml/reproduce.py ──► ml/metrics.json ────────────────────────┘ (copied, plus derived evidence badges)

backend/app/main.py (FastAPI)          frontend/ (Vite, React, TypeScript, react-three-fiber)
  POST /api/predict  ◄──────────────── patient form generated from /api/schema
  GET  /api/schema /metrics /presets   3D heart: colours only from scene_state
       /anatomy /meta                  results panel, model card (numbers from /api/metrics)
  serves frontend/dist on /
```

- **Features** (`ml/features.py`): the encoding is copied from `reproduce.py` (Sex male = 1; VHD none/mild/moderate/severe = 0–3; BBB one-hot, drop first; Y/N = 1/0; `Exertional CP` dropped as constant). The 54 clinical inputs, their labels, units, groups, observed ranges and defaults (median or mode) live in `artifacts/feature_schema.json`, the single source of input definitions. Values outside the training range are rejected. BMI is derived from weight and height.
- **Served models** (`ml/train.py`): for each target (CAD, LAD, LCX, RCA), logistic regression vs random forest (the two prespecified families of `reproduce.py`) is chosen by a separate 5-fold CV on the training data, never by reported results. Chosen: CAD LR, LAD RF, LCX RF, RCA LR. The final model is `CalibratedClassifierCV(method="sigmoid", cv=5)` fit on all 303 rows. Five calibrated fold models give the "fold-model range" shown as the band, which is model variability, not a confidence interval for the patient. A calibrated logistic regression on the 12 ESC features is the guideline baseline shown next to the CAD probability.
- **Vessels are independent models.** The hierarchical P(CAD) × P(vessel | CAD) model gave no gain in the analysis, so it is not served.
- **Explanations**: SHAP values come from `shap.LinearExplainer` (LR, log-odds, all 303 rows as background) or `shap.TreeExplainer` (RF, probability scale) applied to the **uncalibrated** base model refit on all rows, not to the calibrated wrapper. The top 8 contributions are shown in clinical units; the BBB dummies are summed back into one input.
- **Colour**: one function, `prob_to_color(p)` in the backend (green to amber to red). The frontend only uses the `scene_state` colours it receives.
- **Evidence badge** per vessel: mean of LR and RF within-CAD AUC of the dedicated model scores; below 0.60 weak, 0.60–0.70 limited, above 0.70 moderate. In the 3D view, moderate is solid, limited is 70% opacity and weak is 45% opacity with a dashed outline.
- **3D heart** (`frontend/public/heart.glb`, 1.1 MB, built by `tools/build_heart.py` from BodyParts3D): BodyParts3D models the cavities, atrial walls and the full coronary tree but not the ventricular wall, so the epicardium is reconstructed as an offset shell around the cavities (at the 25th percentile of coronary distance, so the epicardial arteries lie on it). Meshes: `LAD` (with diagonals, conus and septal branches), `LCX`, `RCA` (with conus, marginal, posterior ventricular and posterior interventricular branches; right-dominant), `LM`, and the surface split into `myo-LAD`, `myo-LCX`, `myo-RCA` (nearest artery, ventricles only) and `myo-base`. To rebuild: download `partof_BP3D_4.0_obj_99.zip` and `partof_element_parts.txt`, unzip into one folder, `pip install -r requirements-dev.txt`, then `python tools/build_heart.py <folder>`.
- **Sample patients** are the first dataset rows matching four fixed case definitions (no CAD; LAD only; three-vessel; LCX/RCA without LAD), excluding row 93. Labels stay hidden until "Reveal". These rows were in the training data, so their predictions are optimistic, and the UI says so.

## Results

From `ml/metrics.json` (`reproduce.py`: repeated stratified 5×10 CV, seed 2026, 2,000 patient-level bootstrap resamples; 303 patients, 216 with CAD). AUC with 95% CI.

**Vessel scores vs CAD probability**

| Vessel | Model | Dedicated model | CAD probability as score | Dedicated minus CAD prob. | Within CAD (dedicated) | Evidence |
|---|---|---|---|---|---|---|
| LAD | LR | 0.820 (0.772–0.866) | 0.844 (0.798–0.885) | -0.024 (-0.047 to -0.000) | 0.606 (0.524–0.689) | limited (0.638) |
| LAD | RF | 0.855 (0.810–0.895) | 0.849 (0.802–0.889) | +0.006 (-0.014 to 0.027) | 0.670 (0.585–0.753) | |
| LCX | LR | 0.674 (0.613–0.730) | 0.711 (0.652–0.767) | -0.037 (-0.087 to 0.010) | 0.518 (0.438–0.599) | weak (0.570) |
| LCX | RF | 0.748 (0.693–0.801) | 0.722 (0.661–0.780) | +0.026 (-0.019 to 0.070) | 0.622 (0.543–0.701) | |
| RCA | LR | 0.731 (0.672–0.784) | 0.739 (0.680–0.791) | -0.008 (-0.055 to 0.039) | 0.605 (0.528–0.678) | weak (0.597) |
| RCA | RF | 0.719 (0.659–0.776) | 0.714 (0.654–0.768) | +0.004 (-0.044 to 0.055) | 0.589 (0.510–0.661) | |

No dedicated vessel model beats the CAD probability used as the vessel score. In one case (LAD, LR) the CI only just excludes zero, in the direction of the dedicated model being *worse*. Within CAD patients, vessel AUCs are 0.52–0.67.

**What each modality adds (CAD)**

| Feature set | LR | RF |
|---|---|---|
| ESC inputs (12) | 0.910 (0.868–0.947) | 0.901 (0.858–0.939) |
| + ECG | 0.927 (0.892–0.957) | 0.914 (0.876–0.947) |
| + ECG + echo | 0.939 (0.909–0.965) | 0.933 (0.900–0.962) |
| All 55 columns | 0.927 (0.895–0.956) | 0.925 (0.890–0.955) |

ECG + echo over the ESC inputs: LR +0.029 (0.013 to 0.048), RF +0.031 (0.011 to 0.053).

![Model card](docs/figures/07_vessel_auc_chart.png)

## Limitations

- One centre, 303 patients, all already referred for angiography. Probabilities do not transfer to screening or emergency populations.
- No external validation. Bootstrap CIs ignore refit variance and are somewhat narrow.
- Vessel scores mostly re-encode overall CAD risk and must not be read as lesion location. Vessel labels are stenotic/normal only.
- Row 93 (0-based) has a stenotic vessel but is labelled no CAD; it is kept as published.
- The served models are refit on all rows; the reported metrics are cross-validated estimates of the same model families, not of the exact served objects.
- The heart is one anatomical model (BodyParts3D), not this patient's anatomy; territories assume right-dominant circulation.

## Licences and data

- Dataset: Alizadehsani R., Roshanzamir M., Sani Z. *Extension of Z-Alizadeh Sani dataset*. UCI Machine Learning Repository, DOI 10.24432/C5461K. CC BY 4.0. SHA-256 of the file used: `739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b` (checked at load).
- 3D heart: **BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International** (https://dbarchive.biosciencedbc.jp/en/bodyparts3d/). The coronary arteries are the original meshes; the epicardial surface was reconstructed from the cavity and atrial-wall meshes and all meshes were decimated (`tools/build_heart.py`). The credit is shown on the 3D view.
- Fonts bundled from npm, no runtime CDN: Bricolage Grotesque, Public Sans, IBM Plex Mono (SIL Open Font License 1.1).
- Libraries: scikit-learn, pandas, NumPy, SHAP, FastAPI, React, three.js, react-three-fiber, drei, Vite (each under its own open-source licence).

## AI assistance

Tools: **Claude Code** (Anthropic, Claude Opus 5.5) and **Claude** (Anthropic).

- `ml/reproduce.py`, `ml/metrics.json`, `ml/oof_predictions.csv` and the analysis design were produced with Claude.
- With Claude Code: `ml/features.py`, `ml/train.py`, the artifacts, the FastAPI backend, the React/three.js frontend (including the charts and the BodyParts3D heart pipeline), all tests, the Dockerfile, the screenshot script and this README.
- The team chose the research question, reviewed the outputs and is responsible for every claim in the submission.

## Repository

```
data/          dataset (hash-checked)
ml/            features.py, train.py, reproduce.py (+ its metrics and OOF predictions)
artifacts/     schema, metrics, presets, anatomy map (models.joblib is rebuilt by train.py)
backend/app/   FastAPI app
frontend/      Vite + React + TypeScript + react-three-fiber (public/heart.glb)
tools/         build_heart.py: BodyParts3D -> heart.glb
tests/         leakage, API and rendered-colour tests
docs/          6-page write-up (md + pdf), screenshot script, figures
```
