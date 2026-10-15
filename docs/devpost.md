# Devpost submission fields

## About the project (paste into "Project Story")

## Inspiration

Ask a model trained on bedside data "which coronary artery is blocked?" and it will happily paint the LAD red. The question behind CardioMap: is that red LAD real information about *where* the disease is, or just overall CAD risk wearing a different label?

The Z-Alizadeh Sani extension dataset (303 patients referred for angiography, 55 bedside variables, per-vessel stenosis labels for LAD, LCX and RCA) makes the question testable. The answer shaped the whole product.

## What it does

CardioMap takes history, symptoms, exam, ECG, labs and echo, then shows:

- **Probability of CAD** with a fold-model range, next to a **guideline baseline** built from the 12 ESC inputs alone ("Guideline inputs alone: 87% → all 54 inputs: 78%").
- **A score per vessel (LAD, LCX, RCA)**, always printed beside $P(\text{CAD})$, with an **evidence badge** (weak / limited / moderate) earned from how well that vessel model ranks patients *who already have CAD*.
- **An interactive 3D heart.** Each artery is coloured by its score; opacity and a dashed outline encode the evidence badge, so a weak signal *looks* weak. Click an artery to tint its myocardial territory, light up an AHA 17-segment bullseye and send a glowing stream along the vessel.
- **A SHAP chart in clinical units** ("Typical chest pain = Yes", "Age = 50 years") plus one plain-language sentence.
- **A model card** with every metric and its 95% CI, the vessel AUC comparison, the modality ablation and the limitations.

Every percentage is labelled *among patients referred for angiography*, and every screen carries the disclaimer: decision support / education only, not a diagnosis.

## The finding

Paired cross-validation (5×10 repeated stratified CV, 2,000 patient-level bootstrap resamples):

| Vessel (served model) | Dedicated model AUC | CAD probability used as the vessel score | Within CAD patients |
|---|---|---|---|
| LAD (RF) | 0.85 | 0.85 | 0.67 |
| LCX (RF) | 0.75 | 0.72 | 0.62 |
| RCA (LR) | 0.73 | 0.74 | 0.61 |

- No dedicated vessel model beats simply reusing $P(\text{CAD})$ as the vessel score. Every difference CI includes zero (one LR case only just excludes it, in the direction of the dedicated model being *worse*).
- Among patients who do have CAD, vessel AUCs fall to **0.52 to 0.67**: barely above chance for telling *which* artery.
- A hierarchical model $P(\text{CAD}) \times P(\text{vessel} \mid \text{CAD})$ gave **no gain**.
- What bedside data *does* do well: CAD itself. ESC inputs alone reach AUC **0.910**; adding ECG and echo reaches **0.939**, a gain of $+0.029$ (95% CI 0.013 to 0.048).

So the UI never lets a red artery stand alone. Evidence badge rule, from the mean within-CAD AUC of the LR and RF dedicated models:

$$
\text{badge} = \begin{cases} \text{weak} & \overline{\text{AUC}}_{\text{within CAD}} < 0.60 \\ \text{limited} & 0.60 \le \overline{\text{AUC}}_{\text{within CAD}} \le 0.70 \\ \text{moderate} & \overline{\text{AUC}}_{\text{within CAD}} > 0.70 \end{cases}
$$

Result: LAD limited, LCX weak, RCA weak.

Classification metrics at a threshold of 0.5 fixed in advance (out-of-fold, random forest): CAD accuracy 0.87, precision 0.87, recall 0.95, F1 0.91. Vessel recall is low (LCX 0.34, RCA 0.32) because vessel scores are rarely confident, which is exactly the point.

## How we built it

- **ML (Python, scikit-learn):** for each target, logistic regression vs random forest chosen by an inner 5-fold CV on training data only, never by reported results. Final models wrapped in `CalibratedClassifierCV(method="sigmoid")`; five fold models give the variability band. Label columns (`LAD`, `LCX`, `RCA`, `Cath`) are asserted absent from every feature set and rejected by the API with a 400.
- **Explanations:** exact SHAP on the uncalibrated base model. For logistic regression, $\phi_j = \beta_j \, (x_j - \bar{x}_j)$ in standardised space; for random forests, path-dependent TreeSHAP. Both are implemented in numpy and tested equal to the `shap` library (max error about $10^{-16}$), which kept the serverless function small.
- **Backend:** FastAPI. One function, `prob_to_color(p)`, turns probabilities into colours; the frontend never recomputes them, and a Playwright test checks that the colour rendered on each artery mesh equals the API's colour.
- **Frontend:** React, TypeScript, three.js via react-three-fiber. Custom shaders for the territory tint (smooth per-vertex weights), the dashed weak-evidence outline and the additive glowing stream. `frameloop="demand"` keeps it light on laptops without a dedicated GPU.
- **3D heart:** the textured "Realistic Human Heart" model (neshallads, CC BY 4.0) with real coronary arteries from BodyParts3D (CC BY 4.0) fitted onto it: rigid ICP, then non-rigid NRICP registration, then a seating pass so no artery segment floats above or sinks below the surface. Territory weights are balanced to the AHA 17-segment shares (LAD 7 : LCX 5 : RCA 5).
- **Reproducibility:** dataset SHA-256 checked at load, seeds fixed, 98 tests, a Dockerfile that retrains and tests at build time, and a 6-page write-up.

## Challenges we ran into

- **Honesty vs. a flashy demo.** A glowing red artery is persuasive. Designing the evidence encoding (badge, opacity, dashed outline, $P(\text{CAD})$ always beside the vessel score) so the visual could not overclaim took more iterations than any model.
- **Fitting two anatomies together.** The arteries from BodyParts3D and the textured surface came from different hearts. Early fits left vessels floating in mid-air or buried in muscle; non-rigid registration plus a seating pass brought both to 0%.
- **A glow that vanished on Apple GPUs.** The artery meshes had no normals, so the shader normalised a zero vector and produced NaN on Metal. Fixed by exporting normals, recomputing them when missing and making the shader NaN-safe.
- **Serverless size limits.** pandas + shap + scipy + scikit-learn exceeded the function size limit. Rewriting SHAP in numpy (verified exact) and making the inference path pandas-free cut the dependencies to what serving actually needs.

## Accomplishments we're proud of

- A negative result turned into a product feature instead of being hidden.
- Every number in the UI is read from artifacts produced by the evaluation script, never hard-coded.
- Tests that guard the claims: no label leakage, rendered colour equals API colour, arteries seated on the surface, SHAP exact.

## What we learned

- Per-vessel prediction from bedside data is mostly overall risk in disguise. Measuring *within-CAD* discrimination is the test that exposes it.
- Calibration and uncertainty display matter as much as AUC when the output is a colour on an organ.
- Guideline inputs are a strong baseline; ECG and echo add a small but real gain.

## What's next for CardioMap

- External validation on a second centre and a population not pre-selected for angiography.
- Inputs that carry real location signal (regional wall-motion by segment, ECG lead territories, CT calcium by vessel), with the evidence badges re-earned on that data.
- Patient-specific anatomy from CT angiography instead of a generic heart.

**Team:** Samarth Verma, Yashi Rajoriya, Anshuman Dubey. AI assistance (Claude Code, Claude) is disclosed in the README.

---

## Built with (tags)

python, scikit-learn, numpy, pandas, shap, fastapi, uvicorn, react, typescript, vite, three.js, react-three-fiber, drei, glsl, webgl, playwright, pytest, docker, vercel, trimesh, bodyparts3d, sketchfab, machine-learning, explainable-ai, claude-code

## "Try it out" links

- Live app: <Vercel URL, after the project is imported>
- Demo video: https://www.youtube.com/watch?v=l9Fn8tcCOsU
- Source code: https://github.com/samarthverma4/CardioMap
- Write-up (PDF): https://github.com/samarthverma4/CardioMap/blob/main/docs/writeup.pdf

## Image gallery (docs/devpost/, 1800×1200, 3:2)

1. `00_poster.jpg`: CardioMap, vessel scores beside P(CAD)
2. `01_patient_view.jpg`: full patient view, LAD selected
3. `02_defaults.jpg`: dataset-default patient
4. `03_rca_territory.jpg`: RCA territory tint and AHA bullseye
5. `04_shap_lad.jpg`: SHAP explanation in clinical units
6. `05_vessel_rows.jpg`: vessel scores with evidence badges
7. `06_vessel_auc.jpg`: vessel AUC: dedicated vs CAD probability vs within CAD
8. `07_ablation.jpg`: what each modality adds for CAD
9. `08_mobile.jpg`: mobile layout

## Video demo link

https://www.youtube.com/watch?v=l9Fn8tcCOsU
