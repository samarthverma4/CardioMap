# Track A reproducibility package

Reproduces every "own analysis" number in the Track A research report (Section 8).

## Files
- `reproduce.py`: the full experiment. Run `python reproduce.py` in this folder (about 20 min on one CPU core).
- `metrics.json`: all results (AUCs with 95% bootstrap CIs, paired differences, within-CAD AUCs, hierarchical vs independent, modality ablation, CAD calibration by sex).
- `oof_predictions.csv`: per-patient out-of-fold probabilities (averaged over 10 repeats) for every model, target and feature set, plus labels.
- `extention of Z-Alizadeh sani dataset.xlsx`: the file analysed.

## Data
Alizadehsani R., Roshanzamir M., Sani Z. *Extension of Z-Alizadeh Sani dataset*. UCI Machine Learning Repository, DOI 10.24432/C5461K. Licence CC BY 4.0.
SHA-256 of the file used: `739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b`.
This copy came from a public GitHub repo (UCI was not reachable from the analysis sandbox); its size matches the UCI file. Check the hash against a fresh UCI download.

## Design
- Splits: RepeatedStratifiedKFold, 5 folds x 10 repeats, seed 2026, stratified on the number of stenotic vessels. The same splits are used for all targets and models, so comparisons are paired.
- Models (prespecified, reported separately): L2 logistic regression (C=0.1, standardised); random forest (200 trees, min_samples_leaf=3).
- CIs: 2,000 patient-level bootstrap resamples of repeat-averaged out-of-fold scores; differences use paired bootstraps. These ignore model-refit variance, so they are somewhat narrow.
- Leakage: LAD, LCX, RCA and Cath are asserted absent from inputs. `Cath` is the CAD target. `Exertional CP` is dropped as constant.

## Environment
Python 3.12.3, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, openpyxl 3.1.5.

## AI assistance
Code and analysis were produced with Claude (Anthropic). Disclose this in Devpost "Built With" and your README if you reuse it.
