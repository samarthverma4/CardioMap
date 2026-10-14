"""Load + encode the Z-Alizadeh Sani extension dataset, exactly as ml/reproduce.py does.

This module is the single source of truth for inputs. `feature_schema.json` is exported from here,
and both ml/train.py and the backend use `encode_rows` to turn clinical inputs into model columns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "extention of Z-Alizadeh sani dataset.xlsx"
DATA_SHA256 = "739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b"

LABELS = ["LAD", "LCX", "RCA", "Cath"]  # never model inputs; Cath is the CAD target
VESSELS = ["LAD", "LCX", "RCA"]
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
DROPPED = ["Exertional CP"]  # constant ('N') in the dataset

ESC = ["Age", "Sex", "Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
       "FH", "Current Smoker", "EX-Smoker", "DLP", "HTN", "DM"]
ECG = ["Tinversion", "St Depression", "St Elevation", "Q Wave", "LVH", "Poor R Progression"]
ECHO = ["EF-TTE", "Region RWMA", "VHD"]

VHD_LEVELS = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}
BBB_LEVELS = ["N", "LBBB", "RBBB"]

# name -> (display label, unit, type, group). Type: binary | ordinal | continuous | categorical.
INPUTS: dict[str, tuple[str, str, str, str]] = {
    "Age": ("Age", "years", "continuous", "demographic"),
    "Sex": ("Sex", "", "binary", "demographic"),
    "Weight": ("Weight", "kg", "continuous", "demographic"),
    "Length": ("Height", "cm", "continuous", "demographic"),
    "BMI": ("Body-mass index", "kg/m²", "continuous", "demographic"),
    "DM": ("Diabetes mellitus", "", "binary", "history"),
    "HTN": ("Hypertension", "", "binary", "history"),
    "Current Smoker": ("Current smoker", "", "binary", "history"),
    "EX-Smoker": ("Ex-smoker", "", "binary", "history"),
    "FH": ("Family history of CAD", "", "binary", "history"),
    "Obesity": ("Obesity", "", "binary", "history"),
    "CRF": ("Chronic renal failure", "", "binary", "history"),
    "CVA": ("Prior stroke (CVA)", "", "binary", "history"),
    "Airway disease": ("Airway disease", "", "binary", "history"),
    "Thyroid Disease": ("Thyroid disease", "", "binary", "history"),
    "CHF": ("Congestive heart failure", "", "binary", "history"),
    "DLP": ("Dyslipidaemia", "", "binary", "history"),
    "Typical Chest Pain": ("Typical chest pain", "", "binary", "symptoms"),
    "Atypical": ("Atypical chest pain", "", "binary", "symptoms"),
    "Nonanginal": ("Non-anginal chest pain", "", "binary", "symptoms"),
    "Dyspnea": ("Dyspnoea", "", "binary", "symptoms"),
    "Function Class": ("Functional class", "class", "ordinal", "symptoms"),
    "LowTH Ang": ("Low-threshold angina", "", "binary", "symptoms"),
    "BP": ("Blood pressure", "mmHg", "continuous", "exam"),
    "PR": ("Pulse rate", "bpm", "continuous", "exam"),
    "Edema": ("Oedema", "", "binary", "exam"),
    "Weak Peripheral Pulse": ("Weak peripheral pulse", "", "binary", "exam"),
    "Lung rales": ("Lung rales", "", "binary", "exam"),
    "Systolic Murmur": ("Systolic murmur", "", "binary", "exam"),
    "Diastolic Murmur": ("Diastolic murmur", "", "binary", "exam"),
    "Q Wave": ("Q wave", "", "binary", "ecg"),
    "St Elevation": ("ST elevation", "", "binary", "ecg"),
    "St Depression": ("ST depression", "", "binary", "ecg"),
    "Tinversion": ("T-wave inversion", "", "binary", "ecg"),
    "LVH": ("Left ventricular hypertrophy", "", "binary", "ecg"),
    "Poor R Progression": ("Poor R-wave progression", "", "binary", "ecg"),
    "BBB": ("Bundle branch block", "", "categorical", "ecg"),
    "FBS": ("Fasting blood sugar", "mg/dL", "continuous", "lab"),
    "CR": ("Creatinine", "mg/dL", "continuous", "lab"),
    "TG": ("Triglycerides", "mg/dL", "continuous", "lab"),
    "LDL": ("LDL cholesterol", "mg/dL", "continuous", "lab"),
    "HDL": ("HDL cholesterol", "mg/dL", "continuous", "lab"),
    "BUN": ("Blood urea nitrogen", "mg/dL", "continuous", "lab"),
    "ESR": ("ESR", "mm/h", "continuous", "lab"),
    "HB": ("Haemoglobin", "g/dL", "continuous", "lab"),
    "K": ("Potassium", "mEq/L", "continuous", "lab"),
    "Na": ("Sodium", "mEq/L", "continuous", "lab"),
    "WBC": ("White cell count", "cells/µL", "continuous", "lab"),
    "Lymph": ("Lymphocytes", "%", "continuous", "lab"),
    "Neut": ("Neutrophils", "%", "continuous", "lab"),
    "PLT": ("Platelets", "×10³/µL", "continuous", "lab"),
    "EF-TTE": ("Ejection fraction (TTE)", "%", "continuous", "echo"),
    "Region RWMA": ("Regional wall-motion abnormality", "regions", "ordinal", "echo"),
    "VHD": ("Valvular heart disease", "", "ordinal", "echo"),
}
GROUPS = ["demographic", "history", "symptoms", "exam", "ecg", "lab", "echo"]
OPTION_LABELS = {
    "VHD": {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"},
    "Sex": {0: "Female", 1: "Male"},
}


def file_sha256(path: Path = DATA_FILE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    assert file_sha256(path) == DATA_SHA256, "dataset hash mismatch"
    return pd.read_excel(path)


def targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    Y = {"CAD": df.Cath.eq("CAD").astype(int).values}
    for v in VESSELS:
        Y[v] = df[v].eq("Stenotic").astype(int).values
    return Y


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding copied from reproduce.py. Returns the model matrix (float) in reproduce.py's column order."""
    X = df.drop(columns=[c for c in LABELS + DROPPED if c in df.columns]).copy()
    X["Sex"] = X.Sex.eq("Male").astype(int)
    X["VHD"] = X.VHD.map(VHD_LEVELS)
    X = pd.get_dummies(X, columns=["BBB"], drop_first=True)
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("str", "string"):
            X[c] = X[c].map({"Y": 1, "N": 0})
    X = X.astype(float)
    assert_no_labels(X.columns)
    assert X.notna().all().all()
    return X


def assert_no_labels(columns) -> None:
    leaked = set(LABELS) & set(columns)
    assert not leaked, f"label leakage: {sorted(leaked)}"


def model_columns() -> list[str]:
    """Model matrix columns: inputs in dataset order, BBB replaced by its drop-first dummies at the end."""
    cols = [c for c in INPUTS if c != "BBB"]
    order = list(load_raw().columns)
    cols.sort(key=order.index)
    return cols + ["BBB_N", "BBB_RBBB"]


def clinical_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Raw dataset rows -> clinical input dicts in the schema's encoding (binary 0/1, VHD 0-3, BBB string)."""
    out = pd.DataFrame(index=df.index)
    for name, (_, _, typ, _) in INPUTS.items():
        s = df[name]
        if name == "Sex":
            out[name] = s.eq("Male").astype(int)
        elif name == "VHD":
            out[name] = s.map(VHD_LEVELS)
        elif name == "BBB":
            out[name] = s
        elif s.dtype == object or str(s.dtype) in ("str", "string"):
            out[name] = s.map({"Y": 1, "N": 0})
        else:
            out[name] = s
    return out


def encode_rows(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Clinical input dicts (schema encoding, all inputs present) -> model matrix with `columns`."""
    recs = []
    for r in rows:
        assert_no_labels(r.keys())
        rec = {k: float(v) for k, v in r.items() if k != "BBB"}
        rec["BBB_N"] = float(r["BBB"] == "N")
        rec["BBB_RBBB"] = float(r["BBB"] == "RBBB")
        recs.append(rec)
    X = pd.DataFrame(recs)[columns].astype(float)
    assert_no_labels(X.columns)
    return X


def build_schema(df: pd.DataFrame) -> dict:
    C = clinical_frame(df)
    inputs = []
    for name, (label, unit, typ, group) in INPUTS.items():
        s = C[name]
        item = {"name": name, "label": label, "unit": unit, "type": typ, "group": group,
                "in_esc_set": name in ESC}
        if typ == "categorical":
            item["options"] = [{"value": v, "label": {"N": "None", "LBBB": "Left BBB", "RBBB": "Right BBB"}[v]}
                               for v in BBB_LEVELS]
            item["default"] = str(s.mode().iloc[0])
        else:
            lo, hi = float(s.min()), float(s.max())
            item["range"] = [lo, hi]
            if typ == "continuous":
                med = float(s.median())
                item["default"] = round(med, 1) if name == "BMI" else (int(med) if med.is_integer() else round(med, 2))
                item["step"] = 0.1 if not np.allclose(s, s.round()) else 1
            else:
                item["default"] = int(s.mode().iloc[0])
                vals = sorted(int(v) for v in s.unique())
                labels = OPTION_LABELS.get(name, {})
                item["options"] = [{"value": v, "label": labels.get(v, ("Yes" if v else "No") if typ == "binary" else str(v))}
                                   for v in vals]
        if name == "BMI":
            item["derived_from"] = "Weight / (Height/100)^2; computed when not supplied"
        inputs.append(item)
    return {
        "dataset_sha256": DATA_SHA256,
        "n_rows": int(len(df)),
        "groups": GROUPS,
        "esc_features": ESC,
        "ecg_features": ECG,
        "echo_features": ECHO,
        "label_fields_rejected": LABELS,
        "dropped": {"Exertional CP": "constant ('N') in the dataset"},
        "range_note": "Allowed range = range observed in the 303 training rows; values outside are rejected.",
        "default_note": "Default = dataset median (continuous) or mode (binary/ordinal/categorical).",
        "inputs": inputs,
    }


class InputError(ValueError):
    """Bad request payload (unknown field, label field, out of range)."""


def complete_inputs(features: dict, schema: dict) -> dict:
    """Validate a request's features and fill missing ones with schema defaults."""
    spec = {i["name"]: i for i in schema["inputs"]}
    label_hits = sorted(set(features) & set(LABELS))
    if label_hits:
        raise InputError(f"label fields are not accepted as inputs: {label_hits}")
    unknown = sorted(set(features) - set(spec))
    if unknown:
        raise InputError(f"unknown fields: {unknown}")
    out = {}
    for name, it in spec.items():
        if name not in features or features[name] is None:
            continue
        v = features[name]
        if it["type"] == "categorical":
            allowed = [o["value"] for o in it["options"]]
            if v not in allowed:
                raise InputError(f"{name} must be one of {allowed}")
            out[name] = v
            continue
        if isinstance(v, bool):
            v = int(v)
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            raise InputError(f"{name} must be a number")
        if it["type"] in ("binary", "ordinal") and v not in [o["value"] for o in it["options"]]:
            raise InputError(f"{name} must be one of {[o['value'] for o in it['options']]}")
        lo, hi = it["range"]
        if not lo <= v <= hi:
            raise InputError(f"{name}={v} is outside the range seen in the training data [{lo}, {hi}]")
        out[name] = v
    if "BMI" not in out and "Weight" in out and "Length" in out:
        bmi = out["Weight"] / (out["Length"] / 100) ** 2
        lo, hi = spec["BMI"]["range"]
        if not lo <= bmi <= hi:
            raise InputError(f"BMI from weight/height = {bmi:.1f}, outside the training range [{lo:.1f}, {hi:.1f}]")
        out["BMI"] = bmi
    for name, it in spec.items():
        out.setdefault(name, it["default"])
    return out


if __name__ == "__main__":
    df = load_raw()
    print(encode(df).shape, len(INPUTS), "inputs")
    print(json.dumps(build_schema(df)["inputs"][:3], indent=1, ensure_ascii=False))
