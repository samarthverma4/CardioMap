"""Requirement 1e: accuracy, precision, recall, F1 and ROC-AUC are reported for CAD and each vessel."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CM = json.loads((ROOT / "artifacts" / "metrics.json").read_text())["derived"]["classification_metrics"]
REPRO = json.loads((ROOT / "ml" / "metrics.json").read_text())


@pytest.mark.parametrize("fam", ["LR", "RF"])
@pytest.mark.parametrize("target", ["CAD", "LAD", "LCX", "RCA"])
def test_metrics_present_and_consistent(fam, target):
    r = CM[fam][target]
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "specificity"]:
        v = r[k]
        assert 0 <= v["ci95"][0] <= v["value"] <= v["ci95"][1] <= 1, (k, v)
    # same evaluation as reproduce.py: ROC-AUC must match its all-inputs AUC
    assert r["roc_auc"]["value"] == pytest.approx(REPRO[fam]["ablation"]["all"][target]["auc"], abs=0.001)
    c = r["confusion"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == REPRO["n"]
