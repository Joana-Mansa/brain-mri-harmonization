import json

import pandas as pd
import pytest

from brain_harmonization.config import load_config
from brain_harmonization.data.synthetic import create_demo
from brain_harmonization.pipeline import analyze, prepare, process, qc


def test_end_to_end_synthetic(tmp_path):
    path = create_demo(tmp_path / "phantoms", tmp_path / "outputs", subjects=30, bootstrap=20)
    config = load_config(path)
    out = prepare(config)
    with pytest.raises(FileExistsError):
        prepare(config)
    process(config)
    qc(config)
    analyze(config)
    assert json.loads((out / "completion.json").read_text())["status"] == "complete"
    metrics = pd.read_csv(out / "tables/metrics.csv")
    assert len(metrics) == 18
    assert metrics[["value", "ci_low", "ci_high"]].notna().all().all()
    assert "SYNTHETIC SOFTWARE VALIDATION ONLY" in (out / "report.md").read_text()
    cohort = pd.read_csv(out / "tables/analysis_cohort.csv")
    assert cohort.subject_id.is_unique
    baseline = pd.read_csv(out / "tables/predictions_baseline.csv")
    combat = pd.read_csv(out / "tables/predictions_combat.csv")
    assert baseline.subject_id.equals(combat.subject_id)
    assert set(baseline.subject_id).isdisjoint(cohort.loc[cohort.split == "train", "subject_id"])
    assert (out / "figures/performance.pdf").stat().st_size > 1000
