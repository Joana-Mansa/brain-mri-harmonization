import numpy as np
import pandas as pd
import pytest

from brain_harmonization.harmonization.combat import Combat
from brain_harmonization.models.baselines import split_subjects
from brain_harmonization.evaluation.statistics import paired_bootstrap


def test_split_reproducible_and_disjoint():
    table = pd.DataFrame(
        {"subject_id": [f"s{i}" for i in range(45)], "site": np.repeat(["A", "B", "C"], 15)}
    )
    first = split_subjects(table, 42, 0.2, 0.2)
    pd.testing.assert_frame_equal(first, split_subjects(table, 42, 0.2, 0.2))
    assert first.groupby("split").site.nunique().eq(3).all()
    assert len(first) == first.subject_id.nunique()
    with pytest.raises(ValueError, match="leakage"):
        split_subjects(pd.concat([table, table.iloc[:1]]), 42, 0.2, 0.2)


def test_combat_training_apply_and_unseen_site():
    rng = np.random.default_rng(12)
    sites = np.repeat(["A", "B", "C"], 20)
    x = rng.normal(size=(60, 29)) + np.repeat([0, 2, 5], 20)[:, None]
    combat = Combat().fit(x, sites)
    transformed = combat.transform(x, sites)
    np.testing.assert_allclose(transformed, combat.training_result, rtol=1e-6, atol=1e-6)
    before = np.var([x[sites == s].mean(axis=0) for s in np.unique(sites)])
    after = np.var([transformed[sites == s].mean(axis=0) for s in np.unique(sites)])
    assert after < before
    single = combat.transform(x[:1], sites[:1])
    together = combat.transform(np.vstack([x[:1], x[1:2] * 100]), sites[:2])
    np.testing.assert_allclose(single[0], together[0])
    with pytest.raises(ValueError, match="not part"):
        combat.transform(x[:1], np.array(["unknown"]))


def test_identical_predictions_have_zero_paired_difference():
    truth = pd.DataFrame({"site": ["A", "A", "B", "B"], "age": [20, 30, 40, 50]})
    p = {
        "site": truth.site.to_numpy(),
        "age": truth.age.to_numpy() + 1,
        "proba": np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]),
        "classes": np.array(["A", "B"]),
    }
    metrics, differences = paired_bootstrap(truth, {"baseline": p, "combat": p}, 20, 42)
    assert metrics.value.notna().all()
    assert differences[["delta", "ci_low", "ci_high"]].eq(0).all().all()
