import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from brain_harmonization.features.extract import extract_features
from brain_harmonization.preprocessing.images import (
    foreground_mask,
    load_image,
    normalize_intensity,
    preprocess,
)
from brain_harmonization.qc.metrics import apply_reference, fit_reference, quality_metrics


def phantom():
    grid = np.indices((20, 22, 24))
    r = sum(((grid[i] - 10) / 8) ** 2 for i in range(3))
    return np.where(r < 1, 100 - 30 * r, 0).astype(np.float32)


def test_nifti_preprocessing_and_features(tmp_path):
    data = phantom()
    p = tmp_path / "scan.nii.gz"
    image = nib.Nifti1Image(data, np.diag([-1.0, 1.0, 1.0, 1.0]))
    image.header.set_xyzt_units("mm")
    nib.save(image, p)
    before = p.read_bytes()
    processed = preprocess(load_image(p), {"spacing": [2, 2, 2], "interpolation_order": 1})
    assert nib.aff2axcodes(processed.affine) == ("R", "A", "S")
    assert processed.header.get_zooms() == (2, 2, 2)
    assert p.read_bytes() == before
    data = processed.get_fdata()
    mask = foreground_mask(data)
    features = extract_features(data, mask, (2, 2, 2))
    assert len(features) == 29 and np.isfinite(list(features.values())).all()
    normalized = normalize_intensity(data, mask, [1, 99])
    assert normalized.min() >= 0 and normalized.max() <= 1
    metrics = quality_metrics(data, mask, (2, 2, 2))
    assert metrics["foreground_volume_ml"] == mask.sum() * 8 / 1000


@pytest.mark.parametrize("shape, value", [((4, 4, 4, 2), 1), ((4, 4, 4), 0), ((4, 4, 4), np.nan)])
def test_invalid_image(tmp_path, shape, value):
    p = tmp_path / "bad.nii"
    nib.save(nib.Nifti1Image(np.full(shape, value, dtype=np.float32), np.eye(4)), p)
    with pytest.raises(ValueError):
        load_image(p)


def test_corrupt_image(tmp_path):
    p = tmp_path / "bad.nii.gz"
    p.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        load_image(p)


def test_robust_qc_only_flags_outlier():
    from brain_harmonization.qc.metrics import QC_COLUMNS

    train = pd.DataFrame({k: [1, 2, 3, 4, 5] for k in QC_COLUMNS})
    target = pd.DataFrame({k: [3, 100] for k in QC_COLUMNS})
    target["status"], target["reasons"] = "PASS", ""
    result = apply_reference(target, fit_reference(train), 3.5)
    assert list(result.status) == ["PASS", "REVIEW"]
