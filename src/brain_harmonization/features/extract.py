import numpy as np
from scipy.stats import kurtosis, skew


def extract_features(data: np.ndarray, mask: np.ndarray, spacing: tuple) -> dict[str, float]:
    """Global foreground intensity distribution and local finite-difference summaries.

    No voxel correspondence or anatomical claim is assumed. Geometry is retained
    in QC but deliberately omitted from the harmonized model feature panel.
    """
    v = data[mask].astype(float)
    if v.size < 2 or not np.isfinite(v).all() or v.std() == 0:
        raise ValueError("Feature extraction requires finite nonconstant foreground")
    features = {f"intensity_p{p}": float(np.percentile(v, p)) for p in range(5, 100, 5)}
    features.update(
        mean=float(v.mean()), std=float(v.std()), skew=float(skew(v)), kurtosis=float(kurtosis(v))
    )
    for axis in range(3):
        gradient = np.abs(np.gradient(data, spacing[axis], axis=axis))[mask]
        features[f"gradient_{axis}_mean"] = float(gradient.mean())
        features[f"gradient_{axis}_p90"] = float(np.percentile(gradient, 90))
    if not np.isfinite(list(features.values())).all():
        raise ValueError("Nonfinite extracted features")
    return features
