import numpy as np
import pandas as pd
from scipy.stats import entropy


def quality_metrics(data: np.ndarray, mask: np.ndarray, spacing: tuple) -> dict:
    values, background = data[mask], data[~mask]
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid foreground for QC")
    hist, _ = np.histogram(values, bins=64)
    bg_std = float(background.std()) if background.size else np.nan
    bg_mean = float(background.mean()) if background.size else np.nan
    return {
        "foreground_mean": float(values.mean()),
        "foreground_std": float(values.std()),
        "foreground_volume_ml": float(mask.sum() * np.prod(spacing) / 1000),
        "foreground_fraction": float(mask.mean()),
        "entropy": float(entropy(hist, base=2)),
        "background_mean": bg_mean,
        "background_std": bg_std,
        "contrast_proxy": float(values.mean() - bg_mean),
        "snr_proxy": float(values.mean() / bg_std) if bg_std > 0 else np.nan,
    }


QC_COLUMNS = [
    "foreground_mean",
    "foreground_std",
    "foreground_volume_ml",
    "foreground_fraction",
    "entropy",
    "zero_fraction",
]


def fit_reference(table: pd.DataFrame) -> dict:
    """Fit robust reference statistics on training subjects only."""
    ref = {}
    for name in QC_COLUMNS:
        values = table[name].dropna().to_numpy(float)
        if not len(values):
            continue
        median = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - median)))
        if scale == 0:
            scale = float((np.percentile(values, 75) - np.percentile(values, 25)) / 1.349)
        ref[name] = {"median": median, "scale": scale}
    return ref


def apply_reference(table: pd.DataFrame, reference: dict, threshold: float) -> pd.DataFrame:
    result = table.copy()
    result["reasons"] = result["reasons"].fillna("").astype(str)
    for idx, row in result.iterrows():
        reasons = [r for r in str(row.get("reasons", "")).split(";") if r and r != "nan"]
        for name, stats in reference.items():
            value = row[name]
            if pd.isna(value):
                reasons.append(f"missing_qc:{name}")
            elif stats["scale"] > 0 and abs(value - stats["median"]) / stats["scale"] > threshold:
                reasons.append(f"robust_outlier:{name}")
            elif stats["scale"] == 0 and not np.isclose(value, stats["median"]):
                reasons.append(f"deviation_from_constant:{name}")
        if reasons and row["status"] != "EXCLUDE":
            result.loc[idx, "status"] = "REVIEW"
        result.loc[idx, "reasons"] = ";".join(dict.fromkeys(reasons))
    return result
