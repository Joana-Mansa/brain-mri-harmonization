"""NIfTI validation, RAS orientation, spacing and an explicitly non-brain foreground mask."""

from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from scipy import ndimage


def load_image(path: str | Path) -> nib.Nifti1Image:
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"Expected a 3D volume, got {image.shape}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8:
        raise ValueError("Invalid or singular affine")
    data = image.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("Non-finite voxel intensities")
    if np.ptp(data) == 0:
        raise ValueError("Constant image")
    if np.any(np.array(image.header.get_zooms()[:3]) <= 0):
        raise ValueError("Non-positive voxel spacing")
    return image


def inspect_image(image: nib.Nifti1Image) -> dict:
    data = image.get_fdata(dtype=np.float32)
    metrics = {f"dimension_{a}": int(n) for a, n in zip("xyz", image.shape)}
    metrics.update({f"spacing_{a}": float(v) for a, v in zip("xyz", image.header.get_zooms())})
    metrics.update(
        {
            "orientation": "".join(nib.aff2axcodes(image.affine)),
            "spatial_units": image.header.get_xyzt_units()[0],
            "qform_code": int(image.header["qform_code"]),
            "sform_code": int(image.header["sform_code"]),
            "raw_min": float(data.min()),
            "raw_max": float(data.max()),
            "raw_mean": float(data.mean()),
            "raw_std": float(data.std()),
            "zero_fraction": float(np.mean(data == 0)),
        }
    )
    metrics.update({f"raw_p{p}": float(np.percentile(data, p)) for p in [1, 5, 50, 95, 99]})
    return metrics


def preprocess(image: nib.Nifti1Image, config: dict) -> nib.Nifti1Image:
    """Reorient and resample without intensity changes; preserve inputs on disk."""
    units = image.header.get_xyzt_units()[0]
    if units not in ("mm", "unknown"):
        raise ValueError(f"Unsupported spatial unit {units}; convert explicitly to mm")
    canonical = nib.as_closest_canonical(image)
    result = resample_to_output(
        canonical, voxel_sizes=config["spacing"], order=config["interpolation_order"]
    )
    result.header.set_xyzt_units("mm")
    return result


def foreground_mask(data: np.ndarray, percentile: float = 20) -> np.ndarray:
    """Largest connected positive-intensity component; includes non-brain tissue.

    This inexpensive fallback is not skull stripping. The percentile is a tunable
    algorithm parameter, not a validated quality cutoff.
    """
    positive = data[data > 0]
    if positive.size < 2:
        raise ValueError("Insufficient positive foreground")
    mask = data > np.percentile(positive, percentile)
    labels, n = ndimage.label(mask)
    if n == 0:
        raise ValueError("Empty foreground mask")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return ndimage.binary_fill_holes(labels == counts.argmax())


def normalize_intensity(data: np.ndarray, mask: np.ndarray, percentiles: list) -> np.ndarray:
    """Per-scan foreground percentile scaling; no cohort statistics or target needed."""
    lo, hi = np.percentile(data[mask], percentiles)
    if hi <= lo:
        raise ValueError("Degenerate normalization range")
    return ((np.clip(data, lo, hi) - lo) / (hi - lo)).astype(np.float32)
