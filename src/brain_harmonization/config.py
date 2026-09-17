"""Validated configuration; relative paths resolve against the YAML directory."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    required = {"dataset", "output", "subject_limit", "seed", "preprocessing", "qc", "experiment"}
    if set(config) != required:
        raise ValueError(f"Configuration keys must be exactly {sorted(required)}")
    c = deepcopy(config)
    for group, expected in {
        "dataset": {"root", "metadata", "kind"},
        "preprocessing": {
            "spacing",
            "interpolation_order",
            "mask_percentile",
            "normalization_percentiles",
        },
        "qc": {"robust_z", "report_subjects", "include_review"},
        "experiment": {
            "test_fraction",
            "validation_fraction",
            "bootstrap",
            "logistic_c",
            "ridge_alpha",
            "methods",
        },
    }.items():
        if not isinstance(c[group], dict) or set(c[group]) != expected:
            raise ValueError(f"{group} keys must be exactly {sorted(expected)}")
    for key in ("root", "metadata"):
        c["dataset"][key] = str((path.parent / c["dataset"][key]).resolve())
    c["output"] = str((path.parent / c["output"]).resolve())
    root, out = Path(c["dataset"]["root"]), Path(c["output"])
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Input and output trees must be separate")
    if c["dataset"]["kind"] not in {"IXI", "synthetic"}:
        raise ValueError("dataset.kind must be IXI or synthetic")
    for value, name, minimum in [
        (c["seed"], "seed", 0),
        (c["qc"]["report_subjects"], "report_subjects", 0),
        (c["experiment"]["bootstrap"], "bootstrap", 20),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if c["subject_limit"] is not None and (
        type(c["subject_limit"]) is not int or c["subject_limit"] < 1
    ):
        raise ValueError("subject_limit must be a positive integer or null")
    p, e = c["preprocessing"], c["experiment"]
    if len(p["spacing"]) != 3 or not np.all(np.isfinite(p["spacing"])) or min(p["spacing"]) <= 0:
        raise ValueError("spacing must contain three finite positive values")
    if p["interpolation_order"] not in (0, 1, 3):
        raise ValueError("interpolation_order must be 0, 1, or 3")
    lo, hi = p["normalization_percentiles"]
    if not 0 <= lo < hi <= 100 or not 0 <= p["mask_percentile"] < 100:
        raise ValueError("Invalid percentiles")
    if not np.isfinite(c["qc"]["robust_z"]) or c["qc"]["robust_z"] <= 0:
        raise ValueError("robust_z must be finite and positive")
    if type(c["qc"]["include_review"]) is not bool:
        raise ValueError("include_review must be boolean")
    if not (
        0 < e["test_fraction"] < 1
        and 0 < e["validation_fraction"] < 1
        and e["test_fraction"] + e["validation_fraction"] < 1
    ):
        raise ValueError("Split fractions must be positive and sum to less than 1")
    for key in ("logistic_c", "ridge_alpha"):
        if not e[key] or not np.all(np.isfinite(e[key])) or min(e[key]) <= 0:
            raise ValueError(f"{key} must contain finite positive values")
    if (
        not e["methods"]
        or len(set(e["methods"])) != len(e["methods"])
        or not set(e["methods"]) <= {"baseline", "robust_intensity", "combat"}
        or "baseline" not in e["methods"]
    ):
        raise ValueError("Methods must be unique, supported, and include baseline")
    return c
