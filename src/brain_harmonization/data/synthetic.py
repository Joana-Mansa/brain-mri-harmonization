"""Small artificial phantoms for software verification, never empirical MRI evidence."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml


def create_demo(
    destination: Path, output: Path, subjects: int = 60, seed: int = 42, bootstrap: int = 100
) -> Path:
    if subjects < 18:
        raise ValueError("Demo requires >=18 subjects")
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Demo directory must be new or empty")
    images = destination / "T1"
    images.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    grid = np.indices((24, 26, 28)).astype(float)
    rows = []
    for i in range(subjects):
        site = ["Guys", "HH", "IOP"][i % 3]
        age = float(rng.uniform(20, 80))
        radius = sum(((grid[j] - (grid.shape[j + 1] - 1) / 2) / (8 + j)) ** 2 for j in range(3))
        mask = radius < 1
        # Age changes the radial profile; site adds scale and offset. Pure simulation.
        signal = 100 + (age - 20) * 0.25 - (20 + age * 0.2) * radius
        data = np.where(
            mask, signal * (1 + (i % 3) * 0.3) + (i % 3) * 15 + rng.normal(0, 2, radius.shape), 0
        ).astype(np.float32)
        image = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        image.header.set_xyzt_units("mm")
        nib.save(image, images / f"IXI{i + 1:03d}-{site}-SYNTHETIC-T1.nii.gz")
        rows.append(
            {
                "IXI_ID": i + 1,
                "AGE": age,
                "SEX_ID (1=m, 2=f)": int(rng.integers(1, 3)),
                "source": "SYNTHETIC PHANTOM",
            }
        )
    pd.DataFrame(rows).to_csv(destination / "demographics.csv", index=False)
    (destination / "SYNTHETIC.txt").write_text("Artificial phantoms. Not IXI or human MRI data.\n")
    config = {
        "dataset": {
            "root": str(images),
            "metadata": str(destination / "demographics.csv"),
            "kind": "synthetic",
        },
        "output": str(output.resolve()),
        "subject_limit": subjects,
        "seed": seed,
        "preprocessing": {
            "spacing": [2.0, 2.0, 2.0],
            "interpolation_order": 1,
            "mask_percentile": 20,
            "normalization_percentiles": [1, 99],
        },
        "qc": {"robust_z": 3.5, "report_subjects": 3, "include_review": True},
        "experiment": {
            "test_fraction": 0.2,
            "validation_fraction": 0.2,
            "bootstrap": bootstrap,
            "logistic_c": [0.1, 1.0, 10.0],
            "ridge_alpha": [0.1, 1.0, 10.0, 100.0],
            "methods": ["baseline", "robust_intensity", "combat"],
        },
    }
    path = destination / "demo.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
