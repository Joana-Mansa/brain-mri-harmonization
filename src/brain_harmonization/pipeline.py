"""Incremental disk-backed pipeline with a frozen cohort and common comparisons."""

import logging
import pickle
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from brain_harmonization.data.ixi import create_manifest
from brain_harmonization.evaluation.report import figures, research_report
from brain_harmonization.evaluation.statistics import paired_bootstrap
from brain_harmonization.features.extract import extract_features
from brain_harmonization.harmonization.combat import Combat, eligible_features
from brain_harmonization.models.baselines import fit_models, split_subjects
from brain_harmonization.preprocessing.images import (
    foreground_mask,
    inspect_image,
    load_image,
    normalize_intensity,
    preprocess,
)
from brain_harmonization.qc.metrics import apply_reference, fit_reference, quality_metrics
from brain_harmonization.qc.report import slice_report
from brain_harmonization.utils.provenance import provenance, sha256, write_json

LOG = logging.getLogger(__name__)


def prepare(config: dict) -> Path:
    out = Path(config["output"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new run directory.")
    manifest, unknown = create_manifest(config)
    for directory in [out, out / "tables", out / "figures", out / "derivatives", out / "models"]:
        directory.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "tables/manifest.csv", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(out / "provenance.json", provenance(config))
    write_json(
        out / "dataset_summary.json",
        {
            "total_records": len(manifest),
            "selected": int(manifest.selected.sum()),
            "status": manifest.status.value_counts().to_dict(),
            "sites": manifest.site.value_counts().to_dict(),
            "unrecognized_nifti_files": unknown,
            "stage": "metadata_and_file_inventory_only",
        },
    )
    LOG.info(
        "Prepared %d selected subjects from %d records", manifest.selected.sum(), len(manifest)
    )
    return out


def check_run(config: dict) -> Path:
    out = Path(config["output"])
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    if saved != config:
        raise ValueError("Run configuration changed; start a new output directory")
    return out


def process(config: dict) -> None:
    out = check_run(config)
    if (out / "tables/qc_initial.csv").exists():
        raise FileExistsError("Preprocessing already completed; use a new run directory")
    manifest = pd.read_csv(out / "tables/manifest.csv", keep_default_na=False)
    qc_rows, feature_rows = [], {"baseline": [], "robust_intensity": []}
    for _, row in manifest[manifest.selected].iterrows():
        record = row.to_dict()
        try:
            path = Path(row.path)
            record["input_sha256"] = sha256(path)
            image = load_image(path)
            record.update(inspect_image(image))
            if record["spatial_units"] == "unknown":
                record["status"] = "REVIEW"
                record["reasons"] += ";unknown_spatial_units_assumed_mm"
            processed = preprocess(image, config["preprocessing"])
            data = processed.get_fdata(dtype=np.float32)
            mask = foreground_mask(data, config["preprocessing"]["mask_percentile"])
            spacing = processed.header.get_zooms()
            record.update(quality_metrics(data, mask, spacing))
            normalized = normalize_intensity(
                data, mask, config["preprocessing"]["normalization_percentiles"]
            )
            # Extract both before appending either, so failures cannot misalign methods.
            raw_features = extract_features(data, mask, spacing)
            norm_features = extract_features(normalized, mask, spacing)
            nib.save(processed, out / "derivatives" / f"{row.subject_id}_preprocessed.nii.gz")
            nib.save(
                nib.Nifti1Image(mask.astype(np.uint8), processed.affine),
                out / "derivatives" / f"{row.subject_id}_foreground.nii.gz",
            )
            for method, values in [("baseline", raw_features), ("robust_intensity", norm_features)]:
                feature_rows[method].append({"subject_id": row.subject_id, **values})
        except Exception as exc:
            LOG.exception("Subject %s failed", row.subject_id)
            record["status"] = "EXCLUDE"
            record["reasons"] += f";{type(exc).__name__}:{exc}"
        qc_rows.append(record)
    pd.DataFrame(qc_rows).to_csv(out / "tables/qc_initial.csv", index=False)
    for method, rows in feature_rows.items():
        pd.DataFrame(rows).to_csv(out / "tables" / f"features_{method}.csv", index=False)
    LOG.info("Processed %d subjects; failures remain in qc_initial.csv", len(qc_rows))


def qc(config: dict) -> None:
    out = check_run(config)
    if (out / "tables/qc.csv").exists():
        raise FileExistsError("QC already completed; use a new run directory")
    table = pd.read_csv(out / "tables/qc_initial.csv")
    table.age = pd.to_numeric(table.age, errors="coerce")
    # Shared age-complete cohort enables paired site/age comparisons.
    eligible = table[(table.status != "EXCLUDE") & np.isfinite(table.age)].copy()
    if eligible.empty:
        raise ValueError("No processed subjects with age metadata")
    splits = split_subjects(
        eligible,
        config["seed"],
        config["experiment"]["test_fraction"],
        config["experiment"]["validation_fraction"],
    )
    train_ids = splits.loc[splits.split == "train", "subject_id"]
    reference = fit_reference(eligible[eligible.subject_id.isin(train_ids)])
    checked = apply_reference(table[table.status != "EXCLUDE"], reference, config["qc"]["robust_z"])
    checked = pd.concat([checked, table[table.status == "EXCLUDE"]]).sort_values("subject_id")
    checked["analysis_eligible"] = (checked.status != "EXCLUDE") & np.isfinite(checked.age)
    if not config["qc"]["include_review"]:
        checked["analysis_eligible"] &= checked.status == "PASS"
    splits["analysis_included"] = splits.subject_id.isin(
        checked.loc[checked.analysis_eligible, "subject_id"]
    )
    checked.to_csv(out / "tables/qc.csv", index=False)
    splits.to_csv(out / "tables/splits.csv", index=False)
    write_json(
        out / "qc_reference.json",
        {"training_subjects": train_ids.tolist(), "statistics": reference},
    )
    write_json(
        out / "qc_summary.json",
        {
            "processed": len(checked),
            "status": checked.status.value_counts().to_dict(),
            "analysis_included": int(checked.analysis_eligible.sum()),
            "by_site_status": checked.groupby(["site", "status"])
            .size()
            .reset_index(name="n")
            .to_dict("records"),
        },
    )
    # Review flags take priority in the bounded visual report set.
    visual = checked[checked.status != "EXCLUDE"].assign(
        priority=lambda t: (t.status != "REVIEW").astype(int)
    )
    visual = visual.sort_values(["priority", "subject_id"]).head(config["qc"]["report_subjects"])
    for _, row in visual.iterrows():
        data = nib.load(out / "derivatives" / f"{row.subject_id}_preprocessed.nii.gz").get_fdata()
        mask = nib.load(out / "derivatives" / f"{row.subject_id}_foreground.nii.gz").get_fdata() > 0
        slice_report(data, mask, row, out / "figures" / f"qc_{row.subject_id}")
    LOG.info("QC complete: %s", checked.status.value_counts().to_dict())


def analyze(config: dict) -> None:
    out = check_run(config)
    if (out / "report.md").exists():
        raise FileExistsError("Analysis already completed; use a new output directory")
    quality = pd.read_csv(out / "tables/qc.csv")
    splits = pd.read_csv(out / "tables/splits.csv")
    table = quality.merge(
        splits[["subject_id", "split", "analysis_included"]], on="subject_id", validate="one_to_one"
    )
    table = table[table.analysis_included].sort_values("subject_id").reset_index(drop=True)
    for site, group in table.groupby("site"):
        counts = group.split.value_counts()
        if (
            counts.get("train", 0) < 3
            or counts.get("test", 0) < 1
            or counts.get("validation", 0) < 1
        ):
            raise ValueError(f"Insufficient subjects after QC at {site}; inspect splits.csv")
    if table.site.nunique() < 2:
        raise ValueError("Need multiple sites after QC")
    raw = (
        pd.read_csv(out / "tables/features_baseline.csv")
        .set_index("subject_id")
        .loc[table.subject_id]
    )
    norm = (
        pd.read_csv(out / "tables/features_robust_intensity.csv")
        .set_index("subject_id")
        .loc[table.subject_id]
    )
    train = (table.split == "train").to_numpy()
    test = (table.split == "test").to_numpy()
    sites = table.site.to_numpy()
    keep = eligible_features(raw.to_numpy()[train], sites[train])
    # Same feature columns in every arm, screened using training partitions only.
    keep &= eligible_features(norm.to_numpy()[train], sites[train])
    if keep.sum() < 3:
        raise ValueError("Too few common variable features")
    write_json(
        out / "feature_panel.json",
        {"retained": raw.columns[keep].tolist(), "removed": raw.columns[~keep].tolist()},
    )
    methods = config["experiment"]["methods"]
    x = raw.to_numpy(float)[:, keep]
    representations = {"baseline": x}
    if "robust_intensity" in methods:
        representations["robust_intensity"] = norm.to_numpy(float)[:, keep]
    if "combat" in methods:
        harmonizer = Combat().fit(x[train], sites[train])
        representations["combat"] = harmonizer.transform(x, sites)
        write_json(out / "combat_estimates.json", harmonizer.estimates)
        with (out / "models/combat.pkl").open("wb") as stream:
            pickle.dump(harmonizer, stream)
    predictions, settings = {}, {}
    for method, features in representations.items():
        site_model, age_model, parameters = fit_models(features, table, config["experiment"])
        settings[method] = parameters
        predicted = {
            "site": site_model.predict(features[test]),
            "age": age_model.predict(features[test]),
            "proba": site_model.predict_proba(features[test]),
            "classes": site_model.classes_,
        }
        predictions[method] = predicted
        frame = table.loc[test, ["subject_id", "site", "age"]].copy()
        frame["predicted_site"], frame["predicted_age"] = predicted["site"], predicted["age"]
        for i, site in enumerate(predicted["classes"]):
            frame[f"probability_{site}"] = predicted["proba"][:, i]
        frame.to_csv(out / "tables" / f"predictions_{method}.csv", index=False)
        pd.DataFrame(features, columns=raw.columns[keep], index=table.subject_id).to_csv(
            out / "tables" / f"analysis_features_{method}.csv"
        )
        with (out / "models" / f"{method}.pkl").open("wb") as stream:
            pickle.dump({"site": site_model, "age": age_model}, stream)
    metrics, differences = paired_bootstrap(
        table[test], predictions, config["experiment"]["bootstrap"], config["seed"]
    )
    metrics.to_csv(out / "tables/metrics.csv", index=False)
    differences.to_csv(out / "tables/paired_differences.csv", index=False)
    table.to_csv(out / "tables/analysis_cohort.csv", index=False)
    benchmarks = {
        "mean_age_mae": mean_absolute_error(
            table.age[test], np.repeat(table.age[train].mean(), test.sum())
        ),
        "site_mean_age_mae": mean_absolute_error(
            table.age[test], table.site[test].map(table[train].groupby("site").age.mean())
        ),
    }
    write_json(out / "model_selection.json", settings)
    write_json(out / "age_benchmarks.json", benchmarks)
    figures(
        out,
        table,
        quality,
        representations,
        predictions,
        metrics,
        config["dataset"]["kind"] == "synthetic",
    )
    research_report(out, config, table, metrics, differences, benchmarks)
    write_json(
        out / "completion.json",
        {"status": "complete", "analysis_subjects": len(table), "methods": list(representations)},
    )
    LOG.info("Analysis complete: %s", out / "report.md")
