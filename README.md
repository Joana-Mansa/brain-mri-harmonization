# Quality Control and Harmonization of Multi-Centre Brain MRI

A modular Python research pipeline asking: **can MRI harmonization reduce acquisition-site information while preserving age-related information?**

The software runs end to end on generated NIfTI phantoms. **An empirical IXI experiment has not yet been performed.** Synthetic metrics demonstrate execution, not neuroimaging findings.

## Implemented

- IXI T1 parsing, original demographic retention, missing/duplicate record audits, deterministic site-balanced cohort selection.
- NIfTI validation, RAS orientation, configurable resampling, foreground masks, raw and percentile-normalized image features.
- Interpretable QC, training-relative MAD/IQR flags, PASS/REVIEW/EXCLUDE labels and three-plane visual reports.
- Subject-level splits, balanced logistic site classification, ridge age regression and validation-based regularization.
- Baseline, image-level percentile normalization, and training-only feature-level empirical Bayes ComBat.
- Balanced accuracy, macro F1, ROC-AUC, confusion matrices, age MAE/RMSE/R², paired bootstrap intervals and naive age benchmarks.
- PNG/vector PDF figures, CSV tables, fitted models, input hashes, configuration/version provenance and Markdown research reports.

This first baseline uses **whole-foreground image summaries**, not anatomical measurements. The mask may include skull/scalp; it is not skull stripping. Registration is deferred because these features do not require voxel correspondence. Brain volume and registration similarity are not reported when neither has been measured.

## Installation

Python 3.10–3.12 is targeted. Use an isolated environment.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
ruff check .
ruff format --check .
```

Direct dependencies are pinned in pyproject.toml. requirements-lock.txt records the tested environment including transitive packages. Recreate it with `pip install -r requirements-lock.txt`, then `pip install -e . --no-deps`. Every run records actual package versions. GitHub Actions tests synthetic data without IXI.

## Synthetic demonstration

```bash
brain-harmonization demo --data demo-data --output results/demo --subjects 60
```

Open results/demo/report.md and results/demo/figures/performance.png. Artificial images are labeled in metadata and reports. The simulation deliberately inserts age- and site-related intensity changes; success does not establish validity on human MRI.

## Download and arrange IXI

1. Read the [official IXI page](https://brain-development.org/ixi-dataset/) and CC BY-SA 3.0 terms. Download the **T1 images** archive and **demographic spreadsheet** using its links. Acknowledge IXI in derived work. The software license does not replace the data license.
2. Extract T1 files locally, retaining original filenames. The pipeline never writes to the input directory.
3. Arrange files below, or set dataset.root and dataset.metadata in a copied configuration.

```text
data/ixi/
├── IXI.xls
└── T1/
    ├── IXI002-Guys-0828-T1.nii.gz
    └── ...
```

CSV exports with IXI_ID, AGE, and SEX_ID (1=m, 2=f) are supported, as are normalized subject_id, age, sex columns. Extra columns are preserved as JSON. Sex is retained as supplied, not inferred.

IXI describes Hammersmith (Philips 3T), Guy's (Philips 1.5T), and Institute of Psychiatry (GE 1.5T). These are **site-level descriptions**, not per-scan verification. The manifest labels that provenance and retains filename acquisition identifiers, demographic fields and image geometry. NIfTI does not necessarily contain complete sequence parameters; unavailable fields are not invented.

The download may include the entire T1 archive, but **processing is limited** to 50 subjects in debug.yaml and 150 in experiment.yaml. Selection balances available sites rather than reproducing population proportions. Set subject_limit to null for the complete dataset later.

## Run incrementally

Relative paths resolve against the YAML directory. Use a fresh output directory per experiment. Completed stages refuse to overwrite outputs.

```bash
brain-harmonization prepare --config configs/debug.yaml
brain-harmonization preprocess --config configs/debug.yaml
brain-harmonization qc --config configs/debug.yaml
brain-harmonization analyze --config configs/debug.yaml
```

Or run everything in a fresh directory:

```bash
brain-harmonization run --config configs/experiment.yaml
```

Equivalent entry points live in scripts/. run_site_experiment.py, run_harmonization.py and run_analysis.py are aliases for the **same joint analysis stage**: execute one, not all three. Harmonization is fitted inside the experiment after splitting, never to the entire cohort.

Inspect dataset_summary.json after preparation, tables/qc_initial.csv after preprocessing, and tables/qc.csv plus slice reports after QC. Missing files, metadata-only subjects, duplicate scans and failures remain in audit outputs. Unrecognized filenames are listed separately. Missing age prevents inclusion in the paired analysis but does not suppress image QC.

REVIEW subjects remain by default. A robust z cutoff of 3.5 is an exploratory convention, not a validated clinical threshold. Set qc.include_review to false in a sensitivity run. If exclusion leaves insufficient site counts, the analysis fails rather than regenerating splits. Each site needs >=3 training and >=1 validation/test subject; much larger numbers are needed for useful inference.

## Outputs

```text
results/<run>/
├── config.yaml, provenance.json, run.log
├── dataset_summary.json, qc_summary.json, qc_reference.json
├── feature_panel.json, combat_estimates.json, model_selection.json
├── age_benchmarks.json, completion.json, report.md
├── derivatives/   # resampled images and foreground masks
├── tables/        # manifest, QC, splits, features, predictions, metrics
├── figures/       # PNG and vector PDF
└── models/        # load only trusted pickle files
```

Configuration, seed, original metadata, input SHA-256 hashes, Git revision/dirty state, package versions, parameters and predictions make runs auditable. Preprocessing handles one scan at a time. Automatic per-subject resume is not implemented; repeat failed stages in a new run directory.

MRI files, data/, demo-data/, results/, virtual environments and archives are ignored by Git. Keep demographics in the documented data directory and inspect staged files before publishing.

## Docker

```bash
docker build -t brain-harmonization .
docker run --rm -v /absolute/ixi:/data:ro -v /absolute/configs:/config:ro -v /absolute/results:/outputs brain-harmonization run --config /config/container.yaml
```

Copy experiment.yaml to container.yaml and set dataset.root to /data/T1, dataset.metadata to /data/IXI.xls, and output to /outputs/run-001. Only package source and packaging files enter the image. Input data mount read-only. A running Docker engine is required.

## Scientific safeguards

- Unique subjects are split once. Duplicate images are excluded pending review.
- QC references, feature filtering, ComBat, scalers and PCA fit on training data only. Validation selects regularization.
- Age and sex are never model features. **Age is not passed to ComBat in the age-prediction experiment.**
- ComBat requires known site during application: this is a retrospective residual-site diagnostic, not blind site discovery or unseen-scanner generalization.
- Age/site confounding remains possible. Mean-age and site-mean benchmarks aid interpretation but do not resolve it.
- Normalization can remove meaningful contrast; ComBat can remove age variation correlated with site. Downstream evaluation exposes this tradeoff.
- Global, correlated image summaries only approximately satisfy ComBat assumptions and are weak biological proxies. Anatomical features and validated masks remain future work.
- Paired bootstrap intervals condition on fitted models and a single split. No automatic significance, equivalence or causal claims are made.
- neuroCombatFromTraining is marked “in development” upstream and emits a NumPy deprecation warning. Versions are pinned; tests verify training/apply equivalence and independence from other held-out subjects.

## References

- [IXI dataset](https://brain-development.org/ixi-dataset/).
- [Esteban et al. (2017), MRIQC](https://doi.org/10.1371/journal.pone.0184661). Motivates metrics plus visual review; this is not an MRIQC implementation or validated quality classifier.
- [Fortin et al. (2018), cortical thickness harmonization](https://doi.org/10.1016/j.neuroimage.2017.11.024). Motivates feature-level ComBat and biological preservation; does not validate our global-feature panel.
- [Nyul and Udupa (1999), intensity standardization](https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291522-2594%28199912%2942%3A6%3C1072%3A%3AAID-MRM11%3E3.0.CO%3B2-M). Our percentile scaling is a simpler baseline, not their landmark method.
- [neuroCombat implementation](https://github.com/Jfortin1/neuroCombat).

See [implementation plan](docs/implementation-plan.md), [methods](docs/methods.md), and [verification](docs/verification.md).
