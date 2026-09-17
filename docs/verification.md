# Verification record

Verified locally on 2026-09-17, Windows, Python 3.10, in a dedicated virtual environment.

- 17 tests passed, including synthetic end-to-end processing, QC, ComBat, prediction, bootstrap comparisons and report generation.
- Tests cover config paths and validation, IXI IDs, duplicate/missing records, NIfTI orientation and resampling, corrupt/nonfinite/constant/4D inputs, features, robust QC, subject split isolation, ComBat train/apply equivalence and independence from other test subjects, and zero paired differences for identical predictions.
- A separate 60-subject synthetic CLI run completed all stages and created the Markdown report, CSV tables, fitted objects, and PNG/PDF figures.
- The performance plot and a three-plane QC report were visually inspected. All numerical results belong to artificial phantoms; no empirical IXI claim is supported.
- Ruff lint and formatting checks pass. Dependency versions are recorded in requirements-lock.txt and each run's provenance.
- Remaining warnings originate in upstream neuroCombat/NumPy and matplotlib/pyparsing compatibility; no pipeline test failed.
- Docker is not installed on this host. The Dockerfile and mounts are supplied, but local container execution was not tested.
- Real IXI processing, manual anatomical QC, scanner metadata verification, and biological-preservation conclusions remain outstanding.

Reproduce using the README commands. The CI workflow validates a fresh installation using generated data and does not download MRI data.
