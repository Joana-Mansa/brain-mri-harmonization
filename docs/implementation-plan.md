# Implementation plan

1. Configuration, IXI metadata ingestion, deterministic site-balanced cohort manifest.
2. NIfTI inspection and conservative orientation/spacing preprocessing.
3. Interpretable QC metrics, robust training-relative outlier flags, slice reports.
4. Image-derived features and subject-level baseline site classification.
5. Image-level robust intensity normalization and feature-level ComBat.
6. Identical held-out site evaluation across methods.
7. Age regression without supplying age to harmonization.
8. Paired stratified bootstrap comparisons, tables, and figures.
9. Synthetic tests, CI, container, and reproducibility documentation.
10. Generated research report with explicit data provenance and limitations.

No real IXI results will be reported until actual images and demographics are supplied. Synthetic data verify software behavior only. Original inputs are immutable. Scanner metadata inferred from the IXI site description are labeled as such, not represented as per-scan measurements.
