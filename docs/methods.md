# Methodological decisions

## Preprocessing and feature scope

RAS orientation makes slice displays and array-axis summaries consistent without losing physical coordinates. A common 2 mm spacing controls voxel-scale differences and reduces cost. Linear interpolation changes intensity/smoothness, so baseline means *preprocessed, without intensity harmonization*, not untouched raw MRI. Original geometry and intensity statistics are retained.

The foreground mask uses the largest component above the 20th percentile of positive intensities, then fills holes. Twenty is an exposed segmentation parameter, not a validated quality cutoff. This is not skull stripping or a brain-volume measurement. Unknown spatial units trigger REVIEW with an explicit millimetre assumption; other non-mm units require conversion.

N4 correction, validated skull stripping, tissue segmentation and template registration are deferred until anatomical features justify them. No registration score is invented. Constant, nonfinite, corrupt or non-3D scans are excluded with reasons.

The feature panel has 19 intensity percentiles, mean, standard deviation, skewness, kurtosis and six gradient summaries. No header, filename, demographics, site or QC label enters a model. This limits inference to residual site information in these 29 global image features.

## Harmonization

The image-level arm clips/scales each image using foreground 1st/99th percentiles and re-extracts features using the unchanged mask. It fits no cohort histogram and does not implement Nyul landmark standardization.

ComBat independently operates on baseline features. Training-only within-site near-constant filtering is intersected across baseline/normalized panels so comparisons use identical columns. Training scaling precedes parametric empirical Bayes ComBat. Its fitted location/scale effects apply to known-site held-out scans without recomputing moments. No age or sex covariates are provided. Finite outputs and training/application equivalence are checked.

Correlated, non-Gaussian features challenge ComBat assumptions. This is an exploratory feature panel, not evidence that the method is appropriate for every summary. Inspect fitted effects and residual distributions before scientific claims. Unseen sites are rejected.

## QC and cohort selection

Inventory covers all metadata/image records, but NIfTI inspection processes only the selected cohort. Unselected records are not declared image-usable. Raw per-subject QC is computed before splitting; pooled statistical references fit only to age-complete training subjects.

Scale is 1.4826 times MAD, falling back to IQR/1.349. Robust z >3.5 is a review flag. Deviations from a constant reference are flagged separately. Pooled references may flag precisely the site shifts being studied; REVIEW remains included by default. PASS means no implemented check raised a flag, not diagnostic quality.

Foreground/background contrast and foreground mean divided by nonforeground deviation are descriptive proxies. Nonforeground may contain tissue and is not a pure noise region. Zero deviation yields missing SNR, not infinity. Foreground volume is not brain volume.

Site and age experiments use the same age-complete subjects. Missing-age images still receive QC. Site-stratified split counts are rounded separately and may differ from requested global fractions. Splits are never regenerated after QC exclusion. Age/sex matching is not implemented.

## Statistics and interpretation

Validation selects logistic C and ridge alpha; fits remain training-only. Test probabilities, predictions and model choices are saved. Balanced accuracy/macro F1 address imbalance; multiclass AUC uses macro one-vs-rest. Age errors use years, with training mean and training site-mean age benchmarks.

Paired site-stratified subject bootstrap gives 95% percentile intervals conditional on fitted models and a single split. It omits training and selection uncertainty. Differences are post minus baseline; negative site accuracy and positive age MAE show a potential tradeoff. Valid resample counts are retained where R² is undefined. No p-value, multiple-comparison, noninferiority or causal claim is implied.

PCA fits training data only and transforms held-out samples. Each method has separate axes; visual mixing is not evidence of successful harmonization.

Known site is supplied to ComBat at application, so its downstream classifier measures residual separability after a site-dependent transform. This is not blind site classification. Age/site confounding can preserve or erase spurious age signals. A stronger study needs overlap assessment, matched sensitivity cohorts, per-site errors, anatomical features, and a predefined biological noninferiority margin. Age-protected ComBat belongs in a separate known-age association study, not age-prediction validation.

## Literature

Official IXI documentation supplies site descriptions. Esteban et al. (2017) motivates interpretability plus visual QC. Fortin et al. (2018) motivates feature-level ComBat and biological preservation. Nyul and Udupa (1999) supplies context for MRI intensity scales. Upstream neuroCombat train/apply source was inspected before implementation. Full references and distinctions are in README.md.
