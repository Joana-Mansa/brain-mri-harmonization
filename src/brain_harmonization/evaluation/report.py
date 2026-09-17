from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import StandardScaler

from brain_harmonization.qc.report import save_figure


def markdown_table(table: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| "
            + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in row)
            + " |"
        )
    return "\n".join(lines)


def figures(
    out: Path,
    cohort: pd.DataFrame,
    qc: pd.DataFrame,
    representations: dict,
    predictions: dict,
    metrics: pd.DataFrame,
    synthetic: bool,
) -> None:
    suffix = " — SYNTHETIC SOFTWARE TEST" if synthetic else ""
    directory = out / "figures"
    fig, ax = plt.subplots(figsize=(6, 4))
    cohort.groupby(["site", "split"]).size().unstack(fill_value=0).plot.bar(ax=ax)
    ax.set(ylabel="Subjects", title="Analysis cohort by site and split" + suffix)
    fig.tight_layout()
    save_figure(fig, directory / "site_distribution")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, ["foreground_mean", "foreground_volume_ml", "entropy"]):
        sites = sorted(cohort.site.unique())
        vals = [qc.loc[qc.site == s, name].dropna() for s in sites]
        ax.boxplot(vals, tick_labels=sites)
        ax.set_title(name.replace("_", " "))
    fig.suptitle("QC distributions" + suffix)
    fig.tight_layout()
    save_figure(fig, directory / "qc_distributions")
    train = (cohort.split == "train").to_numpy()
    test = (cohort.split == "test").to_numpy()
    for method, x in representations.items():
        scaler = StandardScaler().fit(x[train])
        pca = PCA(n_components=2).fit(scaler.transform(x[train]))
        z = pca.transform(scaler.transform(x[test]))
        fig, ax = plt.subplots(figsize=(6, 5))
        for site in sorted(cohort.site.unique()):
            chosen = cohort.loc[test, "site"].to_numpy() == site
            ax.scatter(z[chosen, 0], z[chosen, 1], label=site, alpha=0.8)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"Held-out features: {method}" + suffix)
        ax.legend()
        save_figure(fig, directory / f"pca_{method}")
        p = predictions[method]
        cm = confusion_matrix(cohort.site[test], p["site"], labels=p["classes"])
        pd.DataFrame(cm, index=p["classes"], columns=p["classes"]).to_csv(
            out / "tables" / f"confusion_{method}.csv"
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm, display_labels=p["classes"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Site classification: {method}" + suffix)
        save_figure(fig, directory / f"confusion_{method}")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["balanced_accuracy", "macro_f1", "age_mae"]):
        sub = metrics[metrics.metric == metric]
        positions = np.arange(len(sub))
        ax.scatter(positions, sub.value, color="#176b87", zorder=3)
        ax.vlines(positions, sub.ci_low, sub.ci_high, color="#176b87")
        ax.set_xticks(positions, sub.method, rotation=25, ha="right")
        ax.set_title(metric.replace("_", " "))
        if metric == "balanced_accuracy":
            ax.axhline(1 / cohort.site.nunique(), linestyle="--", color="gray", label="1 / sites")
            ax.legend()
    fig.suptitle("Held-out estimates and 95% bootstrap intervals" + suffix)
    fig.tight_layout()
    save_figure(fig, directory / "performance")
    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4), squeeze=False)
    age = cohort.loc[test, "age"].to_numpy()
    for ax, (method, p) in zip(axes[0], predictions.items()):
        for site in sorted(cohort.site.unique()):
            selected = cohort.loc[test, "site"].to_numpy() == site
            ax.scatter(age[selected], p["age"][selected], label=site)
        bounds = [min(age.min(), p["age"].min()), max(age.max(), p["age"].max())]
        ax.plot(bounds, bounds, "--", color="gray")
        ax.set(xlabel="Actual age (years)", ylabel="Predicted age (years)", title=method)
        ax.legend()
    fig.suptitle("Age prediction" + suffix)
    fig.tight_layout()
    save_figure(fig, directory / "age_prediction")


def research_report(
    out: Path,
    config: dict,
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    differences: pd.DataFrame,
    benchmark: dict,
) -> None:
    synthetic = config["dataset"]["kind"] == "synthetic"
    warning = (
        "**SYNTHETIC SOFTWARE VALIDATION ONLY. These images are artificial phantoms, not IXI data. "
        "No empirical claim about human MRI or biological preservation can be made.**"
        if synthetic
        else "Exploratory IXI analysis. Automated QC and selected images require human review."
    )
    sections = [
        "# Quality Control and Harmonization of Multi-Centre Brain MRI",
        warning,
        "## Background\nMRI acquisition differences can affect quantitative image summaries. Harmonization must be evaluated jointly for residual site information and useful subject variation.",
        "## Research question\nCan harmonization reduce acquisition-site information while preserving age-related information?",
        f"## Dataset\nSource: {config['dataset']['kind']}. Analysis cohort: {len(cohort)} subjects; {sum(cohort.split == 'test')} held out for testing. Cohort selection balances available sites. See manifest.csv for all available and missing records, QC CSV for exclusions, and splits.csv for subject assignments.",
        markdown_table(
            cohort.groupby("site")
            .agg(
                n=("subject_id", "size"),
                age_mean=("age", "mean"),
                age_min=("age", "min"),
                age_max=("age", "max"),
            )
            .reset_index()
        ),
        "## Methods\nT1 volumes are validated, reoriented to RAS and resampled using the configured spacing and interpolation. A largest-component foreground mask is used, not a brain mask. No registration, bias correction, or tissue segmentation is performed. Global intensity percentiles, moments and gradient summaries do not require anatomical correspondence. This conservative baseline cannot establish preservation of regional anatomy.\n\nQC flags use training-only median/MAD references (IQR fallback) and retain REVIEW subjects by default. The same subject splits and image feature panel are used for all methods. Baseline has no intensity scaling; robust_intensity uses per-image percentile scaling; ComBat operates on baseline features, fitted only to training subjects. Constant-within-site features are removed using training data. ComBat receives known site at application and no biological covariates.\n\nBalanced logistic regression and ridge regression select regularization on validation data; their fits remain training-only. Age is never supplied to harmonization. Test predictions are evaluated once. Site-stratified paired bootstrap intervals condition on the fitted models and do not measure training-cohort uncertainty. Differences are post minus baseline; no p-values or significance claims are made.",
        "## Results",
        markdown_table(metrics),
        "### Paired differences",
        markdown_table(differences),
        f"Training-mean age baseline test MAE: {benchmark['mean_age_mae']:.3f} years. Training site-mean age baseline test MAE: {benchmark['site_mean_age_mae']:.3f} years.",
        "![Performance](figures/performance.png)\n\n![Age](figures/age_prediction.png)\n\n![Cohort](figures/site_distribution.png)\n\n![QC](figures/qc_distributions.png)",
        "## Discussion\nLower residual site accuracy is compatible with reduced acquisition information but does not establish its removal. ComBat explicitly uses site labels during transformation; this diagnostic is not blind scanner prediction. Lower age MAE suggests better age recoverability only if it improves on naive and site-mean benchmarks. Age/site imbalance may let models recover age through acquisition effects. Conversely, removing site effects can erase age variation when those variables are confounded. Age MAE intervals overlapping zero for method differences do not prove biological equivalence. A prospective noninferiority margin, more informative anatomical features, and external validation are needed.\n\nThe feature panel contains correlated, differently shaped distributions; ComBat's location/scale and empirical-Bayes assumptions are exploratory here. Inspect its fitted effects and residual distributions before drawing conclusions. Global image summaries are weak proxies for biology. Foreground segmentation may retain skull or omit dark tissue; its volume is not brain volume. SNR is a contrast/noise proxy, not physical MRI SNR. PCA views use training-only fits and separate axes for each method; visual mixing is not inferential evidence. Small held-out cohorts give unstable intervals. Scanner, field strength and site are confounded in IXI. No unseen-site claim is supported.",
        "## Future work\nAfter manual QC and real IXI validation, add validated skull stripping, anatomical registration, tissue volumes and age/site-matched sensitivity cohorts. Consider ComBat with protected age only in a separate association analysis, never as evidence of held-out age prediction. Extend to ABIDE, ADNI, longitudinal and multimodal imaging before considering disease prediction, lesion segmentation, or deep-learning harmonization.",
        "## References\n- IXI dataset: https://brain-development.org/ixi-dataset/\n- Esteban et al. (2017), MRIQC, PLOS ONE, doi:10.1371/journal.pone.0184661.\n- Fortin et al. (2018), Harmonization of cortical thickness measurements across scanners and sites, NeuroImage, doi:10.1016/j.neuroimage.2017.11.024.\n- Nyul and Udupa (1999), On standardizing the MR image intensity scale, Magnetic Resonance in Medicine, doi:10.1002/(SICI)1522-2594(199912)42:6<1072::AID-MRM11>3.0.CO;2-M. Our percentile scaling is a simpler baseline, not an implementation of their landmark method.\n- ComBat implementation: https://github.com/Jfortin1/neuroCombat",
    ]
    for method in config["experiment"]["methods"]:
        sections.append(
            f"### {method} diagnostics\n![Confusion](figures/confusion_{method}.png)\n\n![PCA](figures/pca_{method}.png)"
        )
    examples = sorted((out / "figures").glob("qc_IXI*.png"))
    if examples:
        sections.append(f"### Example visual QC\n![Visual QC](figures/{examples[0].name})")
    (out / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
