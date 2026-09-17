import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def scores(site, age, predicted_site, predicted_age, probabilities=None, classes=None) -> dict:
    values = {
        "balanced_accuracy": balanced_accuracy_score(site, predicted_site),
        "macro_f1": f1_score(site, predicted_site, average="macro", zero_division=0),
        "age_mae": mean_absolute_error(age, predicted_age),
        "age_rmse": float(np.sqrt(mean_squared_error(age, predicted_age))),
        "age_r2": r2_score(age, predicted_age) if len(age) > 1 and np.var(age) > 0 else np.nan,
    }
    if probabilities is not None:
        if len(classes) == 2:
            values["roc_auc"] = roc_auc_score(np.asarray(site) == classes[1], probabilities[:, 1])
        else:
            values["roc_auc"] = roc_auc_score(
                site, probabilities, multi_class="ovr", average="macro", labels=classes
            )
    return values


def paired_bootstrap(
    truth: pd.DataFrame, predictions: dict, repetitions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Percentile intervals conditional on fitted models; identical resamples for methods.

    Site-stratified resampling preserves all classes and subjects remain the unit.
    Differences are post minus baseline, not formal hypothesis-test p-values.
    """
    sites, ages = truth.site.to_numpy(), truth.age.to_numpy(float)
    groups = [np.flatnonzero(sites == site) for site in np.unique(sites)]
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in predictions}
    points = {}
    for method, p in predictions.items():
        points[method] = scores(sites, ages, p["site"], p["age"], p["proba"], p["classes"])
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for method, p in predictions.items():
            draws[method].append(
                scores(
                    sites[idx],
                    ages[idx],
                    p["site"][idx],
                    p["age"][idx],
                    p["proba"][idx],
                    p["classes"],
                )
            )
    metric_rows, differences = [], []
    for method in predictions:
        for metric, point in points[method].items():
            vals = np.array([r[metric] for r in draws[method]])
            finite = vals[np.isfinite(vals)]
            lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            metric_rows.append(
                dict(
                    method=method,
                    metric=metric,
                    value=point,
                    ci_low=lo,
                    ci_high=hi,
                    valid_bootstraps=len(finite),
                )
            )
            if method != "baseline":
                delta = vals - np.array([r[metric] for r in draws["baseline"]])
                finite = delta[np.isfinite(delta)]
                lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
                differences.append(
                    dict(
                        method=method,
                        metric=metric,
                        delta=point - points["baseline"][metric],
                        ci_low=lo,
                        ci_high=hi,
                        valid_bootstraps=len(finite),
                    )
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(
        differences, columns=["method", "metric", "delta", "ci_low", "ci_high", "valid_bootstraps"]
    )
