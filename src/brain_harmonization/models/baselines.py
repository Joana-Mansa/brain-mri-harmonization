import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def split_subjects(
    table: pd.DataFrame, seed: int, test_fraction: float, validation_fraction: float
) -> pd.DataFrame:
    if table.subject_id.duplicated().any():
        raise ValueError("Repeated subject IDs would cause leakage")
    result = table[["subject_id", "site"]].copy()
    result["split"] = "train"
    rng = np.random.default_rng(seed)
    for site, group in result.groupby("site", sort=True):
        n = len(group)
        nt, nv = max(1, round(n * test_fraction)), max(1, round(n * validation_fraction))
        if n - nt - nv < 3:
            raise ValueError(
                f"Site {site} has {n} subjects; need >=3 training and >=1 per held-out split"
            )
        indices = rng.permutation(group.sort_values("subject_id").index)
        result.loc[indices[:nt], "split"] = "test"
        result.loc[indices[nt : nt + nv], "split"] = "validation"
    if result.site.nunique() < 2:
        raise ValueError("Site experiment requires multiple sites")
    return result


def fit_models(x: np.ndarray, table: pd.DataFrame, config: dict) -> tuple:
    train = (table.split == "train").to_numpy()
    valid = (table.split == "validation").to_numpy()
    if not train.any() or not valid.any():
        raise ValueError("Empty training or validation partition after QC")
    candidates = []
    for c in config["logistic_c"]:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000)
        )
        model.fit(x[train], table.site[train])
        score = balanced_accuracy_score(table.site[valid], model.predict(x[valid]))
        candidates.append((score, c, model))
    best_site = max(candidates, key=lambda value: value[0])
    candidates = []
    for alpha in config["ridge_alpha"]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(x[train], table.age[train])
        score = mean_absolute_error(table.age[valid], model.predict(x[valid]))
        candidates.append((score, alpha, model))
    best_age = min(candidates, key=lambda value: value[0])
    # Keep training-only fits; validation selects regularization, test remains untouched.
    return (
        best_site[2],
        best_age[2],
        {
            "logistic_c": best_site[1],
            "ridge_alpha": best_age[1],
            "validation_balanced_accuracy": best_site[0],
            "validation_mae": best_age[0],
        },
    )
