import numpy as np
import pandas as pd
from neuroCombat import neuroCombat, neuroCombatFromTraining
from sklearn.preprocessing import StandardScaler


def eligible_features(x: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Remove features constant within any training site before all comparisons."""
    keep = np.isfinite(x).all(axis=0) & (np.std(x, axis=0) > 1e-8)
    for site in np.unique(sites):
        keep &= np.std(x[sites == site], axis=0) > 1e-8
    if keep.sum() < 3:
        raise ValueError("ComBat requires at least three nondegenerate features")
    return keep


class Combat:
    """Parametric empirical Bayes ComBat on continuous image features.

    Uses the maintained implementation's training/apply interface. No age or sex
    covariate is supplied: age is an evaluation target. Known site is required at
    inference. This is a retrospective residual-site diagnostic, not blind site
    discovery or unseen-scanner generalization.
    """

    def fit(self, x: np.ndarray, sites: np.ndarray) -> "Combat":
        sites = np.asarray(sites, dtype=str)
        if len(np.unique(sites)) < 2 or min(np.unique(sites, return_counts=True)[1]) < 3:
            raise ValueError(
                "ComBat requires at least two sites and three training subjects per site"
            )
        if not eligible_features(x, sites).all():
            raise ValueError("Degenerate features must be filtered using training data")
        self.scaler = StandardScaler().fit(x)
        standardized = self.scaler.transform(x)
        result = neuroCombat(
            dat=standardized.T,
            covars=pd.DataFrame({"site": sites}),
            batch_col="site",
            eb=True,
            parametric=True,
        )
        if not np.isfinite(result["data"]).all():
            raise ValueError("ComBat returned nonfinite training estimates")
        self.estimates = result["estimates"]
        self.training_result = result["data"].T
        return self

    def transform(self, x: np.ndarray, sites: np.ndarray) -> np.ndarray:
        result = neuroCombatFromTraining(
            dat=self.scaler.transform(x).T,
            batch=np.asarray(sites, dtype=str),
            estimates=self.estimates,
        )["data"].T
        if not np.isfinite(result).all():
            raise ValueError("ComBat returned nonfinite transformed features")
        return result
