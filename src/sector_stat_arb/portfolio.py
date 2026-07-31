"""Portfolio construction: raw weights, dollar & beta neutralization, caps, and gross exposure normalization."""

from typing import Dict
import numpy as np
import pandas as pd


def _project_neutral(weights: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project `weights` to be orthogonal to columns of A (least-squares projection)."""
    # w_neutral = w - A @ pinv(A'A) @ A' @ w
    pinv = np.linalg.pinv(A.T @ A)
    correction = A @ (pinv @ (A.T @ weights))
    return weights - correction


def build_portfolio(signals: pd.DataFrame, betas: pd.Series, config: Dict) -> pd.DataFrame:
    """Build portfolio weights from signals and trailing betas.

    signals: DataFrame indexed by date with columns for assets
    betas: DataFrame or Series indexed by date (if Series, assumed constant) matching assets
    Returns DataFrame of weights indexed by date.
    """
    max_abs = config.get("max_abs_weight", 0.25)
    gross_target = config.get("gross_exposure", 1.0)

    assets = signals.columns.tolist()
    dates = signals.index
    weights = pd.DataFrame(0.0, index=dates, columns=assets)

    # ensure betas is a DataFrame
    if isinstance(betas, pd.Series):
        betas = pd.DataFrame([betas.values] * len(dates), index=dates, columns=assets)
    elif isinstance(betas, pd.DataFrame):
        betas = betas.reindex(index=dates, columns=assets).fillna(method='ffill').fillna(0.0)
    else:
        raise ValueError("betas must be Series or DataFrame")

    for date in dates:
        s = signals.loc[date].fillna(0.0).values
        if np.allclose(s, 0.0):
            continue
        # raw weights proportional to signal
        w_raw = s.copy()
        # projection matrix A: ones and beta vector
        beta_vec = betas.loc[date].values
        A = np.vstack([np.ones(len(assets)), beta_vec]).T  # shape (n_assets, 2)
        w_neu = _project_neutral(w_raw, A)
        # clip
        w_clipped = np.clip(w_neu, -max_abs, max_abs)
        # normalize gross exposure
        g = np.sum(np.abs(w_clipped))
        if g == 0:
            weights.loc[date] = 0.0
        else:
            weights.loc[date] = (w_clipped / g) * gross_target

    return weights

