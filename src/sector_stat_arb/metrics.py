"""Basic performance metrics for strategy returns."""

import pandas as pd
import numpy as np


def compute_metrics(returns: pd.Series) -> dict:
    """Compute simple annualized metrics from daily return series."""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    mean_daily = r.mean()
    vol_daily = r.std()
    ann_ret = mean_daily * 252
    ann_vol = vol_daily * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }

