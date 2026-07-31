"""Signal calculation: trailing volatility, rolling PCA residuals, and z-score signal."""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_signal(prices: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute mean-reversion signals for each asset.

    prices: DataFrame of adjusted close prices (columns are tickers)
    config: configuration dictionary with keys described in project spec

    Returns a DataFrame `signals` indexed by date with columns for each asset.
    """
    vol_window = config.get("volatility_window", 60)
    pca_window = config.get("pca_window", 252)
    n_components = config.get("n_components", 3)
    residual_lookback = config.get("residual_lookback", 20)
    signal_clip = config.get("signal_clip", 3.0)

    returns = np.log(prices / prices.shift(1))

    # trailing volatility (sample std over trailing window)
    trailing_vol = returns.rolling(vol_window, min_periods=int(vol_window/2)).std()
    scaled_returns = returns.divide(trailing_vol)

    assets = [c for c in prices.columns if c != config.get("benchmark")]

    residuals = pd.DataFrame(index=prices.index, columns=assets, dtype=float)

    for i in range(len(prices)):
        if i < pca_window - 1:
            continue
        window_idx = prices.index[i - pca_window + 1 : i + 1]
        Rwin = scaled_returns.loc[window_idx, assets].dropna(axis=1, how="any")
        if Rwin.shape[0] < pca_window or Rwin.shape[1] < 1:
            continue
        # standardize features (assets) across time
        scaler = StandardScaler(with_mean=True, with_std=True)
        try:
            Rstd = scaler.fit_transform(Rwin.values)
        except Exception:
            continue
        # fit PCA on standardized returns
        pca = PCA(n_components=min(n_components, Rstd.shape[1]))
        scores = pca.fit_transform(Rstd)
        recon = pca.inverse_transform(scores)
        # inverse transform to original scaled-returns space
        recon_orig = scaler.inverse_transform(recon)
        # the reconstructed common component for the last row
        recon_last = recon_orig[-1, :]
        # residual = original last scaled return - reconstructed last
        orig_last = Rwin.values[-1, :]
        res_last = orig_last - recon_last
        residuals.loc[prices.index[i], Rwin.columns] = res_last

    # aggregate residuals over lookback
    residual_score = residuals.rolling(residual_lookback, min_periods=1).sum()

    # cross-sectional zscore per date
    signals = (-(residual_score.sub(residual_score.mean(axis=1), axis=0)
                 .div(residual_score.std(axis=1), axis=0)))
    signals = signals.clip(-signal_clip, signal_clip)
    return signals

