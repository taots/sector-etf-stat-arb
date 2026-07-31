import pandas as pd
import numpy as np
from sector_stat_arb import data


def make_prices(days=100, tickers=("A","B","C")):
    idx = pd.bdate_range("2020-01-01", periods=days)
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        steps = rng.normal(loc=0.0002, scale=0.01, size=len(idx))
        prices[t] = 100 * np.exp(np.cumsum(steps))
    return prices


def test_validate_prices_ok():
    p = make_prices()
    data.validate_prices(p, required_tickers=list(p.columns))


def test_compute_returns():
    p = make_prices()
    r = data.compute_returns(p, method="log")
    assert r.shape == p.shape
    assert not r.isnull().all().all()
