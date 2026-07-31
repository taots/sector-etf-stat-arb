import pandas as pd
import numpy as np

from sector_stat_arb import data, signal, portfolio, backtest, metrics


def make_prices(days=800, tickers=("X1","X2","X3","SPY")):
    idx = pd.bdate_range("2005-01-01", periods=days)
    rng = np.random.default_rng(123)
    prices = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        steps = rng.normal(loc=0.0001, scale=0.01, size=len(idx))
        prices[t] = 100 * np.exp(np.cumsum(steps))
    return prices


def test_end_to_end_signal_and_backtest():
    cfg = {
        "volatility_window": 60,
        "pca_window": 252,
        "n_components": 1,
        "residual_lookback": 20,
        "signal_clip": 3.0,
        "benchmark": "SPY",
        "max_abs_weight": 0.25,
        "gross_exposure": 1.0,
        "execution_lag_days": 1,
        "transaction_cost_bps": 5,
        "annual_short_borrow_bps": 200,
    }

    prices = make_prices()
    # validate
    data.validate_prices(prices, required_tickers=list(prices.columns))
    signals = signal.compute_signal(prices, cfg)
    # extract betas as simple zeros for test
    assets = [c for c in prices.columns if c != cfg["benchmark"]]
    betas = pd.Series({a: 1.0 for a in assets})
    weights = portfolio.build_portfolio(signals, betas, cfg)
    bt = backtest.run_backtest(weights, prices[assets], cfg)
    m = metrics.compute_metrics(bt["net_return"]) if "net_return" in bt else {}

    assert signals.shape[0] > 0
    assert weights.shape == signals.shape
    assert "net_return" in bt.columns
    assert isinstance(m, dict)
