"""Demo runner to produce example backtest figures using synthetic data."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import signal as signal_mod
from . import portfolio as portfolio_mod
from . import backtest as backtest_mod


def make_prices(days=1000, tickers=("XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY","SPY")):
    idx = pd.bdate_range("2005-01-01", periods=days)
    rng = np.random.default_rng(2026)
    prices = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        steps = rng.normal(loc=0.0001, scale=0.01, size=len(idx))
        prices[t] = 100 * np.exp(np.cumsum(steps))
    return prices


def run_demo(output_dir: str = "reports/figures") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = {
        "volatility_window": 60,
        "pca_window": 252,
        "n_components": 3,
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
    # compute signals
    signals = signal_mod.compute_signal(prices, cfg)
    assets = [c for c in prices.columns if c != cfg["benchmark"]]
    # simple betas (ones) for demo
    betas = pd.Series({a: 1.0 for a in assets})
    weights = portfolio_mod.build_portfolio(signals, betas, cfg)
    bt = backtest_mod.run_backtest(weights, prices[assets], cfg)

    # plot net vs gross equity
    plt.figure(figsize=(10, 6))
    plt.plot(bt.index, bt["net_equity"], label="Net Equity")
    plt.plot(bt.index, bt["gross_equity"], label="Gross Equity")
    plt.legend()
    plt.title("Demo Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig(out / "equity_curves.png")
    plt.close()

    # drawdown
    cum = bt["net_equity"]
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    plt.figure(figsize=(10, 4))
    plt.plot(bt.index, drawdown, color="red")
    plt.fill_between(bt.index, drawdown, 0, color="red", alpha=0.2)
    plt.title("Net Equity Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out / "drawdown.png")
    plt.close()

    # turnover
    plt.figure(figsize=(10, 4))
    plt.plot(bt.index, bt["turnover"].fillna(0))
    plt.title("Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.tight_layout()
    plt.savefig(out / "turnover.png")
    plt.close()

    print(f"Demo figures written to {out}")


if __name__ == "__main__":
    run_demo()
