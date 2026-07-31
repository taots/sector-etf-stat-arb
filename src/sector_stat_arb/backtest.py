"""Simple backtest engine implementing lagged execution, turnover, costs, and borrow costs."""

from typing import Dict
import pandas as pd
import numpy as np


def run_backtest(weights: pd.DataFrame, prices: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Run a simple daily backtest.

    weights: DataFrame indexed by date with columns for assets (target weights at each rebalance)
    prices: price DataFrame with same columns
    Returns a DataFrame with gross and net returns and some diagnostics.
    """
    execution_lag = config.get("execution_lag_days", 1)
    tx_bps = config.get("transaction_cost_bps", 5)
    annual_borrow_bps = config.get("annual_short_borrow_bps", 200)

    returns = np.log(prices / prices.shift(1))

    # align weights and returns
    common_idx = weights.index.intersection(returns.index)
    w = weights.reindex(common_idx).fillna(0.0)
    r = returns.reindex(common_idx).fillna(0.0)

    # apply execution lag: positions are applied after lag days
    w_executed = w.shift(execution_lag).fillna(0.0)

    gross_ret = (w_executed * r).sum(axis=1)

    turnover = (w_executed - w_executed.shift(1)).abs().sum(axis=1).fillna(0.0)
    tx_cost = turnover * (tx_bps / 10000.0)

    short_gross = w_executed.clip(upper=0.0).abs().sum(axis=1)
    borrow_cost = short_gross * (annual_borrow_bps / 10000.0) / 252.0

    net_ret = gross_ret - tx_cost - borrow_cost

    df = pd.DataFrame(index=common_idx)
    df["gross_return"] = gross_ret
    df["transaction_cost"] = tx_cost
    df["borrow_cost"] = borrow_cost
    df["net_return"] = net_ret
    df["turnover"] = turnover
    df["gross_equity"] = (1 + df["gross_return"]).cumprod()
    df["net_equity"] = (1 + df["net_return"]).cumprod()

    return df

