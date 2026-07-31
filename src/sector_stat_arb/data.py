"""Data download and validation utilities.

Lightweight implementation using `yfinance` when available. Includes
basic validation checks required by the project spec.
"""

from typing import List
import pandas as pd
import numpy as np


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices for `tickers` using yfinance.

    Returns a DataFrame indexed by date with columns for each ticker.
    Raises ImportError if yfinance is not installed.
    """
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError("yfinance is required to download market data; install it or provide prices manually") from e

    data = yf.download(tickers, start=start, end=end, progress=False, threads=True, group_by='ticker')
    # yfinance returns different shapes when single vs multiple tickers
    if isinstance(tickers, (list, tuple)) and len(tickers) > 1:
        # try to extract Adjusted Close columns
        if ("Adj Close" in data.columns) or ("Adj Close" in data):
            adj = data["Adj Close"]
        else:
            # some yfinance versions return a nested dict-like frame
            cols = [col for col in data.columns.levels[0]]
            if set(tickers).issubset(cols):
                adj = pd.DataFrame({t: data[t]["Adj Close"] for t in tickers})
            else:
                adj = pd.DataFrame(data)
    else:
        # single ticker
        if "Adj Close" in data.columns:
            adj = data["Adj Close"].to_frame()
            adj.columns = tickers
        else:
            adj = pd.DataFrame(data)

    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index()
    return adj


def validate_prices(df: pd.DataFrame, required_tickers: List[str] = None) -> None:
    """Run basic data quality checks on price DataFrame.

    Raises ValueError on validation failure.
    Checks:
    - index is monotonic increasing and of type DatetimeIndex
    - no duplicate dates
    - all required tickers present
    - no non-positive prices
    - no forward-filled (constant) columns entirely
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Price index must be a DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Price index must be sorted in increasing order")

    if df.index.has_duplicates:
        raise ValueError("Duplicate dates found in price data")

    if required_tickers is not None:
        missing = set(required_tickers) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required tickers: {sorted(list(missing))}")

    if (df <= 0).any().any():
        raise ValueError("Non-positive prices found in data")

    # detect columns identical across time (likely placeholder/fill)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const_cols:
        raise ValueError(f"Columns with constant prices (possible bad data): {const_cols}")


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """Compute daily returns from price series.

    method: 'log' or 'pct'
    """
    if method == "log":
        return np.log(prices / prices.shift(1))
    elif method == "pct":
        return prices.pct_change()
    else:
        raise ValueError("Unknown return method")

