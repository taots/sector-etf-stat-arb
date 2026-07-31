# Leakage-Aware Sector ETF Statistical Arbitrage

A reproducible Python research pipeline for a leakage-aware sector ETF statistical arbitrage signal.

This project demonstrates a closed-form research workflow with:

- trailing PCA residual signal construction;
- volatility-scaled returns;
- weekly rebalance with execution lag;
- dollar- and beta-neutral portfolio construction;
- cost-aware backtesting with turnover and short borrow cost;
- automated tests and demo figure generation.

## Disclaimer

⚠️ This repository and its content are for **educational and research purposes only**.

- Nothing contained in this project constitutes financial, investment, legal, or tax advice.
- Do not use this software or its outputs to make real-world trading or investment decisions.
- The author(s) accept no responsibility or liability for any financial losses, trading damages, or risks incurred by using this code.

> Research and educational use only. Not investment advice.

## Demo

Run the demo to generate example output figures in `reports/figures`:

```bash
python -m sector_stat_arb.cli demo --output reports/figures
```

The demo produces:

- `reports/figures/equity_curves.png`
- `reports/figures/drawdown.png`
- `reports/figures/turnover.png`

## Install

```bash
python -m pip install -r requirements.txt
```

## Test

```bash
pytest -q
```

## Project structure

- `src/sector_stat_arb/`
  - `data.py`: price loading, validation, and returns
  - `signal.py`: trailing PCA residual signal implementation
  - `portfolio.py`: neutralized weight construction
  - `backtest.py`: execution lag, turnover, and cost accounting
  - `metrics.py`: performance metrics
  - `demo.py`: synthetic example runner with plot generation
- `config/`: configuration templates
- `tests/`: automated tests
- `reports/figures/`: demo output charts

## Notes

The current demo uses synthetic price series for reproducibility. The repository is structured so real ETF data can be loaded via `src/sector_stat_arb/data.py` and evaluated through the same signal/backtest pipeline.
