# RL Portfolio Optimization Framework

A deep reinforcement learning framework for portfolio optimization using PPO (Proximal Policy Optimization) with PyTorch. The agent learns to allocate capital across a dynamic stock universe (S&P 500, ~500 stocks) by observing market features and current holdings, then rebalancing at configurable intervals.

## Architecture

![Framework architecture](rl_portfolio_architecture.svg)

The framework follows a standard actor-critic loop. Price data flows through a feature engine that computes technical indicators and return statistics. The **portfolio environment** packages these into a state (per-stock features, market-level features, and current weights) and passes it to two separate networks:

- **Actor** — a policy network with cross-asset multi-head attention that outputs a Gaussian distribution over portfolio weights. Attention lets the model learn inter-stock dependencies rather than treating each stock independently.
- **Critic** — a value network (same attention architecture, separate parameters) that estimates V(s) for advantage computation.

Raw actions are mapped to valid weights through an **action processing** layer: softmax for long-only portfolios (weights in [0,1] summing to 1) or tanh with mean-centering for long-short (weights in [-1,1] summing to 0). A PPO update with clipped objective and Generalized Advantage Estimation keeps training stable.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Typed dataclasses for all hyperparameters |
| `features.py` | Technical indicators and rolling normalization |
| `environment.py` | Gym-like portfolio environment with transaction costs |
| `networks.py` | Actor-critic networks with cross-asset attention |
| `algo.py` | PPO agent with GAE and rollout buffer |
| `training.py` | Training loop with evaluation and checkpointing |
| `forwardbacktest.py` | Walk-forward optimization engine |
| `universe.py` | Dynamic stock universe handling (IPOs and delistings) |
| `fin_data.py` | S&P 500 constituent and price data download |
| `utils.py` | Performance metrics, Plotly dashboard, and evaluation utilities |
| `utils_html.py` | Chart.js fallback dashboard (no Plotly dependency) |
| `main_backtest.py` | Example: full walk-forward backtest with dashboard output |

## Quickstart

```bash
pip install torch pandas numpy plotly yfinance requests_html
```

**Walk-forward optimization** (recommended for production backtesting):

```python
import pandas as pd
from rl_agent.fin_data import download_fin_data, get_sp500
from rl_agent.forwardbacktest import WalkForwardBacktestEngine
from rl_agent.utils import compute_portfolio_metrics, generate_dashboard

# Load S&P 500 price data (or supply your own DataFrame)
ticker, sp500 = get_sp500()
_, _, prices = download_fin_data(ticker=ticker, sp500=sp500)

# Run walk-forward optimization
engine = WalkForwardBacktestEngine(prices)
results = engine.run()

# Generate interactive dashboard
generate_dashboard(
    rl_results=results["oos_returns"],
    bm_daily_returns=bm_returns,          # equal-weight benchmark returns
    rl_dates=oos_dates,
    periods_per_year=12,
    output_path="dashboard.html",
)
```

Or run the included example directly:

```bash
python rl_agent/main_backtest.py
```

## Configuration

All settings live in typed dataclasses. The most commonly adjusted classes:

```python
from rl_agent.config import (
    EnvironmentConfig, BacktestConfig, TrainingConfig, FeatureConfig
)
from rl_agent.forwardbacktest import WalkForwardBacktestEngine

engine = WalkForwardBacktestEngine(
    prices,
    wf_config=BacktestConfig(
        train_window_years=5.0,      # Years of data per training window
        step_size=1,                 # Roll forward by N months each step
        episodes_per_window=200,     # Training episodes per window
        warmstart=True,              # Fine-tune from the previous window's model
        warmstart_lr_factor=0.3,     # Learning rate multiplier when warm-starting
        patience=30,                 # Early-stopping patience (episodes)
        min_episodes=50,
    ),
)
```

Key `EnvironmentConfig` options:

```python
EnvironmentConfig(
    mode="long_only",             # or "long_short"
    rebalance_freq="monthly",     # or "weekly"
    reward_type="return",         # "return" (default), "sharpe", "mse", "combined"
    transaction_cost_bps=10,      # One-way transaction cost in basis points
    slippage_bps=5,               # Slippage estimate in basis points
    max_position_size=0.10,       # Maximum weight per stock
    turnover_threshold=0.20,      # Turnover below this level is not penalized
    drawdown_penalty=0.5,         # Used by the "combined" reward type only
    turnover_penalty=0.1,         # Used by the "combined" reward type only
)
```

## Features

The feature engine computes per-stock and market-level indicators from raw prices. All features use rolling z-score normalization with a 252-day window to prevent look-ahead bias and are winsorized to [−5, 5].

**Per-stock features:**

| Feature | Windows / parameters |
|---|---|
| Rolling returns | 21, 63, 128, 252 days |
| Rolling volatility (annualized) | 21, 63, 128, 252 days |
| PPO (Percentage Price Oscillator) | fast=12, slow=26, signal=9 |
| RSI | 14 days, normalized to [−1, 1] |
| Bollinger Band position | 21-day window, 2σ bands |
| Cross-sectional return rank | 21, 63, 128, 252 days (percentile, if enabled) |

**Market-level features:**

| Feature | Windows |
|---|---|
| Equal-weight market return | 21, 252 days |
| Equal-weight market volatility | 21, 252 days |
| Cross-sectional return dispersion | 21, 252 days |

The PPO (Percentage Price Oscillator) expresses the MACD oscillator as a percentage of the slow EMA, making it scale-invariant across stocks at any price level — a $10 stock and a $200 stock produce directly comparable values.

## Reward Options

| Type | Description |
|---|---|
| **return** (default) | `log(1 + net_return)` per rebalancing period. Simple, unbiased, and free of the estimation noise that affects Sharpe from short windows. |
| **sharpe** | Annualized Sharpe ratio computed from all daily returns accumulated during the episode (not just the current holding period), giving a stable estimate that improves as the episode progresses. |
| **mse** | Negative squared deviation from a target return series (e.g., an index). |
| **combined** | Cumulative-episode Sharpe minus a drawdown penalty minus an excess-turnover penalty. Only turnover above `turnover_threshold` is penalized, leaving a free rebalancing band that avoids discouraging necessary trades. |

## Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- plotly (interactive dashboard)
- yfinance / requests_html (data download via `fin_data.py`)
- matplotlib (optional, for training-curve plots)

## Walk-Forward Strategy

The walk-forward engine implements a rolling-window approach for long time series with a dynamic stock universe:

```
Year:  1997 ──────────── 2002 ──── 2003 ──── 2004 ─────────── 2026
       │◄── 5yr train ──►│◄─ OOS ─►│
                │◄── 5yr train ──►│◄─ OOS ─►│
                         │◄── 5yr train ──►│◄─ OOS ─►│  ...
```

Each window trains a PPO agent on 5 years of daily data, then applies the learned weights **deterministically** to the next month. The window rolls forward by one month and repeats, producing a fully out-of-sample return series.

**Best-checkpoint selection:** During training, the engine snapshots the network weights whenever the rolling-average episode reward improves. The snapshot with the highest reward — not the final weights — is used for OOS application. This prevents overfitting during the later training episodes from degrading OOS performance.

**Realistic initial state:** At the start of each OOS period, the agent's current-weight input is the *drifted* end-of-previous-period portfolio (initial weights compounded by the prior month's per-stock returns and renormalized). This matches the true pre-rebalance state the agent would face in production and improves transaction-cost estimation accuracy.

**Dynamic universe handling:** Stocks may IPO or delist during any window. The `DynamicUniverse` class tracks each stock's lifecycle and produces boolean masks. Inactive stock slots receive large negative logits before the softmax/tanh layer, driving their weights to zero. Weights are renormalized over active stocks only. Delisted stocks retain their terminal returns in the training data to avoid survivorship bias.

**Warm-starting:** By default, each window initializes the network from the previous window's best parameters with a reduced learning rate (0.3×). This cuts training time significantly compared to training from scratch, since the model only needs to adapt to the incremental data shift. Early stopping (patience on rolling reward) further avoids wasted computation.

**Transaction costs:** The backtest applies one-way costs of `transaction_cost_bps + slippage_bps` (default: 15 bps total) to portfolio turnover. The equal-weight benchmark also has transaction costs applied — each month it must rebalance from the drifted weights back to equal weight, incurring costs proportional to that drift-driven turnover.
