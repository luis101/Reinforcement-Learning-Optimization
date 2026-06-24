# RL Portfolio Optimization Framework

A deep reinforcement learning framework for portfolio optimization using PPO (Proximal Policy Optimization) with PyTorch. The agent learns to allocate capital across a dynamic stock universe (S&P 500, ~500 stocks) by observing market features and current holdings, then rebalancing at configurable intervals.

## Architecture

![Framework architecture](rl_portfolio_architecture.svg)

The framework follows an actor-critic loop. Price data flows through a feature engine that computes technical indicators and return statistics. The **portfolio environment** packages these into a state (per-stock features, market-level features, and the current portfolio weights) and passes it to two separate networks:

- **Actor** - a policy network with cross-asset multi-head attention that outputs a Gaussian distribution over portfolio weights. Attention lets the model learn inter-stock dependencies rather than treating each stock independently.

- **Critic** - a value network with the same attention architecture (separate parameters) that estimates V(s)for advantage computation.

**Cross-asset attention** lets the model relate stocks to one another rather than treating each independently. 

**Action processing** maps raw actions to valid weights:
- *long-only:* softmax → strict per-asset cap by iterative clip-and-redistribute (weights in `[0, max_position_size]`, still summing to 1).
- *long-short:* tanh → mean-center to dollar-neutral → scale to the leverage limit → clip to `±max_position_size`.

A PPO update with a clipped objective and Generalized Advantage Estimation keeps training stable. Three optional components are **off by default** (see Configuration): an LSTM temporal context, an HMM market-regime feature, and block-bootstrap training-episode diversity.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Typed dataclasses for all hyperparameters and architectural choices |
| `features.py` | Technical indicators, rolling normalization, optional HMM regime |
| `environment.py` | Gym-like portfolio environment; weight mapping, transaction costs, block bootstrap |
| `networks.py` | Actor-critic networks with masked cross-asset attention; optional LSTM |
| `algo.py` | PPO agent with GAE, rollout buffer, and stored-state recurrence |
| `training.py` | Training loop with evaluation, early stopping, and checkpointing |
| `forwardbacktest.py` | Walk-forward optimization engine |
| `universe.py` | Dynamic stock universe handling (IPOs and delistings) |
| `fin_data.py` | S&P 500 constituent and price data download |
| `utils.py` | Performance metrics, Plotly dashboard, and evaluation utilities |
| `utils_html.py` | Chart.js fallback dashboard (no Plotly dependency) |
| `main_backtest.py` | Example: full walk-forward backtest with dashboard output |
| `main_run.ipynb` | Notebook walkthrough |

## Quickstart

```bash
pip install torch pandas numpy plotly scikit-learn hmmlearn yfinance requests_html 
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

# Generate interactive dashboard. One date per OOS period.
oos_dates = pd.DatetimeIndex([r.oos_start for r in results["window_results"]])
generate_dashboard(
    rl_results=results["oos_returns"],
    bm_daily_returns=bm_returns,          # per-period equal-weight benchmark returns
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

All settings live in typed dataclasses (`Config` aggregates `env`, `features`, `network`, `training`, `backtest`). The most commonly adjusted classes:

```python
from rl_agent.config import (
    EnvironmentConfig, BacktestConfig, TrainingConfig, FeatureConfig, NetworkConfig
)
from rl_agent.forwardbacktest import WalkForwardBacktestEngine

engine = WalkForwardBacktestEngine(
    prices,
    wf_config=BacktestConfig(
        train_window_years=5.0,      # Years of data per training window
        step_size=1,                 # Roll forward by N months each step
        episodes_per_window=500,     # Training episodes per window
        warmstart=True,              # Fine-tune from the previous window's model
        warmstart_lr_factor=0.5,     # Learning-rate multiplier when warm-starting
        patience=50,                 # Early-stopping patience (episodes)
        min_episodes=100,
    ),
)
```

Key `EnvironmentConfig` options:

```python
EnvironmentConfig(
    mode="long_only",             # or "long_short"
    rebalance_freq="monthly",     # or "weekly"
    reward_type="dsr",            # "dsr" (default), "return", "sharpe", "mse", "combined"
    transaction_cost_bps=10,      # One-way transaction cost in basis points
    slippage_bps=5,               # Slippage estimate in basis points
    max_position_size=0.10,       # Maximum weight per stock (strictly enforced)
    turnover_threshold=0.20,      # "combined" reward: turnover below this is not penalized
    drawdown_penalty=0.3,         # Used by the "combined" reward type only
    turnover_penalty=0.1,         # Used by the "combined" reward type only
    use_bootstrap=False,          # Block-bootstrap training-episode (training-only)
    bootstrap_block=21,           # Mean block length in trading days
    bootstrap_refresh_every=5,    # Resample a new synthetic path every N episodes
)
```

Key `NetworkConfig` options:

```python
NetworkConfig(
    use_attention=True,                     # Cross-asset attention (with inactive-stock mask)
    policy_head="shared_head_compressed",   # Permutation-equivariant actor head
    use_two_stage=True,                     # Log-sigmoid inclusion gate on the logits
    use_temperature=True,                   # Learned softmax temperature (anneals 1.0 -> 0.3)
    use_lstm=False,                         # Optional temporal context (stored-state recurrent PPO)
)
```

Key `TrainingConfig` knobs: `gamma` (discount), `gae_lambda`, `clip_epsilon`, `entropy_coeff`, `kl_target`, and the learning rates. `FeatureConfig` exposes the indicator windows and an optional HMM regime feature (`use_regime=False`).

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
| Regime probabilities (optional) | next-step Gaussian-HMM state probs |

The PPO (Percentage Price Oscillator) expresses the MACD oscillator as a percentage of the slow EMA, making it scale-invariant across stocks at any price level — a $10 stock and a $200 stock produce directly comparable values. The optional **regime feature** (`use_regime`) fits a Gaussian HMM on daily market return/volatility and appends predicted next-step regime probabilities.

## Reward Options

Set via `EnvironmentConfig.reward_type`.

| Type | Description |
|---|---|
| **dsr** (default) | **Differential Sharpe Ratio** (Moody & Saffell 1998): the per-step marginal contribution of the period's net return to a moving-window Sharpe estimate, maintained via EMAs of the first and second moments (decay `dsr_eta`). |
| **return** | `log(1 + net_return)` per rebalancing period. Simple and unbiased baseline. |
| **sharpe** | **Full-window Sharpe**: 0 until the episode ends, then the annualized Sharpe of *all* daily portfolio returns accumulated over the episode. The episode return therefore equals the true full-window Sharpe (no per-step double-counting). |
| **mse** | Negative squared deviation from a target return series (e.g., an index). |
| **combined** | Annualized Sharpe ratio computed from all daily returns accumulated during the episode (not just the current holding period) minus a drawdown penalty minus an excess-turnover penalty. Gives a stable Sharpe estimate that improves as the episode progresses. Only turnover above `turnover_threshold` is penalized, leaving a free rebalancing band. |

## Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- scikit-learn (k-fold warm-start init)
- hmmlearn (HMM regime feature)
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

**Best-checkpoint selection:** During training, the engine snapshots the network weights whenever the rolling-average episode reward improves. The snapshot with the highest reward - not the final weights - is used for OOS application, preventing late-training overfitting from degrading OOS performance.

**Realistic initial state:** At the start of each OOS period, the agent's current-weight input is the *drifted* end-of-previous-period portfolio (prior weights compounded by the month's per-stock returns and renormalized). This matches the true pre-rebalance state and improves transaction-cost estimation.

**Dynamic universe handling:** Stocks may IPO or delist during any window. The `DynamicUniverse` class tracks each stock's lifecycle and produces boolean masks. Inactive stocks are excluded everywhere they could leak: from the **cross-asset attention** (key mask), from the **critic's pooling** (masked mean + max over active stocks), and from the **action** (large negative logits before softmax). Active weights are renormalized and capped over active stocks only. Delisted stocks retain their terminal returns in the training data to avoid survivorship bias.

**Warm-starting:** By default, each window initializes the network from the previous window's best parameters with a reduced learning rate (0.5×), so the model only adapts to the incremental data shift instead of training from scratch. Early stopping (patience on rolling reward) further avoids wasted computation. An optional k-fold initialization (`run(kfold_init=True)`) cross-validates the first window before the main loop.

**Episode diversity (optional):** With `use_bootstrap=True`, each training episode resamples a synthetic return path via a **stationary block bootstrap** over whole rows (preserving each day's cross-sectional structure) and recomputes features on it. This trains the agent on a distribution of plausible histories rather than the single realized path to omit overfitting. It is training-only, the OOS evaluation always uses the actual real path.

**Transaction costs:** The backtest applies one-way costs of `transaction_cost_bps + slippage_bps` (default: 15 bps total) to portfolio turnover. The equal-weight benchmark also applies costs - each month it must rebalance from the drifted weights back to equal weight, incurring costs proportional to that drift-driven turnover.