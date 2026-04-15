# RL Portfolio Optimization Framework

A deep reinforcement learning framework for portfolio optimization using PPO (Proximal Policy Optimization) with PyTorch. The agent learns to allocate capital across 100 stocks by observing market features and current holdings, then rebalancing at configurable intervals.

## Architecture

![Framework architecture](rl_portfolio_architecture.svg)

The framework follows a standard actor-critic loop. Price data flows through a feature engine that computes technical indicators and return statistics. The **portfolio environment** packages these into a state (per-stock features, market-level features, and current weights) and passes it to two separate networks:

- **Actor** — a policy network with cross-asset multi-head attention that outputs a Gaussian distribution over portfolio weights. Attention lets the model learn inter-stock dependencies rather than treating the 100-stock weight vector as flat.
- **Critic** — a value network (same attention architecture, separate parameters) that estimates V(s) for advantage computation.

Raw actions are mapped to valid weights through an **action processing** layer: softmax for long-only portfolios (weights in [0,1] summing to 1) or tanh with mean-centering for long-short (weights in [-1,1] summing to 0). A PPO update with clipped objective and Generalized Advantage Estimation keeps training stable.

## Modules

| File | Purpose |
|---|---|
| `config.py` | Typed dataclasses for all hyperparameters |
| `features.py` | Technical indicators and rolling normalization |
| `environment.py` | Gym-like portfolio environment with transaction costs |
| `networks.py` | Actor-critic networks with cross-asset attention |
| `agent.py` | PPO agent with GAE and rollout buffer |
| `trainer.py` | Training loop with evaluation and checkpointing |
| `utils.py` | Performance metrics and plotting utilities |
| `universe.py` | Dynamic stock universe handling (IPOs and delistings) |
| `walkforward.py` | Walk-forward optimization engine |
| `main.py` | Example: single-window training with synthetic data |
| `main_walkforward.py` | Example: full walk-forward backtest |

## Quickstart

```bash
pip install torch pandas numpy
```

**Single-window training:**

```python
from rl_portfolio import Config, create_and_train

# prices: pd.DataFrame with DatetimeIndex rows (days) and stock columns
agent, results = create_and_train(prices, Config())
```

**Walk-forward optimization** (recommended for production backtesting):

```python
from rl_portfolio import Config, EnvironmentConfig
from rl_portfolio.walkforward import WalkForwardEngine, WalkForwardConfig

wf_config = WalkForwardConfig(
    train_window_years=5.0,   # Train on 5 years of data
    step_months=1,            # Roll forward monthly
    episodes_per_window=200,  # Training intensity per window
    warmstart=True,           # Initialize from previous window's model
    rl_config=Config(env=EnvironmentConfig(mode="long_only")),
)

engine = WalkForwardEngine(prices, wf_config)
results = engine.run()
```

Or run the included examples:

```bash
python -m rl_portfolio.main               # Single window
python -m rl_portfolio.main_walkforward   # Full walk-forward
```

## Configuration

All settings are controlled through the `Config` dataclass. Key options:

```python
from rl_portfolio import Config, EnvironmentConfig, TrainingConfig

config = Config(
    env=EnvironmentConfig(
        mode="long_only",           # or "long_short"
        rebalance_freq="monthly",   # or "weekly"
        reward_type="combined",     # "sharpe", "mse", or "combined"
        transaction_cost_bps=10,
        max_position_size=0.05,
    ),
    training=TrainingConfig(
        n_episodes=500,
        lr_actor=3e-4,
        lr_critic=1e-3,
    ),
)
```

## Features

The feature engine computes per-stock and market-level indicators from raw prices, all with rolling normalization to prevent look-ahead bias:

- Rolling returns (5, 10, 21, 63 day)
- Rolling volatility (10, 21, 63 day)
- RSI, MACD histogram, Bollinger Band position
- Momentum at multiple horizons
- Cross-sectional return rank
- Average pairwise correlation, market return/volatility, return dispersion

## Reward Options

- **Sharpe** — annualized Sharpe ratio of holding-period returns
- **MSE** — negative squared deviation from an equal-weight benchmark
- **Combined** (default) — Sharpe minus a drawdown penalty and a turnover penalty, encouraging stable, cost-aware portfolios

## Requirements

- Python 3.10+
- PyTorch
- pandas
- NumPy
- matplotlib (optional, for plotting)

## Walk-Forward Strategy

The walk-forward engine implements a rolling-window approach suitable for long time series (25–30 years) with a dynamic stock universe:

```
Year:  1997 ──────────── 2002 ──── 2003 ──── 2004 ─────────── 2026
       │◄── 5yr train ──►│◄─ OOS ─►│
                │◄── 5yr train ──►│◄─ OOS ─►│
                         │◄── 5yr train ──►│◄─ OOS ─►│  ...
```

Each window trains a fresh (or warm-started) PPO agent on 5 years of daily data, then applies the learned weights deterministically to the next month. The window rolls forward by one month and repeats. This produces a fully out-of-sample return series spanning ~22 years.

**Dynamic universe handling:** Stocks may IPO or delist during any window. The `DynamicUniverse` class tracks each stock's lifecycle and produces boolean masks. Inactive stock slots receive large negative logits before the softmax/tanh layer, driving their weights to zero. Weights are renormalized over active stocks only. Delisted stocks retain their terminal returns in the training data to avoid survivorship bias.

**Warm-starting:** By default, each window initializes the network from the previous window's final parameters with a reduced learning rate (0.3×). This cuts training time by roughly 50–70% compared to training from scratch, since the model only needs to adapt to the incremental data shift rather than learn from zero. Early stopping (patience on rolling reward) further avoids wasted computation.