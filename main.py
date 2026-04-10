"""
Example: Training an RL portfolio agent.

This script demonstrates the full pipeline using synthetic price data.
Replace the synthetic data section with your own DataFrame to use real data.
"""
import numpy as np
import pandas as pd
import torch

from .config import (
    Config, 
    EnvironmentConfig, FeatureConfig, 
    NetworkConfig, TrainingConfig,
)
from .environment import PortfolioEnv
from .algo import PPOAgent
from .training import Train, create_and_train
from .utils import (
    compute_portfolio_metrics,
    evaluate_agent,
    format_metrics,
)


###### 1. Generate synthetic price data (replace with your real data) ######

def generate_synthetic_prices(
    n_days: int = 1500,
    n_stocks: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic stock prices with realistic properties:
    - Mean-reverting factors
    - Momentum effects
    - Time-varying volatility
    - Cross-sectional correlation
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start="2018-01-02", periods=n_days, freq="B")
    tickers = [f"STOCK_{i:03d}" for i in range(n_stocks)]

    # Market factor (common driver)
    market_vol = 0.16 / np.sqrt(252)
    market_returns = rng.normal(0.08 / 252, market_vol, n_days)

    # Stock-specific parameters
    betas = rng.uniform(0.5, 1.5, n_stocks)
    alphas = rng.normal(0.0, 0.02 / 252, n_stocks)
    idio_vols = rng.uniform(0.15, 0.40, n_stocks) / np.sqrt(252)

    # Sector factors (3 sectors)
    sector_assignments = rng.integers(0, 3, n_stocks)
    sector_returns = rng.normal(0, 0.05 / np.sqrt(252), (n_days, 3))

    # Generate returns
    returns = np.zeros((n_days, n_stocks))
    for i in range(n_stocks):
        systematic = betas[i] * market_returns + sector_returns[:, sector_assignments[i]] * 0.5
        idiosyncratic = rng.normal(0, idio_vols[i], n_days)
        returns[:, i] = alphas[i] + systematic + idiosyncratic

    # Convert to prices
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=dates[:n_days], columns=tickers)



###### 2. Configuration examples ######

def get_long_only_monthly_config() -> Config:
    """Standard long-only monthly rebalancing configuration."""
    return Config(
        env=EnvironmentConfig(
            n_stocks=100,
            mode="long_only",
            max_position_size=0.05,
            rebalance_freq="monthly",
            holding_period="monthly",
            transaction_cost_bps=10,
            reward_type="combined",
            drawdown_penalty=0.5,
            turnover_penalty=0.1,
        ),
        features=FeatureConfig(
            normalize_method="robust",
        ),
        network=NetworkConfig(
            use_attention=True,
            attention_heads=4,
            attention_dim=64,
            actor_hidden_dims=[256, 128],
            critic_hidden_dims=[256, 128],
            dropout=0.1,
        ),
        training=TrainingConfig(
            lr_actor=3e-4,
            lr_critic=1e-3,
            n_episodes=200,          # Increase for real training
            n_epochs_per_update=4,
            clip_epsilon=0.2,
            entropy_coeff=0.01,
            eval_frequency=20,
            save_frequency=50,
            seed=42,
        ),
    )


def get_long_short_weekly_config() -> Config:
    """Long-short weekly rebalancing configuration."""
    return Config(
        env=EnvironmentConfig(
            n_stocks=100,
            mode="long_short",
            max_position_size=0.05,
            leverage_limit=2.0, 
            rebalance_freq="weekly",
            holding_period="weekly",
            transaction_cost_bps=10,
            reward_type="sharpe",
        ),
        features=FeatureConfig(
            normalize_method="robust",
        ),
        network=NetworkConfig(
            use_attention=True,
        ),
        training=TrainingConfig(
            lr_actor=1e-4,            # Lower LR for more frequent trading
            lr_critic=5e-4,
            n_episodes=300,
            entropy_coeff=0.02,       # More exploration for L/S
            seed=42,
        ),
    )


def main():

    ###### 3. Training and evaluation ######
    
    # Generate synthetic data
    print("Generating synthetic price data...")
    prices = generate_synthetic_prices(n_days=1500, n_stocks=100)
    print(f"  Shape: {prices.shape}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    # Choose configuration
    config = get_long_only_monthly_config()

    # Quick training demo (reduce episodes for testing)
    config.training.n_episodes = 50
    config.training.eval_frequency = 10

    # Use the training function to train the agent and evaluate
    print("\nStarting training...")
    agent, results = create_and_train(prices, config)

    # Print final results
    print("\n" + "=" * 50)
    print("  Training Summary")
    print("=" * 50)
    if "training" in results["final_eval"]:
        print("\n  Training set performance:")
        print(format_metrics(results["final_eval"]["training"]["metrics"]))

    if "validation" in results["final_eval"]:
        print("\n  Validation set performance:")
        print(format_metrics(results["final_eval"]["validation"]["metrics"]))


    ###### 4. Test set evaluation ######

    print("\n" + "=" * 50)
    print("  Out-of-Sample Test")
    print("=" * 50)

    n = len(prices)
    warmup = config.features.normalize_window + 63
    test_prices = prices.iloc[max(0, int(n * 0.8) - warmup) :]
    test_env = PortfolioEnv(test_prices, config.env, config.features)

    test_result = evaluate_agent(agent, test_env)
    print("\n  Test set performance:")
    print(format_metrics(test_result["metrics"]))

    # Compare with benchmark
    benchmark_values = np.array([1.0])
    test_returns = test_prices.pct_change().fillna(0)
    ew_daily = test_returns.mean(axis=1).values[test_env.feature_engine.valid_start_idx:]
    benchmark_values = np.cumprod(np.concatenate([[1.0], 1 + ew_daily]))

    bm_metrics = compute_portfolio_metrics(benchmark_values)
    print("\n  Equal-weight benchmark:")
    print(format_metrics(bm_metrics))


if __name__ == "__main__":
    main()