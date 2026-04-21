"""
RL Portfolio Optimization Framework

Deep reinforcement learning framework for portfolio optimization using
Actor-Critic (PPO) with PyTorch.

Example usage:
    from rl_portfolio import Config, create_and_train

    agent, results = create_and_train(prices_df, Config())
"""

from .config import (
    Config, 
    EnvironmentConfig, FeatureConfig, 
    NetworkConfig, TrainingConfig,
)
from .environment import PortfolioEnv, MultiPeriodEnv
from .features import FeatureConstructor
from .networks import ActorCritic, ActorNetwork, CriticNetwork
from .algo import PPOAgent
from .universe import DynamicUniverse
from .forwardbacktest import WalkForwardBacktestEngine
from .training import Train
from .fin_data import download_fin_data, get_sp500
from .utils import (
    compute_portfolio_metrics,
    generate_realistic_universe,
    evaluate_agent,
    format_metrics,
    plot_training_results,
)

__all__ = [
    "Config",
    "EnvironmentConfig",
    "FeatureConfig",
    "NetworkConfig",
    "TrainingConfig",
    "PortfolioEnv",
    "MultiPeriodEnv",
    "FeatureConstructor",
    "ActorCritic",
    "ActorNetwork",
    "CriticNetwork",
    "PPOAgent",
    "DynamicUniverse",
    "WalkForwardBacktestEngine",
    "Train",
    "download_fin_data",
    "get_sp500",
    "compute_portfolio_metrics",
    "generate_realistic_universe",
    "evaluate_agent",
    "format_metrics",
    "plot_training_results",
]