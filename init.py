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
from .training import Train, create_and_train
from .utils import (
    compute_portfolio_metrics,
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
    "Train",
    "create_and_train",
    "compute_portfolio_metrics",
    "evaluate_agent",
    "format_metrics",
    "plot_training_results",
]