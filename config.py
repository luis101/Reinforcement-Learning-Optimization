"""
Configuration of the RL portfolio optimization framework.
Setting all hyperparameters and architectural choices.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EnvironmentConfig:

    """Portfolio environment settings"""

    # Portfolio constraints - determines the action space and risk limits
    mode: Literal["long_only", "long_short"] = "long_only"
    max_position_size: float = 0.10  # Max weight per stock (risk limit)
    min_position_size: float = 0.0  # Min weight per stock (long_only)
    leverage_limit: float = 1.0  # Sum of |weights| for long_short

    # Rebalancing schedule - determines how often the agent can adjust the portfolio
    rebalance_freq: Literal["monthly", "weekly"] = "monthly"
    holding_period: Literal["monthly", "weekly"] = "monthly"

    # Transaction costs (basis points)
    transaction_cost_bps: float = 10.0  # 10 bps per trade
    slippage_bps: float = 5.0  # 5 bps slippage estimate
    transaction_cost = 0.001  # 10 bps in decimal

    # Reward settings - determines the learning signal for the agent
    reward_type: Literal["sharpe", "mse", "combined"] = "combined"
    sharpe_window: int = 252  # Rolling window for Sharpe calculation, i.e. 21, 63, 126, 252 days   
    risk_free_rate: float = 0.0  # Annualized risk-free rate
    drawdown_penalty: float = 0.5  # Penalty weight for drawdowns
    turnover_penalty: float = 0.1  # Penalty weight for portfolio turnover

    # Data lookback for state construction - determines how much historical data the agent sees
    lookback_window: int = 252  # with 63 ~3 months of trading days
    warmup_period: int = 252  # Days needed before first valid state


@dataclass
class FeatureConfig:
    """Feature engineering settings"""

    # Return windows
    return_windows: list[int] = [21, 63, 128, 252]

    # Technical indicators
    volatility_windows: list[int] = [21, 63, 128, 252]
    momentum_windows: list[int] = [21, 63, 128, 252]
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 21
    bollinger_std: float = 2.0
    
    # Cross-sectional features
    use_cross_sectional_rank: bool = False

    # Feature normalization
    normalize_method: Literal["zscore", "minmax"] = "zscore"
    normalize_window: int = 252  # Rolling normalization window (~1 year)


@dataclass
class NetworkConfig:
    """Neural network architecture settings"""

    # Feature extraction
    feature_hidden_dims: list[int] = [256, 128]

    # Cross-asset attention
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64

    # Actor network
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    actor_log_std_min: float = -10.0 # (-20, -10, -5) Minimum log std for action distribution to prevent collapse to deterministic policy
    actor_log_std_max: float = 1.0 # (0, 1, 2, 5) Maximum log std for action distribution to prevent excessive exploration

    # Critic network
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

    # Shared settings
    dropout: float = 0.1
    use_layer_norm: bool = True
    activation: Literal["relu", "gelu", "silu"] = "gelu"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # PPO hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping
    entropy_coeff: float = 0.01  # Entropy bonus for exploration
    value_loss_coeff: float = 0.5  # Value loss weight
    max_grad_norm: float = 0.5  # Gradient clipping

    # Training schedule
    n_episodes: int = 1000
    steps_per_update: int = 256  # Steps before PPO update
    n_epochs_per_update: int = 4  # PPO epochs per update
    minibatch_size: int = 64
    update_frequency: int = 1  # Update every N episodes

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_warmup_episodes: int = 50
    lr_decay_factor: float = 0.1
    lr_decay_episodes: list[int] = field(default_factory=lambda: [500, 800])

    # Evaluation
    eval_frequency: int = 10  # Evaluate every N episodes
    eval_episodes: int = 1  # Number of evaluation episodes

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"  # "cuda" or "cpu", auto-detected at runtime

    # Checkpointing
    save_frequency: int = 50
    checkpoint_dir: str = "checkpoints"

    # Loss function for critic
    critic_loss: Literal["mse", "huber"] = "huber"


@dataclass
class Config:
    """Master configuration."""

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)