"""
Configuration of the RL portfolio optimization framework.
Setting all hyperparameters and architectural choices.
"""
import pandas as pd
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
    reward_type: Literal["sharpe", "mse", "combined", "return", "dsr"] = "dsr"
    sharpe_window: int = 252  # Rolling window for Sharpe calculation, i.e. 21, 63, 126, 252 days
    # Differential Sharpe Ratio (Moody & Saffell 1998): per-step marginal contribution of the
    # current return to a moving-window Sharpe estimate maintained via EMAs of the first and
    # second moments. Decay η controls the effective window: η=0.01 ≈ 200-step half-life,
    # η=0.05 ≈ 40-step. Only used when reward_type == "dsr".
    dsr_eta: float = 0.05
    risk_free_rate: float = 0.0  # Annualized risk-free rate
    drawdown_penalty: float = 0.3  # Penalty weight for drawdowns
    turnover_penalty: float = 0.1  # Penalty weight for portfolio turnover
    turnover_threshold: float = 0.20  # Turnover below this level is not penalized (free rebalancing band)
    target_returns: pd.DataFrame | None = None  # Optional benchmark returns for MSE reward

    # Data lookback for state construction - determines how much historical data the agent sees
    lookback_window: int = 252  # with 63 ~3 months of trading days
    warmup_period: int = 252  # Days needed before first valid state

    # Block-bootstrap of returns for training-episode diversity (training-only). When enabled,
    # each reset resamples a synthetic return path and recomputes features on it.
    use_bootstrap: bool = False
    bootstrap_block: int = 21          # mean block length in trading days (stationary bootstrap)
    bootstrap_refresh_every: int = 5   # resample a new synthetic path every N resets (1, 5, 10, ...)
    bootstrap_seed: int | None = None  # RNG seed for reproducibility


@dataclass
class BacktestConfig:
    """Configuration for walk-forward optimization and backtesting."""

    # Window settings
    train_window_years: float = 5.0  # Training window in years - Amount of historical data to train on for each window
    train_window_start: float = train_window_years - 2.5  # Train window years required in addition to warmup period before first window starts 
    step_size: int = 1  # Roll forward by N months or weeks for the next window

    # Training per window
    episodes_per_window: int = 500  # Episodes to train each window
    warmstart: bool = True  # Initialize from previous window's model
    warmstart_lr_factor: float = 0.5  # Reduce LR when warm-starting

    # Early stopping within each window
    patience: int = 50  # Stop if no improvement for N episodes
    min_episodes: int = 100  # Always train at least this many episodes

    # Universe handling
    min_active_stocks: int = 30  # Minimum number of active stocks required
    handle_delistings: Literal["mask", "fill_zero"] = "mask"

    # K-fold initialization
    n_kfold_splits: int = 5  # Number of folds for k-fold warm-start initialization

    # LSTM warmup: number of prior rebalance dates to step the hidden state through
    # before the deterministic OOS prediction so it isn't zero-initialized at inference
    # (matches the non-zero hidden state the LSTM had at the end of each training
    # episode). The hidden state saturates quickly with typical market features, so
    # ~12 steps (≈1 year monthly) is usually plenty — going further back drags in
    # stale regimes. Set to 0 to disable warmup.
    lstm_warmup_steps: int = 12

    # Output
    save_window_models: bool = False
    output_dir: str = "walkforward_results"


@dataclass
class FeatureConfig:
    """Feature engineering settings"""

    # Return windows
    return_windows: list[int] = field(default_factory=lambda: [21, 63, 128, 252])

    # Technical indicators
    volatility_windows: list[int] = field(default_factory=lambda: [21, 63, 128, 252]) 
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 21
    bollinger_std: float = 2.0
    
    # Cross-sectional features
    use_cross_sectional_rank: bool = True  # Whether to include cross-sectional ranks of features

    # Feature normalization
    normalize_method: Literal["zscore", "minmax"] = "zscore"
    normalize_window: int = 252  # Rolling normalization window (~1 year)

    # Hidden Markov Model regime indicator — fits a Gaussian HMM on daily market returns
    # and appends predicted next-step regime probabilities as market features.
    use_regime: bool = False
    n_regimes: int = 4       # Number of hidden market states (e.g. bull / bear / volatile / calm)
    regime_vol_window: int = 126  # Rolling window (days) for realised-vol observation fed to HMM



@dataclass
class NetworkConfig:
    """Neural network architecture settings"""

    # Feature extraction
    feature_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

    # Cross-asset attention
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 128

    # Actor network
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    actor_log_std_min: float = -10.0 # (-20, -10, -5) Minimum log std for action distribution to prevent collapse to deterministic policy
    actor_log_std_max: float = 2.0 # (0, 1, 2, 5) Maximum log std for action distribution to prevent excessive exploration
    # Output layer init: "orthogonal" (structured near-equal-weight and near-uniform softmax), "normal" (simple near-equal-weight),
    # "xavier" (standard DL init, no equal-weight bias), "kaiming_uniform" / "kaiming_normal" (Init for ReLU/GELU).
    # For near-uniform initial policy use orthogonal or normal; xavier/kaiming concentrate weights more.
    policy_output_init: Literal["orthogonal", "normal", "xavier", "kaiming_uniform", "kaiming_normal"] = "orthogonal"

    # Policy head architecture — controls how per-stock attention embeddings are mapped to action logits.
    # - "flatten_mlp": compress per-stock embeddings to 32 dims, flatten to (N*32 + market) and feed into an
    #   MLP that outputs n_stocks logits. Output and first hidden layer are position-specific.
    # - "shared_head_compressed": keep the 32-dim compression, then apply a shared per-stock MLP to each
    #   stock's (32 + market) vector. Permutation-equivariant, much smaller parameter count.
    # - "shared_head": skip the 32-dim compression and apply a shared per-stock MLP directly to the full
    #   attention output + market features per stock. Largest per-stock representation, still shared weights.
    policy_head: Literal["flatten_mlp", "shared_head_compressed", "shared_head"] = "shared_head_compressed"

    # Softmax temperature — lower value concentrates weights; learned parameter annealing from
    # temperature_init toward temperature_min during training.
    use_temperature: bool = True
    temperature_init: float = 1.0
    temperature_min: float = 0.3

    # Two-stage action head — log-sigmoid inclusion gate added to action logits before
    # softmax, softly zeroing low-conviction stocks without forced full entry/exit.
    use_two_stage: bool = True 

    # Critic network
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])

    # Shared settings
    dropout: float = 0.1
    use_layer_norm: bool = True
    activation: Literal["relu", "gelu", "silu"] = "gelu"

    # LSTM temporal context — augments market features with a hidden state carried across
    # rebalancing steps within each episode. 
    use_lstm: bool = False
    lstm_hidden_dim: int = 64


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # PPO hyperparameters
    lr_actor: float = 0.0005
    lr_critic: float = 0.001
    gamma: float = 0.99  # Discount factor (0.999, 0.99, 0.9, 0.5, ...)
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clipping
    entropy_coeff: float = 0.001  # Entropy bonus for exploration 
    # 0.01 - prevent premature convergence to suboptimal deterministic policies
    # 0.001 - some exploration pressure but allows weight concentration
    # 0.0001 - near-zero regularization, policy can concentrate freely
    # 0.0 - no entropy pressure at all
    value_loss_coeff: float = 0.5  # Value loss weight
    max_grad_norm: float = 0.5  # Gradient clipping
    kl_target: float = 0.03  # Max KL divergence per update epoch before early stopping - 0.3 in between
    # Higher (e.g. 0.05) allows larger policy updates per window - weights can change more per training run 
    # Lower (e.g. 0.01) tighter early stopping - more stable but slower to move away from equal-weight

    # Training schedule
    n_episodes: int = 1000
    steps_per_update: int = 256  # Steps before PPO update
    n_epochs_per_update: int = 8  # PPO epochs per update
    minibatch_size: int = 16  # Minibatch size for PPO updates
    update_frequency: int = 1  # Update every N episodes

    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_warmup_episodes: int = 50
    lr_decay_factor: float = 0.3 # Factor to decay LR by at each milestone episode
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
    critic_loss: Literal["mse", "huber"] = "mse" # Huber is more robust to outliers in returns

    # Feature dropout — zeros out entire feature dimensions (same mask across all stocks)
    # at each training step to improve OOS generalization. 0.0 = disabled.
    feature_dropout_rate: float = 0.0


@dataclass
class Config:
    """Master configuration"""

    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)