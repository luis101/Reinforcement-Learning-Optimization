"""
Portfolio Environment for Reinforcement Learning.

Implements a gym-like interface for portfolio optimization. The agent observes
market features and current holdings, then decides portfolio weights at each
rebalancing point. Between rebalancing points, the portfolio drifts with market returns.
"""

import numpy as np
import pandas as pd
from typing import NamedTuple

from .config import EnvironmentConfig, FeatureConfig
from .features import FeatureConstructor


class StepResult(NamedTuple):
    """Result of an environment step."""

    state: np.ndarray
    stock_features: np.ndarray
    market_features: np.ndarray
    reward: float
    done: bool
    info: dict


class PortfolioEnv:
    """
    Portfolio optimization environment.

    The environment steps through trading days. At rebalancing points,
    the agent chooses new portfolio weights. Between rebalancing points,
    the portfolio evolves with market returns. Rewards are computed based
    on realized portfolio performance over the holding period.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        target_returns: pd.DataFrame | None = None,
        env_config: EnvironmentConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """
        Args:
            prices: DataFrame (n_days, n_stocks) of adjusted close prices.
            target_return: Target return series for reward calculation.
            env_config: Environment configuration.
            feature_config: Feature engineering configuration.
        """
        self.config = env_config or EnvironmentConfig()
        self.prices = prices
        self.returns = prices.pct_change().fillna(0)
        self.n_stocks = prices.shape[1]
        self.n_days = prices.shape[0]
        self.target_returns = target_returns

        # Feature engine
        self.feature_engine = FeatureConstructor(prices, feature_config)

        # Compute rebalancing schedule
        self._rebalance_dates = self._build_rebalance_schedule()

        # State variables (set in reset)
        self._current_weights: np.ndarray | None = None
        self._step_idx: int = 0
        self._rebalance_idx: int = 0
        self._portfolio_values: list[float] = []
        self._trade_history: list[dict] = []

    # Gym-like interface

    def reset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset environment to the start.

        Returns:
            Tuple of (flat_state, stock_features, market_features).
        """
        self._rebalance_idx = 0
        self._step_idx = self._rebalance_dates[0]
        self._current_weights = np.zeros(self.n_stocks, dtype=np.float32)
        if self.config.mode == "long_only":
            # Start with equal weight
            self._current_weights[:] = 1.0 / self.n_stocks
        elif self.config.mode == "long_short":
            # Start with zero weights (cash)
            self._current_weights[:] = 0.0
        self._portfolio_values = [1.0]  # Start with unit value
        self._trade_history = []

        return self._get_state()

    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute one rebalancing step.

        Args:
            action: Raw action from the agent (n_stocks,). 
            Will be processed into valid portfolio weights.

        Returns:
            StepResult with next state, reward, done flag, and info dict.
        """
        # Process raw action into valid portfolio weights
        
        # Convert raw network output into valid portfolio weights.

        # For long_only: softmax-like mapping to [0, 1] summing to 1.
        # For long_short: tanh-like mapping to [-1, 1] summing to 0.

        action = np.clip(action, -10, 10)  # Prevent numerical overflow

        if self.config.mode == "long_only":
            # Softmax to get positive weights summing to 
            exp_a = np.exp(action - np.max(action))  # Numerical stability
            weights = exp_a / np.sum(exp_a)
            # Apply max position constraint
            weights = np.clip(weights, self.config.min_position_size, self.config.max_position_size)
            weights /= weights.sum()  # Re-normalize

        elif self.config.mode == "long_short":
            # Tanh to get values in [-1, 1]
            weights = np.tanh(action)
            # Center to sum to 0 (dollar-neutral)
            weights -= weights.mean()
            # Scale to respect leverage limit
            abs_sum = np.abs(weights).sum()
            if abs_sum > 1e-8:
                weights = weights * self.config.leverage_limit / abs_sum
            # Apply position limits
            weights = np.clip(weights, -self.config.max_position_size, self.config.max_position_size,)
            # Re-center after clipping
            weights -= weights.mean()
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        new_weights = weights.astype(np.float32)

        # Compute transaction costs from rebalancing
        turnover = np.sum(np.abs(new_weights - self._current_weights))
        tc_rate = (self.config.transaction_cost_bps + self.config.slippage_bps) / 10_000
        transaction_cost = turnover * tc_rate

        # Record trade
        self._trade_history.append(
            {
                "rebalance_idx": self._rebalance_idx,
                "date_idx": self._step_idx,
                "date": self.prices.index[self._step_idx],
                "old_weights": self._current_weights.copy(),
                "new_weights": new_weights.copy(),
                "turnover": turnover,
                "transaction_cost": transaction_cost,
            }
        )

        # Update weights
        self._current_weights = new_weights.copy()

        # Simulate holding period and compute portfolio return
        holding_return, holding_returns_daily, period_returns = self._holding_period_returns()

        # Apply transaction costs to the first day's return
        net_return = holding_return - transaction_cost

        # Update portfolio value
        current_value = self._portfolio_values[-1] * (1 + net_return)
        self._portfolio_values.append(current_value)

        # Advance to next rebalancing date
        self._rebalance_idx += 1
        done = self._rebalance_idx >= len(self._rebalance_dates)

        if not done:
            self._step_idx = self._rebalance_dates[self._rebalance_idx]

            # Compute how weights drift due to different stock returns during
            # the holding period (before next rebalance) for tc computation.

            # Get individual stock returns over the holding period
            start = self._rebalance_dates[self._rebalance_idx - 1]
            if self._rebalance_idx < len(self._rebalance_dates):
                end = self._rebalance_dates[self._rebalance_idx]
            else:
                end = self.n_days - 1

            # Compound individual stock returns
            stock_returns = self.returns.iloc[start + 1 : end + 1].values
            cumulative = np.prod(1 + stock_returns, axis=0)

            # Adjust weights proportionally due to differential returns
            adjusted_weights = weights * cumulative
            den = adjusted_weights.sum()
            if abs(den) > 1e-10:
                adjusted_weights /= den
            else:
                adjusted_weights = weights.copy()

            self._current_weights = adjusted_weights.astype(np.float32)

        # Compute reward
        if self.target_returns is not None:
            target_return = self.target_returns.iloc[self._step_idx]
        else:
            target_return = np.mean(period_returns, axis=1).mean()  # Use realized return as target if not provided

        reward = self._compute_reward(
            net_return, target_return, holding_returns_daily, turnover
        )

        # Build info dict
        info = {
            "portfolio_return": net_return,
            "gross_return": holding_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "portfolio_value": current_value,
            "daily_returns": holding_returns_daily,
        }

        if done:
            obs = self._get_state()  # Terminal state (won't be used)
        else:
            obs = self._get_state()

        return StepResult(
            state=obs[0],
            stock_features=obs[1],
            market_features=obs[2],
            reward=reward,
            done=done,
            info=info,
        )

    # Properties to expose environment characteristics and state dimensions
    
    @property
    def state_dim(self) -> int:
        """Dimension of the flat state vector (features + current weights)."""
        return self.feature_engine.n_features + self.n_stocks

    @property
    def stock_feature_dim(self) -> int:
        """Number of per-stock features."""
        return self.feature_engine.n_stock_features + 1  # +1 for current weight

    @property
    def market_feature_dim(self) -> int:
        """Number of market-level features."""
        return self.feature_engine.n_market_features

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return self.n_stocks

    @property
    def n_rebalance_steps(self) -> int:
        """Number of rebalancing steps in an episode."""
        return len(self._rebalance_dates)

    @property
    def portfolio_value_series(self) -> np.ndarray:
        """Portfolio value series for the current episode."""
        return np.array(self._portfolio_values)

     # Utility methods for state construction, reward calculation, etc.

    def _get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build state from features and current weights."""
        # Flat state: features + current weights
        features = self.feature_engine.get_state_features(self._step_idx)
        flat_state = np.concatenate([features, self._current_weights])

        # Per-stock features for attention: (n_stocks, n_features_per_stock + 1)
        stock_features = self.feature_engine.get_stock_features(self._step_idx)
        stock_features = np.hstack(
            [stock_features, self._current_weights.reshape(-1, 1)]
        )

        # Market features
        market_features = self.feature_engine.get_market_features(self._step_idx)

        return (
            flat_state.astype(np.float32),
            stock_features.astype(np.float32),
            market_features.astype(np.float32),
        )
    
    def _build_rebalance_schedule(self) -> list[int]:
        """
        Build list of date indices where rebalancing occurs.
        """
        dates = self.prices.index
        start_idx = self.feature_engine.valid_start_idx
        rebalance_indices = []

        if self.config.rebalance_freq == "monthly":
            # Rebalance on the last trading day of each month
            for i in range(start_idx, len(dates)):
                if i + 1 < len(dates) and dates[i].month != dates[i + 1].month:
                    rebalance_indices.append(i)
                elif i == len(dates) - 1:
                    rebalance_indices.append(i)
            # Also include start as first rebalance point
            if not rebalance_indices or rebalance_indices[0] != start_idx:
                rebalance_indices.insert(0, start_idx)

        elif self.config.rebalance_freq == "weekly":
            # Rebalance on Fridays (or last trading day of the week)
            for i in range(start_idx, len(dates)):
                if i + 1 < len(dates):
                    if dates[i].isocalendar()[1] != dates[i + 1].isocalendar()[1]:
                        rebalance_indices.append(i)
                elif i == len(dates) - 1:
                    rebalance_indices.append(i)
            if not rebalance_indices or rebalance_indices[0] != start_idx:
                rebalance_indices.insert(0, start_idx)
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.config.rebalance_freq}")

        return rebalance_indices

    def _holding_period_returns(self) -> tuple[float, np.ndarray, pd.DataFrame]:
        """
        Simulate portfolio returns over the holding period.

        Returns:
            (total_return, daily_returns_array, period_returns_df)
        """
        start = self._step_idx
        if self._rebalance_idx + 1 < len(self._rebalance_dates):
            end = self._rebalance_dates[self._rebalance_idx + 1]
        else:
            end = self.n_days - 1

        period_returns = self.returns.iloc[start + 1 : end + 1].values  # (days, stocks)

        if len(period_returns) == 0:
            return 0.0, np.array([0.0]), pd.DataFrame() 

        # Daily portfolio returns (constant weights during holding)
        daily_port_ret = period_returns @ self._current_weights
        total_return = np.prod(1 + daily_port_ret) - 1

        return total_return, daily_port_ret, period_returns

    def _compute_reward(
        self, net_return: float, target_return: float, daily_returns: np.ndarray, turnover: float
        ) -> float:
        """
        Compute reward based on configured reward type.
        """
        reward_type = self.config.reward_type

        if reward_type == "mse":
            # MSE reward: penalize deviation from a target return 
            # (e.g., market or any benchmark return). 
            # target_return = self.target_returns.iloc[self._step_idx]
            reward = -((net_return - target_return) ** 2)

        elif reward_type == "sharpe" or reward_type == "combined":
            # Compute Sharpe-ratio-based reward from daily returns.
            if len(daily_returns) < 2:
                return 0.0
            daily_rf = self.config.risk_free_rate / 252
            ret_exc = daily_returns - daily_rf
            mean_excess = ret_exc.mean()
            std_excess = ret_exc.std()
            if std_excess < 1e-8:
                sharpe = mean_excess * np.sqrt(252)  # If no variance, just return scaled mean
            else:
                sharpe = (mean_excess / std_excess) * np.sqrt(252)

            if reward_type == "sharpe":
                reward = sharpe

            elif reward_type == "combined":
                # Combine Sharpe-like reward with penalties
  
                # Drawdown penalty
                dd_penalty = 0.0
                if len(self._portfolio_values) >= 2:
                    peak = max(self._portfolio_values)
                    current = self._portfolio_values[-1]
                    drawdown = (peak - current) / peak
                    dd_penalty = drawdown * self.config.drawdown_penalty

                # Turnover penalty
                turnover_penalty = turnover * self.config.turnover_penalty

                reward = sharpe - dd_penalty - turnover_penalty

        else:
            raise ValueError(f"Unknown reward type: {reward_type}")

        return float(reward)


class MultiPeriodEnv:
    """
    Wrapper that splits data into train/validation/test periods
    and provides separate environments for each.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        env_config: EnvironmentConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        n = len(prices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # Add overlap for feature warmup
        warmup = (feature_config or FeatureConfig()).normalize_window + 63

        self.train_env = PortfolioEnv(
            prices.iloc[:train_end], env_config, feature_config
        )
        self.val_env = PortfolioEnv(
            prices.iloc[max(0, train_end - warmup) : val_end],
            env_config,
            feature_config,
        )
        self.test_env = PortfolioEnv(
            prices.iloc[max(0, val_end - warmup) :],
            env_config,
            feature_config,
        )