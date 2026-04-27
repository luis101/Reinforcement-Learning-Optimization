"""
Walk-Forward Optimization Engine.

Implements a rolling-window training and evaluation strategy:
1. Train the agent on a 5-year window of data - This can be set to any length, 
but 5 years is a common choice to balance learning and adaptability.
2. Apply the learned weights to the next month (out-of-sample prediction) - 
This simulates real-world usage training on historical data and then applying the model to future data.
3. Roll forward by one month and repeat
4. Continue until the end of the time series

This produces a fully out-of-sample backtest where every month's
portfolio weights were determined by a model that only relies on past data.
"""

import copy
import time
import numpy as np
import pandas as pd
import datetime as dt
# import matplotlib.dates as mdates
# import torch
from dataclasses import dataclass, field
from pathlib import Path
# from typing import Literal

from .config import EnvironmentConfig, FeatureConfig, NetworkConfig, TrainingConfig, BacktestConfig
from .environment import PortfolioEnv
from .algo import PPOAgent, Transition
from .features import FeatureConstructor
from .universe import DynamicUniverse
from .utils import compute_portfolio_metrics, format_metrics


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    oos_weights: np.ndarray  # (n_stocks,) applied weights
    oos_return: float
    oos_raw_return: float  # Return before transaction costs  
    oos_daily_returns: np.ndarray 
    active_mask: np.ndarray  # Which stocks were active
    n_active_stocks: int
    train_episodes: int
    train_time_seconds: float
    train_final_reward: float


class WalkForwardBacktestEngine:
    """
    Runs the full walk-forward optimization loop.

    The engine manages:
    - Rolling window construction with proper date alignment
    - Dynamic universe masking (IPOs and delistings)
    - Model warm-starting between windows
    - Out-of-sample weight application and return tracking
    - Comprehensive result aggregation
    """

    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame | None = None,
                 wf_config: BacktestConfig | None = None):
        """
        Args:
            prices: Full price DataFrame. NaN for dates before first listing or after delisting.
                    DatetimeIndex required. Columns are stock identifiers.
            returns: Pre-computed returns with the same shape, index, and columns as prices.
                     If provided, used directly wherever returns are needed instead of
                     recomputing pct_change. Price-level features (MACD, Bollinger) still
                     use clean_prices. If not provided, returns are derived from clean_prices.
        """
        self.wf_config = wf_config or BacktestConfig()
        self.env_config = EnvironmentConfig()
        self.feat_config = FeatureConfig()
        self.net_config = NetworkConfig()
        self.train_config = TrainingConfig()
        self.prices = self._get_clean_prices(prices)
        self.returns = self._get_clean_returns(prices, returns)

        # Dynamic universe tracker
        self.universe = DynamicUniverse(
            prices, max_stocks=self.prices.shape[1]
        )

        # Build the walk-forward schedule
        self._windows = self._get_periods()

        # Results storage
        self.results: list[WindowResult] = []
        self._current_agent: PPOAgent | None = None

        # Output directory
        self.output_dir = Path(self.wf_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        """
        Execute the full optimization based on walk-forward strategy.

        Returns:
            Dictionary containing:
            - oos_returns: monthly out-of-sample returns
            - oos_portfolio_values: cumulative portfolio value
            - window_results: list of WindowResult objects
            - aggregate_metrics: overall performance metrics
        """
        print("=" * 50)
        print("Reinforcement Learning Portfolio Optimization")
        print("=" * 50)
        print(self.universe.summary)
        print(f"  Train window:     {self.wf_config.train_window_years} years")
        print(f"  Warm-start:       {self.wf_config.warmstart}")
        print(f"  Episodes/window:  {self.wf_config.episodes_per_window}")
        print("=" * 50)

        total_start = time.time()

        for i, window in enumerate(self._windows):
            train_start_date, train_end_date, oos_start_date, oos_end_date = window

            # train_start_idx, train_end_idx, oos_start_idx, oos_end_idx = window
            # train_start_date = self.prices.index[train_start_idx]
            # train_end_date = self.prices.index[train_end_idx - 1]
            # oos_start_date = self.prices.index[oos_start_idx]
            # oos_end_date = self.prices.index[min(oos_end_idx - 1, len(self.prices) - 1)]

            print(f"\n--- Window {i + 1}/{len(self._windows)} ---")
            print(f"  Train: {train_start_date} → {train_end_date}")
            print(f"  OOS:   {oos_start_date} → {oos_end_date}")

            # Get active universe for this window
            active_mask = np.zeros(self.prices.shape[1], dtype=bool)
            active_mask[:self.prices.shape[1]] = ~self.prices.isna().loc[oos_start_date].values

            n_active = active_mask.sum()
            print(f"  Active stocks: {n_active}")
            if n_active < self.wf_config.min_active_stocks:
                print(f"  Skip period: Only {n_active} active stocks (minimum: {self.wf_config.min_active_stocks})")
                continue
            
            # Extract training data (handle NaN for missing data)
            window_prices = self.prices.loc[train_start_date:train_end_date].copy()
            window_returns = self.returns.loc[train_start_date:train_end_date].copy()
            # Forward-fill then backward-fill such that we have no NaN for feature computation, 
            # but returns will be 0 for inactive periods (before listing/after delisting)
            # window_prices = window_prices.ffill().bfill()
            # If any column is entirely NaN, fill with a constant to avoid computation errors
            # for col in window_prices.columns:
            #    if window_prices[col].isna().all():
            #        window_prices[col] = 100.0  # Placeholder, will be masked
            # window_prices.fillna(0.0, inplace=True) # Alternatively, fill with 0.0 
            # if we want features to reflect missing data more directly (e.g. 0 return before IPO)

            # Train the agent on this window
            window_start = time.time()
            agent, train_info = self._train_window(window_prices, window_returns, active_mask, i)
            train_time = time.time() - window_start

            # Get deterministic action (weights) from trained agent
            oos_weights, oos_return, oos_raw_return, oos_daily = self._apply_oos(agent, oos_start_date, oos_end_date, active_mask)

            # Store result
            result = WindowResult(
                window_idx=i,
                train_start=train_start_date,
                train_end=train_end_date,
                oos_start=oos_start_date,
                oos_end=oos_end_date,
                oos_weights=oos_weights,
                oos_return=oos_return,
                oos_raw_return=oos_raw_return,
                oos_daily_returns=oos_daily,
                active_mask=active_mask,
                n_active_stocks=int(n_active),
                train_episodes=train_info.get("episodes", 0),
                train_time_seconds=train_time,
                train_final_reward=train_info.get("final_reward", 0.0),
            )
            self.results.append(result)

            print(
                f"OOS return: {oos_return:+.4f} | "
                f"OOS raw return: {oos_raw_return:+.4f} | "
                f"Train time: {train_time:.1f}s | "
                f"Episodes: {train_info.get('episodes', 0)}"
            )

            # Save window model if requested
            if self.wf_config.save_window_models:
                agent.save(str(self.output_dir / f"agent_window_{i:03d}.pt"))

        # Aggregate results
        total_time = time.time() - total_start
        summary = self._aggregate_results()

        print("\n" + "=" * 50)
        print("Backtest Results")
        print("=" * 50)
        print(format_metrics(summary["metrics"]))
        print(f"\n Total time: {total_time / 60:.1f} minutes")
        print(f"Windows completed: {len(self.results)}/{len(self._windows)}")

        return summary
    
    def _get_clean_returns(self, prices: pd.DataFrame,
                           ext_returns: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        If returns are provided they are used directly (assuming they are pre-aligned and cleaned),
        Otherwise returns are derived from clean_prices with row-wise winsorization.
        """

        if ext_returns is not None:
            penny_mask = prices.isna()
            return ext_returns.where(~penny_mask)
        else:
            # Replace penny stocks with prices lower 1 with NaN in the prices DataFrame
            prices = prices.mask(prices < 1)
            returns = prices.pct_change()
        return returns.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=1)
    
    def _get_clean_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Get clean prices to account for larger outliers and data issues.
        """
        
        # Replace penny stocks with prices lower 1 with NaN in the prices DataFrame
        prices = prices.mask(prices < 1)

        # Get raw and winsorized returns
        returns = prices.pct_change()
        returns_win = returns.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=1)
        # returns_win = returns_win.clip(upper=3, lower=-0.5)
        returns_mask = returns != returns_win

        # For each consecutive run of non-NaN prices, reconstruct prices from the
        # first (anchor) price and winsorized returns, replacing outlier-affected days.        

        # Get adjusted prices
        adjusted_prices = []
        first_prices = prices[prices.shift(1).isnull()] # Keep price only if first observation 

        # first_prices = prices.where(prices.shift(1).isna())
        # group_ids = first_prices.notna().cumsum()
        # adj_rets = returns_win.mask(first_prices.notna(), 0)

        # cum_growth = adj_rets.apply(lambda col: (1 + col).groupby(group_ids[col.name]).cumprod())
        # anchor_prices = first_prices.apply(lambda col: col.groupby(group_ids[col.name]).transform("first"))
        # adjusted_prices = anchor_prices * cum_growth

        for col in first_prices.columns:
            group_id = first_prices[col].notna().cumsum()
            adj_rets = returns_win[col].mask(first_prices[col].notna(), 0)

            # Calculate cumulative growth for each group of consecutive non-NaN values
            cum_growth = (1 + adj_rets).groupby(group_id).cumprod() 
            adj_prices = first_prices[col].groupby(group_id).transform('first')

            adjusted_prices.append((adj_prices * cum_growth).rename(col))

        adjusted_prices = pd.concat(adjusted_prices, axis=1)

        # Replace outliers
        clean_prices = prices.copy()
        clean_prices[returns_mask] = adjusted_prices[returns_mask]
        
        return clean_prices


    ###### Determine periods for training and OOS application ######

    def _get_periods(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        
        train_days = int(self.wf_config.train_window_years * 252)
        freq = self.env_config.rebalance_freq
        dates = self.prices.index.to_series()

        if freq == "weekly":
            # End of current week
            period_ends = dates[dates.dt.isocalendar().week != dates.shift(-1).dt.isocalendar().week]
        elif freq == "monthly":
            # End of current month
            period_ends = dates[dates.dt.month != dates.shift(-1).dt.month]
        else:
            raise ValueError(f"freq must be 'weekly' or 'monthly', got {freq!r}")

        # Feature warmup requirement. Minimum trading days needed for feature computation
        warmup = self.feat_config.normalize_window + max(
            max(self.feat_config.return_windows),
            self.feat_config.macd_slow + self.feat_config.macd_signal,
            max(self.feat_config.volatility_windows),
        )

        step = self.wf_config.step_size # Roll forward by step periods (e.g. 1 month)

        # Train window
        train_end_date = period_ends # Exclusive end index for training
        train_start_date = train_end_date - pd.offsets.BDay(n=train_days)

        # Ensure training start is not before price data starts
        train_start_date.loc[train_start_date < self.prices.index[0]] = self.prices.index[0]

        # Filter valid training windows
        valid = ((train_end_date - train_start_date) >= (
            pd.Timedelta(days=(warmup + 252)*(365/252)))) # At least 1 year of effective training data after warmup
        #         ) & (train_start_date >= self.prices.index[0]) # Training start must be within price data

        train_start_date = train_start_date[valid]
        train_end_date = train_end_date[valid]
        period_ends = period_ends[valid]

        # OOS windows
        oos_start_date = train_end_date + pd.offsets.BDay(1)  # Next day available after training end
        oos_start_idx = self.prices.index.get_indexer(oos_start_date, method='bfill') # Find the next available date in prices
        oos_start_date = pd.Series(
            np.where(oos_start_idx >= 0, self.prices.index[np.maximum(oos_start_idx, 0)], pd.NaT),
            index=oos_start_date.index,
        ) 
        oos_end_date = period_ends.shift(-step)  # Next period end after rolling forward by step

        # Filter valid OOS windows        
        valid_oos = oos_end_date.notna() & (oos_end_date <= self.prices.index[-1])  # OOS end must be within price data
        train_start_date = train_start_date[valid_oos]
        train_end_date = train_end_date[valid_oos]
        oos_start_date = oos_start_date[valid_oos]
        oos_end_date = oos_end_date[valid_oos]

        return list(zip(train_start_date, train_end_date, oos_start_date, oos_end_date))

    ###### Training ######

    def _train_window(self,
        train_prices: pd.DataFrame, train_returns: pd.DataFrame,
        active_mask: np.ndarray, window_idx: int,
        ) -> tuple[PPOAgent, dict]:
        """
        Train (or warm-start) an agent on one window of data.

        Returns:
            (trained_agent, training_info_dict)
        """
        # Create environment for this window
        env_config = copy.deepcopy(self.env_config)
        env_config.n_stocks = train_prices.shape[1]

        feature_config = copy.deepcopy(self.feat_config)
        net_config = copy.deepcopy(self.net_config)
        train_config = copy.deepcopy(self.train_config)

        # Adjust training config for walk-forward
        train_config.n_episodes = self.wf_config.episodes_per_window
        train_config.eval_frequency = max(1, self.wf_config.episodes_per_window // 5)
        train_config.save_frequency = self.wf_config.episodes_per_window + 1  # Don't save

        train_env = PortfolioEnv(train_prices, returns=train_returns,
                                 target_returns=self.env_config.target_returns,
                                 env_config=env_config, feature_config=feature_config,
                                 )

        # Create or warm-start agent
        if self.wf_config.warmstart and self._current_agent is not None:
            agent = self._current_agent
            # Reduce learning rate for fine-tuning
            lr_factor = self.wf_config.warmstart_lr_factor
            for pg in agent.actor_optimizer.param_groups:
                pg["lr"] = train_config.lr_actor * lr_factor
            for pg in agent.critic_optimizer.param_groups:
                pg["lr"] = train_config.lr_critic * lr_factor
            agent.buffer.clear()
        else:
            agent = PPOAgent(
                n_stocks=env_config.n_stocks,
                stock_feature_dim=train_env.stock_feature_dim,
                market_feature_dim=train_env.market_feature_dim,
                train_config=train_config,
                net_config=net_config,
                env_config=env_config,
            )

        # Training loop with early stopping
        best_reward = -np.inf
        patience_counter = 0
        episode_rewards = []

        for ep in range(1, train_config.n_episodes + 1):
            # Collect rollout
            flat_state, stock_feats, market_feats = train_env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action, log_prob, value = agent.select_action(
                    stock_feats, market_feats
                )

                # Apply universe mask to action before stepping
                action[~active_mask[: len(action)]] = -1e6  # Will be zeroed by softmax/processing

                result = train_env.step(action)

                agent.store_transition(
                    Transition(
                        stock_features=stock_feats,
                        market_features=market_feats,
                        action=action,
                        log_prob=log_prob,
                        reward=result.reward,
                        value=value,
                        done=result.done,
                    )
                )

                stock_feats = result.stock_features
                market_feats = result.market_features
                done = result.done
                total_reward += result.reward

            # PPO update
            agent.update()
            episode_rewards.append(total_reward)

            # Early stopping check (on rolling average)
            if ep >= self.wf_config.min_episodes:
                recent_avg = np.mean(episode_rewards[-20:])
                if recent_avg > best_reward + 0.001:
                    best_reward = recent_avg
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.wf_config.patience:
                    break

        # Store agent for next window's warm-start
        self._current_agent = agent

        info = {
            "episodes": ep,
            "final_reward": np.mean(episode_rewards[-10:]) if episode_rewards else 0.0,
            "best_reward": best_reward,
        }

        return agent, info

    ###### Out-of-sample application ######

    def _apply_oos(self,
        agent: PPOAgent, oos_start_date: int, oos_end_date: int, active_mask: np.ndarray,
        ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Apply the trained agent's weights to the out-of-sample period.

        Here the agent produces weights deterministically (no exploration) and the 
        portfolio's performance over the OOS month is determined.

        Returns:
            (weights, total_return, daily_returns)
        """
        # Build the state at the OOS start date using full price history
        # We need features computed up to oos_start, so use a lookback window that covers all feature requirements
        lookback = self.feat_config.normalize_window + 100
        feature_start = max(self.prices.index[0], oos_start_date - pd.Timedelta(days=lookback))
        feature_prices = self.prices.loc[feature_start:oos_end_date].copy()
        
        feature_prices = feature_prices.ffill().bfill()
        for col in feature_prices.columns:
            if feature_prices[col].isna().all():
                feature_prices[col] = 100.0

        # Create a temporary environment just to get the state
        env_config = copy.deepcopy(self.env_config)
        env_config.n_stocks = feature_prices.shape[1]
        feat_config = copy.deepcopy(self.feat_config)

        feat_engine = FeatureConstructor(feature_prices, feat_config)

        # Get state at the OOS start (relative to feature_prices)
        # relative_oos_start = oos_start_date - (oos_start_date - feature_start)
        # relative_oos_start_idx = self.prices.index.get_indexer([relative_oos_start], method='ffill')[0]
        relative_oos_start_idx = feature_prices.index.get_indexer([oos_start_date], method='ffill')[0]
        stock_feats = feat_engine.get_stock_features(relative_oos_start_idx)

        # Add dummy current weights (equal weight for active stocks) 
        # Why not last window's weights? Because the agent should learn to produce valid weights 
        # even from a naive starting point, and we want to avoid leaking information.
        n_stocks = feature_prices.shape[1]
        current_weights = np.zeros(n_stocks, dtype=np.float32)
        n_active = active_mask[:n_stocks].sum()
        if n_active > 0:
            current_weights[active_mask[:n_stocks]] = 1.0 / n_active

        stock_feats = np.hstack([stock_feats, current_weights.reshape(-1, 1)])
        # market_feats = feat_engine.get_market_features(relative_oos_start)
        market_feats = feat_engine.get_market_features(relative_oos_start_idx)

        # Get deterministic action
        action, _, _ = agent.select_action(
            stock_feats, market_feats, deterministic=True
        )

        # Apply universe mask
        weights = self._obtain_oos_weights(action, active_mask[:n_stocks])

        # Simulate OOS returns — use pre-computed clean_returns (already winsorized and with NaN for missing data)
        oos_returns = self.returns.loc[oos_start_date:oos_end_date].fillna(0) 
        daily_port_ret = (oos_returns.values @ weights[:n_stocks]).astype(np.float64)

        # Apply transaction costs (assume rebalancing from previous weights)
        tc_rate = (self.env_config.transaction_cost_bps + self.env_config.slippage_bps) / 10_000
        if len(self.results) > 0:
            prev_weights = self.results[-1].oos_weights
            turnover = np.sum(np.abs(weights - prev_weights))
        else:
            turnover = np.sum(np.abs(weights))  # From cash
        tc = turnover * tc_rate

        total_raw_return = np.prod(1 + daily_port_ret) - 1

        # Subtract TC from first day
        if len(daily_port_ret) > 0:
            daily_port_ret[0] -= tc

        total_return = np.prod(1 + daily_port_ret) - 1
        
        return weights, float(total_return), float(total_raw_return), daily_port_ret

    def _obtain_oos_weights(self, raw_action: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
        """Process raw action into valid OOS weights with masking."""
        n = len(raw_action)
        action = raw_action.copy()

        assert raw_action.shape == active_mask.shape, (
            f"Action shape {raw_action.shape} and active mask shape {active_mask.shape} must match."
        )
            
        # Set inactive stocks to very negative (will be zeroed by softmax)
        # action[~active_mask[:n]] = -1e6
        action[~active_mask] = -1e6

        if self.env_config.mode == "long_only":
            exp_a = np.exp(action - np.max(action[active_mask]))
            exp_a[~active_mask] = 0.0
            weights = exp_a / np.sum(exp_a) if np.sum(exp_a) > 1e-10 else np.zeros(n)
            # Apply position limits
            max_pos = self.env_config.max_position_size
            weights = np.clip(weights, 0, max_pos)
            if weights.sum() > 1e-10:
                weights /= weights.sum()
        elif self.env_config.mode == "long_short":
            weights = np.tanh(action)
            weights[~active_mask] = 0.0
            # Center weights to sum to 0
            active_sum = weights[active_mask].sum()
            n_active = active_mask.sum()
            if n_active > 0:
                weights[active_mask] -= active_sum / n_active
            # Leverage limit
            abs_sum = np.abs(weights).sum()
            if abs_sum > 1e-10:
                weights *= self.env_config.leverage_limit / abs_sum
        else:
            weights = np.zeros(n)

        # Pad to max_stocks
        # full_weights = np.zeros(self.env_config.n_stocks, dtype=np.float32)
        # full_weights[:n] = weights
        full_weights = weights

        return full_weights

    ###### Results aggregation ######

    def _aggregate_results(self) -> dict:
        """Compile all window results into summary statistics."""
        if not self.results:
            return {"metrics": {}, "oos_returns": np.array([])}

        # Monthly OOS returns
        oos_returns = np.array([r.oos_return for r in self.results])
        oos_raw_returns = np.array([r.oos_raw_return for r in self.results])

        # Build cumulative portfolio value
        portfolio_values = np.cumprod(np.concatenate([[1.0], 1 + oos_returns]))
        portfolio_values_raw = np.cumprod(np.concatenate([[1.0], 1 + oos_raw_returns]))

        # Concatenate daily returns for more granular metrics
        all_daily = np.concatenate(
            [r.oos_daily_returns for r in self.results if len(r.oos_daily_returns) > 0]
        )
        daily_values = np.cumprod(np.concatenate([[1.0], 1 + all_daily]))

        # Compute metrics on daily returns
        metrics = compute_portfolio_metrics(
            daily_values,
            all_daily,
            risk_free_rate=self.env_config.risk_free_rate,
        )

        # Add walk-forward specific stats
        metrics["n_windows"] = len(self.results)
        metrics["avg_active_stocks"] = np.mean(
            [r.n_active_stocks for r in self.results]
        )
        metrics["avg_train_time"] = np.mean(
            [r.train_time_seconds for r in self.results]
        )
        metrics["pct_positive_months"] = (oos_returns > 0).mean()
        metrics["avg_monthly_return"] = oos_returns.mean()
        metrics["monthly_return_std"] = oos_returns.std()
        metrics["oos_sharpe"] = (oos_returns.mean() / oos_returns.std()) * np.sqrt(12) if oos_returns.std() > 1e-10 else 0.0
        metrics["best_month"] = oos_returns.max()
        metrics["worst_month"] = oos_returns.min()

        # Build date-indexed return series
        oos_dates = [r.oos_start for r in self.results]
        oos_series = pd.Series(oos_returns, index=oos_dates, name="oos_return")

        return {
            "metrics": metrics,
            "oos_returns": oos_returns,
            "oos_raw_returns": oos_raw_returns,
            "oos_series": oos_series,
            "portfolio_values": portfolio_values,
            "portfolio_values_raw": portfolio_values_raw,
            "daily_values": daily_values,
            "daily_returns": all_daily,
            "window_results": self.results,
        }

    def get_weight_history(self) -> pd.DataFrame:
        """
        Get the full history of OOS weights as a DataFrame.

        Returns:
            DataFrame (n_windows, n_stocks) indexed by OOS start date.
        """
        if not self.results:
            return pd.DataFrame()

        dates = [r.oos_start for r in self.results]
        weights = np.stack([r.oos_weights for r in self.results])
        return pd.DataFrame(weights, index=dates,
            columns=self.prices.columns.tolist()
            + [f"_pad_{i}" for i in range(weights.shape[1] - len(self.prices.columns))],
        )