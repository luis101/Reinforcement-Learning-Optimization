"""
Feature engineering for portfolio state construction.

Computes technical indicators, return statistics, and cross-sectional features
from raw price data. All features are computed in a rolling fashion to avoid
look-ahead bias.
"""

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM as _GaussianHMM

from .config import FeatureConfig


class FeatureConstructor:
    """Computes and caches features from price data."""

    def __init__(self, prices: pd.DataFrame, config: FeatureConfig | None = None,
                 returns: pd.DataFrame | None = None,
                 precomputed_features: dict[str, pd.DataFrame] | None = None):
        """
        Args:
            prices: DataFrame with DatetimeIndex rows (days) and stock and/or asset columns.
                    Values are adjusted close prices.
            config: Feature configuration. Uses defaults if None.
            returns: Pre-computed returns aligned with prices. If provided, skips pct_change.
                     Price-level features (MACD, Bollinger) still use prices.
            precomputed_features: Optional dict of {feature_name: DataFrame} computed externally
                     on a wider date range. When provided, features for this instance are
                     sliced from these (using prices.index) instead of being recomputed.
                     This eliminates per-window rolling-normalization distribution shift
                     between training and OOS application.
        """
        self.config = config or FeatureConfig()
        self.prices = prices
        self.n_stocks = prices.shape[1]
        self.returns = returns if returns is not None else prices.pct_change()
        self._using_precomputed = precomputed_features is not None

        # Pre-compute all features (stored as DataFrames aligned to prices index)
        self._features: dict[str, pd.DataFrame] = {}
        if self._using_precomputed:
            # Slice each globally computed feature DataFrame to this window's date range.
            self._features = {
                name: df.loc[prices.index]
                for name, df in precomputed_features.items()
            }
            # When regime is configured but not present in the precomputed dict, fit
            # the HMM locally on this instance's returns. The walk-forward engine uses
            # this path to keep regime free of full-history look-ahead: globals omit
            # regime, and every training window / OOS query refits on data ≤ its own
            # end date.
            if self.config.use_regime and "regime" not in self._features:
                self._build_regime_features()
        else:
            self._build_features()
            if self.config.use_regime:
                self._build_regime_features()

    def get_state_features(self, date_idx: int) -> np.ndarray:
        """
        Get flattened feature vector for a given date index.

        Returns:
            1-D numpy array of shape (n_features,) with all features for the
            given date, suitable for feeding into the neural network.
        """
        state_features = []
        for _, df in self._features.items():
            row = df.iloc[date_idx].values
            state_features.append(row)
        return np.concatenate(state_features).astype(np.float32)

    def get_stock_features(self, date_idx: int) -> np.ndarray:
        """
        Get per-stock feature matrix for attention-based architectures.

        Returns:
            2-D numpy array of shape (n_stocks, n_features_per_stock).
        """
        per_stock_features = []
        for _, df in self._features.items():
            if df.shape[1] == self.n_stocks:
                per_stock_features.append(df.iloc[date_idx].values.reshape(-1, 1))

        # Stack per-stock features: (n_stocks, n_per_stock_features)
        stock_features = np.hstack(per_stock_features).astype(np.float32)
        return stock_features

    def get_market_features(self, date_idx: int) -> np.ndarray:
        """
        Get market-level (non-per-stock) features.

        Returns:
            1-D numpy array of market-wide features.
        """
        market_features = []
        for _, df in self._features.items():
            if df.shape[1] != self.n_stocks:
                market_features.append(df.iloc[date_idx].values)
        if not market_features:
            return np.array([], dtype=np.float32)
        return np.concatenate(market_features).astype(np.float32)

    @property
    def n_features(self) -> int:
        """Total number of features in the flattened state vector."""
        return sum(df.shape[1] for df in self._features.values())

    @property
    def n_stock_features(self) -> int:
        """Number of per-stock features (for attention input)."""
        return sum(1 for df in self._features.values() if df.shape[1] == self.n_stocks)

    @property
    def n_market_features(self) -> int:
        """Number of market-level features."""
        return sum(
            df.shape[1]
            for df in self._features.values()
            if df.shape[1] != self.n_stocks
        )

    @property
    def valid_start_idx(self) -> int:
        """First index where all features are valid (no NaNs from warmup).

        When features were precomputed on a wider date range, the warmup is
        already absorbed there, so we can start from the beginning of the slice.
        """
        if self._using_precomputed:
            return 0
        return self.config.normalize_window + max(
            max(self.config.return_windows),
            self.config.macd_slow + self.config.macd_signal,
            max(self.config.volatility_windows),
            self.config.bollinger_period
        )

    # Feature computation

    def _build_features(self):
        """Compute all features and store them."""
        ret = self.returns

        # Rolling returns at multiple horizons
        for w in self.config.return_windows:
            self._features[f"ret_{w}d"] = ret.rolling(w).sum()

        # Rolling volatility
        for w in self.config.volatility_windows:
            vol = ret.rolling(w).std() * np.sqrt(252)
            self._features[f"vol_{w}d"] = vol

        # RSI
        self._features["rsi"] = self._compute_rsi(ret, self.config.rsi_period)

        # PPO (Percentage Price Oscillator) — scale-invariant version of MACD
        ppo_line, ppo_signal = self._compute_ppo(
            self.prices, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        self._features["ppo"] = ppo_line - ppo_signal

        # Bollinger Band position
        self._features["bbpos"] = self._compute_bollinger_position(
            self.prices, self.config.bollinger_period, self.config.bollinger_std
        )

        # Cross-sectional momentum
        if self.config.use_cross_sectional_rank:
            for w in [21, 63, 128, 252]:  # Use standard return windows for cross-sectional rank
                rolling_ret = ret.rolling(w).sum()
                self._features[f"xsmom_{w}d"] = rolling_ret.rank(
                    axis=1, pct=True
                )

        # Market return and volatility (equal-weight portfolio as proxy)
        mkt_ret = ret.mean(axis=1)
        for w in [21, 252]:  # Short-term and long-term market features
            self._features[f"mkt_ret_{w}d"] = pd.DataFrame(
                mkt_ret.rolling(w).sum().values,
                index=self.prices.index,
                columns=[f"mkt_ret_{w}d"],
            )
            self._features[f"mkt_vol_{w}d"] = pd.DataFrame(
                (mkt_ret.rolling(w).std() * np.sqrt(252)).values,
                index=self.prices.index,
                columns=[f"mkt_vol_{w}d"],
            )

        # Dispersion: cross-sectional stdev of returns (short-term and long-term)
        for w in [21, 252]:
            self._features[f"dispersion_{w}d"] = pd.DataFrame(
                ret.std(axis=1).rolling(w).mean().values,
                index=self.prices.index,
                columns=[f"dispersion_{w}d"],
            )

        # Normalize all features
        self._normalize_features()

        # Fill remaining NaNs (from warmup) with 0
        for name in self._features:
            self._features[name] = self._features[name].fillna(0.0)

    def _normalize_features(self):
        """Apply rolling normalization to avoid look-ahead bias."""
        method = self.config.normalize_method
        window = self.config.normalize_window

        for name, df in self._features.items():
            if name == "regime":
                continue  # already in [0, 1], probabilities need no normalization
            if method == "zscore":
                rolling_mean = df.rolling(window, min_periods=1).mean()
                rolling_std = df.rolling(window, min_periods=1).std().replace(0, 1)
                self._features[name] = (df - rolling_mean) / rolling_std
            elif method == "minmax":
                rolling_min = df.rolling(window, min_periods=1).min()
                rolling_max = df.rolling(window, min_periods=1).max()
                denom = (rolling_max - rolling_min).replace(0, 1)
                self._features[name] = (df - rolling_min) / denom

            # Winsorize extreme values to [-5, 5] for stability
            self._features[name] = self._features[name].clip(-5, 5)

    # Technical indicator implementations

    @staticmethod
    def _compute_rsi(returns: pd.DataFrame, period: int) -> pd.DataFrame:
        """Relative Strength Index, normalized to [-1, 1] range."""
        gain = returns.clip(lower=0)
        loss = (-returns).clip(lower=0)
        avg_gain = gain.ewm(span=period, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        # Normalize from [0, 100] to [-1, 1]
        return (rsi - 50) / 50

    @staticmethod
    def _compute_ppo(prices: pd.DataFrame, fast: int, slow: int, signal: int
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Percentage Price Oscillator — MACD expressed as % of slow EMA.
        Scale-invariant: a $10 stock and a $200 stock produce comparable values."""
        ema_fast = prices.ewm(span=fast, min_periods=fast).mean()
        ema_slow = prices.ewm(span=slow, min_periods=slow).mean()
        ppo_line = (ema_fast - ema_slow) / ema_slow.abs().replace(0, 1e-10) * 100
        signal_line = ppo_line.ewm(span=signal, min_periods=signal).mean()
        return ppo_line, signal_line

    @staticmethod
    def _compute_bollinger_position(prices: pd.DataFrame, period: int, num_std: float
                                    ) -> pd.DataFrame:
        """Position within Bollinger Bands, normalized to roughly [-1, 1]."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + num_std * std
        lower = sma - num_std * std
        band_width = (upper - lower).replace(0, 1e-10)
        position = (prices - lower) / band_width * 2 - 1  # Map to [-1, 1]
        return position.clip(-2, 2)


    # Regime detection

    def _build_regime_features(self) -> None:
        """
        Fit a Gaussian HMM on market observations and store predicted
        next-step regime probabilities as market features.

        Observations (T, 2):
          equal-weight daily market return  (T,)
          rolling realised vol of market return, window = regime_vol_window  (T,)
        Both are 1-dimensional series stacked column-wise → shape (T, 2).

        Output: DataFrame (T, n_regimes) stored as self._features["regime"].
        Probabilities are in [0, 1] and sum to 1 — no further normalisation needed.
        """

        vol_w = self.config.regime_vol_window

        # Equal-weight market return: mean across stocks at each day → shape (T,)
        mkt_ret: np.ndarray = self.returns.mean(axis=1).values

        # Realised vol: rolling std of mkt_ret → shape (T,)
        mkt_vol: np.ndarray = (
            pd.Series(mkt_ret).rolling(vol_w).std().fillna(0).values
        )

        obs = np.column_stack([mkt_ret, mkt_vol]).astype(np.float64)  # (T, 2)

        model = _GaussianHMM(
            n_components=self.config.n_regimes,
            covariance_type="diag",
            n_iter=100,
            tol=0.0001,
            # random_state=42,
        )
        model.fit(obs)

        # posteriors[t] = P(z_t | o_0, … o_t)  shape (T, K)
        posteriors = model.predict_proba(obs)

        # next-step prediction: A^T @ posterior_t  shape (T, K)
        next_proba = posteriors @ model.transmat_.T
        next_proba /= next_proba.sum(axis=1, keepdims=True) + 1e-300

        cols = [f"regime_{k}" for k in range(self.config.n_regimes)]
        self._features["regime"] = pd.DataFrame(
            next_proba, index=self.prices.index, columns=cols).fillna(1.0 / self.config.n_regimes)
