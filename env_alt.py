# ==============================
# 2. Environment with variable stock sets
# ==============================

class PortfolioEnv:
    """
    State: (technical indicators + current weights + availability mask) for all max_stocks.
    Action: target weights for all max_stocks; unavailable ones are masked out.
    """
    def __init__(self, price_df, indicators_df, rebalance_freq='M', hold_period=1,
                 allow_short=False, transaction_cost=0.001, window=60,
                 reward_type='sharpe', risk_free_rate=0.0):
        """
        price_df : DataFrame (days x stocks) with NaNs where stock not available.
        indicators_df : multi-index columns, same shape as price_df (days x stocks * n_indicators).
        """
        self.price_df = price_df
        self.indicators_df = indicators_df
        self.max_stocks = price_df.shape[1]
        self.rebalance_freq = rebalance_freq
        self.hold_period = hold_period
        self.allow_short = allow_short
        self.transaction_cost = transaction_cost
        self.window = window
        self.reward_type = reward_type
        self.risk_free_rate = risk_free_rate

        # Determine rebalance indices (dates)
        if rebalance_freq == 'M':
            self.rebalance_dates = price_df.resample('M').apply(lambda x: x.index[-1]).index
        elif rebalance_freq == 'W':
            self.rebalance_dates = price_df.resample('W').apply(lambda x: x.index[-1]).index
        else:
            raise ValueError("rebalance_freq must be 'M' or 'W'")
        # Keep only dates that exist in price_df index
        self.rebalance_dates = [d for d in self.rebalance_dates if d in price_df.index]
        self.max_step = len(self.rebalance_dates) - hold_period
        self.current_step = 0
        self.current_weights = None

        # Pre-compute for each rebalance period:
        # - available stocks (boolean mask)
        # - period returns (for each stock, cumulative return over hold_period)
        # - daily returns for Sharpe calculation
        self.period_info = []
        for i in range(len(self.rebalance_dates) - hold_period):
            start_date = self.rebalance_dates[i]
            end_date = self.rebalance_dates[i + hold_period]
            # Get prices at start and end
            start_prices = price_df.loc[start_date]
            end_prices = price_df.loc[end_date]
            # Available stocks: both start and end prices are non-NaN
            avail_mask = (~start_prices.isna()) & (~end_prices.isna())
            # Cumulative return vector (only for available stocks, others set to 0)
            period_ret = (end_prices - start_prices) / (start_prices + 1e-8)
            period_ret = period_ret.fillna(0).values  # shape (max_stocks,)
            
            # Daily returns for Sharpe (all days between start+1 and end)
            daily_returns = price_df.loc[start_date:end_date].pct_change().iloc[1:].values
            # For unavailable stocks, daily returns will be NaN or 0; we'll mask later
            daily_returns = np.nan_to_num(daily_returns, nan=0.0)
            self.period_info.append({
                'avail_mask': avail_mask.values,
                'period_ret': period_ret,
                'daily_ret': daily_returns  # (n_days, max_stocks)
            })

        # State dimension: for each stock we have (n_indicators + 1 for mask + current weight)
        n_indicators = len(indicators_df.columns.get_level_values(0).unique())
        self.state_dim = (n_indicators + 1 + 1) * self.max_stocks  # +1 mask, +1 current weight
        # Actually current weight is already part of state, so we include it separately.
        # Simpler: state = [indicators (flattened), current_weights, mask] all concatenated.
        self.state_dim = n_indicators * self.max_stocks + self.max_stocks + self.max_stocks

    def reset(self):
        self.current_step = 0
        # Initial weights: equal among available stocks at first rebalance
        avail = self.period_info[0]['avail_mask'] if self.max_step > 0 else np.ones(self.max_stocks, dtype=bool)
        if self.allow_short:
            self.current_weights = np.zeros(self.max_stocks)
        else:
            self.current_weights = np.zeros(self.max_stocks)
            n_avail = np.sum(avail)
            if n_avail > 0:
                self.current_weights[avail] = 1.0 / n_avail
        return self._get_state()

    def _get_state(self):
        """Return state vector: concatenated indicators + current weights + availability mask."""
        if self.current_step >= len(self.period_info):
            return None
        step_info = self.period_info[self.current_step]
        avail_mask = step_info['avail_mask']
        # Get indicators at the rebalance date
        rebalance_date = self.rebalance_dates[self.current_step]
        ind_at_date = self.indicators_df.loc[rebalance_date].values  # shape (n_indicators * max_stocks,)
        # Reshape to (n_indicators, max_stocks) for clarity
        n_indicators = len(self.indicators_df.columns.get_level_values(0).unique())
        ind_reshaped = ind_at_date.reshape(n_indicators, self.max_stocks)
        # Replace NaN indicators with 0 (for unavailable stocks)
        ind_reshaped = np.nan_to_num(ind_reshaped, nan=0.0)
        # Flatten again
        ind_flat = ind_reshaped.flatten()
        # Current weights and mask
        state = np.concatenate([ind_flat, self.current_weights, avail_mask.astype(np.float32)])
        return state.astype(np.float32)

    def step(self, action):
        if self.current_step >= self.max_step:
            return None, 0.0, True, {}

        step_info = self.period_info[self.current_step]
        avail_mask = step_info['avail_mask']
        # Apply mask: set weights of unavailable stocks to zero
        new_weights_full = action.copy()
        new_weights_full[~avail_mask] = 0.0
        # Renormalize to satisfy portfolio constraints over available stocks only
        n_avail = np.sum(avail_mask)
        if n_avail > 0:
            if self.allow_short:
                # Sum to zero over available stocks
                sum_avail = np.sum(new_weights_full[avail_mask])
                new_weights_full[avail_mask] = new_weights_full[avail_mask] - sum_avail / n_avail
            else:
                # Sum to one, non-negative (clip negative weights to 0)
                new_weights_full = np.maximum(new_weights_full, 0.0)
                sum_avail = np.sum(new_weights_full[avail_mask])
                if sum_avail > 1e-8:
                    new_weights_full[avail_mask] = new_weights_full[avail_mask] / sum_avail
                else:
                    # Fallback to equal weight
                    new_weights_full[avail_mask] = 1.0 / n_avail
        # Transaction cost
        turnover = np.sum(np.abs(new_weights_full - self.current_weights))
        cost = self.transaction_cost * turnover

        # Portfolio return over holding period
        period_ret = step_info['period_ret']  # vector for all stocks (0 for unavailable)
        port_return = np.dot(new_weights_full, period_ret) - cost

        # Compute reward
        if self.reward_type == 'return':
            reward = port_return
        elif self.reward_type == 'sharpe':
            # Daily portfolio returns during the period
            daily_returns = step_info['daily_ret']  # (days, max_stocks)
            daily_port_returns = np.dot(daily_returns, new_weights_full)
            # Subtract transaction cost from first day's return
            if len(daily_port_returns) > 0:
                daily_port_returns[0] -= cost
            excess = daily_port_returns - self.risk_free_rate / 252
            if np.std(excess) > 1e-8:
                sharpe = np.mean(excess) / np.std(excess) * np.sqrt(252)
            else:
                sharpe = 0.0
            reward = sharpe
        else:
            raise ValueError("reward_type must be 'return' or 'sharpe'")

        # Update weights and step
        self.current_weights = new_weights_full
        self.current_step += 1
        next_state = self._get_state()
        done = (self.current_step >= self.max_step)
        return next_state, reward, done, {}

    def action_constraint(self, raw_action):
        """Convert raw network output (any real) to preliminary weights (before mask/renorm)."""
        if self.allow_short:
            return np.tanh(raw_action)  # range [-1, 1]
        else:
            # Softmax over all stocks (including unavailable ones; they will be masked later)
            exp_a = np.exp(raw_action - np.max(raw_action))
            return exp_a / (exp_a.sum() + 1e-8)