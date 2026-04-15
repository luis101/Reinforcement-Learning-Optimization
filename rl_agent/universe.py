"""
Dynamic stock universe  accounting for IPOs and delistings.

Manages a fixed-size tensor representation where stocks can enter (IPO)
and exit (delist) over time. Uses masking to handle the variable universe
while keeping network dimensions constant.
"""
import numpy as np
import pandas as pd


class DynamicUniverse:
    """
    Tracks which stocks are tradeable at each point in time, providing masking utilities.

    The key idea: Always work with a fixed-size array of size `max_stocks`. 
    Each stock is assigned a permanent slot index. At any given date, 
    only a subset of slots are "active" (the stock is listed and is tradeable). 
    Inactive slots get zero weight and are masked out of the loss computation.

    Stocks are identified by their column position in the original price DataFrame. 
    A stock is considered active on dates where its price is not NaN.
    """

    def __init__(self, prices: pd.DataFrame, max_stocks: int | None = None):
        """
        Args:
            prices: DataFrame (n_days, n_stocks) where NaN indicates the
                    stock is not yet listed or has been delisted. Columns
                    are stock tickers/identifiers.
            max_stocks: Fixed tensor size. Defaults to total unique stocks
                       in the DataFrame. Must be >= number of columns.
        """
        self.prices = prices
        self.n_total_stocks = prices.shape[1]
        self.max_stocks = max_stocks or self.n_total_stocks
        self.stock_names = list(prices.columns)

        assert self.max_stocks >= self.n_total_stocks, (
            f"max_stocks ({self.max_stocks}) must be >= total stocks ({self.n_total_stocks})"
        )

        # Build the active mask: (n_days, n_total_stocks) boolean
        # True = stock is tradeable on that day
        self._active_mask = ~prices.isna()

        # Precompute IPO and delist dates for each stock
        self.first_dates: dict[str, pd.Timestamp] = {}
        self.delist_dates: dict[str, pd.Timestamp | None] = {}
        self._compute_valid_dates()

    def _compute_valid_dates(self):
        """Determine IPO and delist dates for each stock."""
        for col in self.prices.columns:
            valid = self.prices[col].dropna()
            if len(valid) == 0:
                continue
            self.first_dates[col] = valid.index[0]

            # A stock is "delisted" if its last valid date is before the
            # end of the full price series (with some buffer for missing data)
            last_valid = valid.index[-1]
            series_end = self.prices.index[-1]
            # If last valid price is more than 10 trading days before series end,
            # treat as delisted
            if (series_end - last_valid).days > 10:
                self.delist_dates[col] = last_valid
            else:
                self.delist_dates[col] = None  # Still active at end

    @property
    def summary(self) -> str:
        """Print summary statistics about the universe."""
        n_active = self._active_mask.sum(axis=1)
        n_ipo = sum(1 for d in self.first_dates.values() if d > self.prices.index[0])
        n_delist = sum(1 for d in self.delist_dates.values() if d is not None)

        lines = [
            f"Dynamic Universe Summary:",
            f"  Total unique stocks:        {self.n_total_stocks}",
            f"  Max concurrent stocks:      {n_active.max()}",
            f"  Min concurrent stocks:      {n_active.min()}",
            f"  Mean concurrent stocks:     {n_active.mean():.0f}",
            f"  Entrants:                   {n_ipo}",
            f"  Delistings:                 {n_delist}",
            f"  Date range:                 {self.prices.index[0].date()} to {self.prices.index[-1].date()}",
            f"  Data size:                  {self.max_stocks}",
        ]
        return "\n".join(lines)