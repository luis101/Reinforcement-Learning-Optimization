"""
Example: Walk-Forward Portfolio Optimization.

Demonstrates the rolling-window strategy on synthetic 27-year data
with IPOs and delistings. Replace the synthetic data section with
your own DataFrame.
"""
import numpy as np
import pandas as pd


from .config import (
    Config, 
    EnvironmentConfig, FeatureConfig, 
    NetworkConfig, TrainingConfig,
)
from .forwardbacktest import WalkForwardBacktestEngine
from .universe import DynamicUniverse
from .utils import (
    compute_portfolio_metrics,
    format_metrics,
)

###### Generate synthetic universe (replace with real data) ######

def generate_realistic_universe(
    n_years: int = 10,
    n_initial_stocks: int = 80,
    n_total_stocks: int = 100,
    annual_ipo_rate: float = 0.05,
    annual_delist_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic price data with IPOs and delistings.

    Mimics a realistic stock universe where:
    - ~80 stocks exist at the start
    - ~5% of universe size IPO each year
    - ~3% of universe size delist each year
    - Delisted stocks may have negative terminal returns
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252
    dates = pd.bdate_range(start="2015-01-02", periods=n_days, freq="B")
    tickers = [f"STOCK_{i:03d}" for i in range(n_total_stocks)]

    # Initialize all prices as NaN
    prices = pd.DataFrame(
        np.nan, index=dates, columns=tickers, dtype=np.float64
    )

    # Market factor
    market_vol = 0.16 / np.sqrt(252)
    market_returns = rng.normal(0.07 / 252, market_vol, n_days)

    # Generate lifecycle events
    stock_params = {}
    active_set = set()

    for i in range(n_total_stocks):
        ticker = tickers[i]

        # Initial stocks start on day 0
        if i < n_initial_stocks:
            ipo_day = 0
        else:
            # IPOs happen uniformly across the full period
            ipo_day = rng.integers(252, n_days - 252)

        # Some stocks delist
        if rng.random() < annual_delist_rate * n_years / n_total_stocks:
            # Delist at least 1 year after IPO, at least 1 year before end
            earliest_delist = ipo_day + 252
            if earliest_delist < n_days - 252:
                delist_day = rng.integers(earliest_delist, n_days - 252)
            else:
                delist_day = None
        else:
            delist_day = None

        beta = rng.uniform(0.5, 1.5)
        alpha = rng.normal(0.0, 0.03 / 252)
        idio_vol = rng.uniform(0.20, 0.50) / np.sqrt(252)

        stock_params[ticker] = {
            "ipo_day": ipo_day,
            "delist_day": delist_day,
            "beta": beta,
            "alpha": alpha,
            "idio_vol": idio_vol,
        }

    # Generate returns and prices
    for ticker, params in stock_params.items():
        ipo = params["ipo_day"]
        delist = params["delist_day"]
        end = delist if delist is not None else n_days

        n_active = end - ipo
        if n_active <= 0:
            continue

        returns = (
            params["alpha"]
            + params["beta"] * market_returns[ipo:end]
            + rng.normal(0, params["idio_vol"], n_active)
        )

        # If delisting, add a negative terminal return
        if delist is not None:
            delist_loss = rng.uniform(-0.50, -0.10)
            returns[-1] = delist_loss

        stock_prices = 100 * np.cumprod(1 + returns)
        prices.loc[dates[ipo:end], ticker] = stock_prices

    return prices


def main():
    # Generate synthetic data with IPOs and delistings
    prices = generate_realistic_universe(
        n_years=27, n_initial_stocks=80, n_total_stocks=100
    )

    # Show universe stats
    universe = DynamicUniverse(prices)
    print(universe.summary())

    # Run walk-forward optimization
    engine = WalkForwardBacktestEngine(prices)
    results = engine.run()

    # Analyze results
    print("\n" + "=" * 50)
    print("  Detailed Results")
    print("=" * 50)

    # Monthly return statistics
    oos = results["oos_series"]
    print(f"\n  Monthly return distribution:")
    print(f"    Mean:   {oos.mean():+.4f}")
    print(f"    Median: {oos.median():+.4f}")
    print(f"    Std:    {oos.std():.4f}")
    print(f"    Skew:   {oos.skew():.3f}")
    print(f"    Kurt:   {oos.kurtosis():.3f}")

    # Annual return breakdown
    if len(oos) > 12:
        annual = (1 + oos).groupby(oos.index.year).prod() - 1
        print(f"\n  Annual OOS returns:")
        for year, ret in annual.items():
            print(f"    {year}: {ret:+.2%}")

    # Weight concentration
    weight_df = engine.get_weight_history()
    if len(weight_df) > 0:
        top5_avg = weight_df.apply(
            lambda row: row.nlargest(5).sum(), axis=1
        ).mean()
        n_nonzero_avg = (weight_df.abs() > 0.001).sum(axis=1).mean()
        print(f"\n  Weight statistics:")
        print(f"    Avg top-5 concentration: {top5_avg:.2%}")
        print(f"    Avg non-zero positions:  {n_nonzero_avg:.0f}")

    # ------------------------------------------------------------------
    # 5. Compare to equal-weight benchmark
    # ------------------------------------------------------------------
    print("\n  Equal-weight benchmark (same OOS periods):")
    bm_returns = []
    for r in results["window_results"]:
        start = prices.index.get_indexer([r.oos_start], method="ffill")[0]
        end = prices.index.get_indexer([r.oos_end], method="ffill")[0]
        period_ret = prices.iloc[start:end + 1].pct_change().fillna(0)
        active = r.active_mask[: prices.shape[1]]
        ew_daily = period_ret.iloc[:, active].mean(axis=1).values
        bm_returns.append(np.prod(1 + ew_daily) - 1)

    bm_values = np.cumprod(np.concatenate([[1.0], 1 + np.array(bm_returns)]))
    bm_metrics = compute_portfolio_metrics(bm_values)
    print(format_metrics(bm_metrics))


if __name__ == "__main__":
    main()