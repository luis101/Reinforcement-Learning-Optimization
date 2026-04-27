"""
Example: Walk-Forward Portfolio Optimization.

Demonstrates the rolling-window strategy on synthetic 27-year data
with IPOs and delistings. Replace the synthetic data section with
your own DataFrame.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# import os
# os.chdir("..")

# from rl_agent.config import (
#    Config, 
#    EnvironmentConfig, FeatureConfig, 
#    NetworkConfig, TrainingConfig, BacktestConfig
# )
from rl_agent.fin_data import download_fin_data, get_sp500
from rl_agent.forwardbacktest import WalkForwardBacktestEngine
from rl_agent.utils import (
    compute_portfolio_metrics,
    format_metrics,
    generate_realistic_universe
)

ticker, sp500 = get_sp500()
prices = pd.read_csv("C:\\Users\\lukas\\Downloads\\prices.csv", index_col=0, parse_dates=True)
engine = WalkForwardBacktestEngine(prices)

def main():
    
    # Obtain data for S&P 500 stocks (replace with your own data loading if needed)
    
    ticker, sp500 = get_sp500()
    # _, _, prices = download_fin_data(ticker=ticker, sp500=sp500)
    #prices.index = prices.index.tz_localize(None)

    prices = pd.read_csv("C:\\Users\\lukas\\Downloads\\prices.csv", index_col=0, parse_dates=True)

    # prices = daily_prices.pivot(index=prices.index, columns="Ticker", values="Price")
    
    # For demonstration, we use the synthetic universe generator. 
    # Replace this with your actual price DataFrame.
    # The DataFrame should have a DatetimeIndex and stock tickers as columns.
    # Each cell contains the adjusted close price, with NaN for non-tradeable days.
    # Generate synthetic data with IPOs and delistings
    # prices = generate_realistic_universe(
    #    n_years=11, n_initial_stocks=80, n_total_stocks=100
    #)

    # Run walk-forward optimization
    engine = WalkForwardBacktestEngine(prices)
    results = engine.run()

    # Analyze results
    print("\n" + "=" * 50)
    print("  Detailed Results")
    print("=" * 50)

    # Annual return breakdown
    oos = results["oos_series"]
    if len(oos) > 12:
        annual = (1 + oos).groupby(oos.index.year).prod() - 1
        print(f"\n  Annual OOS returns:")
        for year, ret in annual.items():
            print(f"    {year}: {ret:+.2%}")

    # Overall metrics
    rl_returns = results["oos_returns"]
    rl_values = np.cumprod(np.concatenate([[1.0], 1 + rl_returns]))
    # pd.DataFrame(rl_values).to_csv("C:\\Users\\lukas\\Downloads\\rl_portfolio_values.csv")
    rl_metrics = compute_portfolio_metrics(rl_values, rl_returns, periods_per_year=12)
    print(format_metrics(rl_metrics))

    # Weight concentration
    weight_df = engine.get_weight_history()
    if len(weight_df) > 0:
        top5_avg = weight_df.apply(
            lambda row: row.nlargest(5).sum(), axis=1
        ).mean()
        n_nonzero_avg = (weight_df.abs() > 0.001).sum(axis=1).mean()
        print(f"\n  Weight statistics:")
        print(f"    Average top-5 concentration: {top5_avg:.2%}")
        print(f"    Average non-zero positions:  {n_nonzero_avg:.0f}")

    weight_df.to_csv("C:\\Users\\lukas\\Downloads\\weights_rl.csv")

    # Compare to equal-weight benchmark
    print("\n  Equal-weight benchmark (same OOS periods):")
    # returns = engine.prices.pct_change().fillna(0)
    bm_prices = engine.prices.mask(engine.prices < 1)
    returns = bm_prices.pct_change()
    returns = returns.apply(lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99)), axis=1).fillna(0)
    
    # bm_returns = []
    period_returns = []
    active_mask = pd.DataFrame()  
    # for r in results["window_results"]:
    #    period_ret = returns.loc[r.oos_start:r.oos_end]
    #    active = r.active_mask[: prices.shape[1]]
    #    ew_daily = period_ret.iloc[:, active].mean(axis=1)
    #    bm_returns.append(np.prod(1 + ew_daily.values) - 1)
    # for r in results["window_results"]:
    #    start = prices.index.get_indexer([r.oos_start], method="ffill")[0]
    #    end = prices.index.get_indexer([r.oos_end], method="ffill")[0]
    #    period_ret = prices.iloc[start:end + 1].pct_change().fillna(0)
    #    active = r.active_mask[: prices.shape[1]]
    #    ew_daily = period_ret.iloc[:, active].mean(axis=1).values
    #    bm_returns.append(np.prod(1 + ew_daily) - 1)
    for r in results["window_results"]:
        active_mask = pd.concat([active_mask, pd.DataFrame([r.active_mask])])
        period_df = returns.loc[r.oos_start:r.oos_end]
        period_returns.append((1 + period_df).prod() - 1)

    masked_returns = pd.DataFrame(period_returns).where(active_mask.values, np.nan)
    bm_returns = masked_returns.mean(axis=1).fillna(0).values

    # Compute equal-weight TC: each period weights drift with returns,
    # and must be rebalanced back to equal weight at the next period start.
    tc_rate_bm = (engine.env_config.transaction_cost_bps + engine.env_config.slippage_bps) / 10_000
    bm_tc_per_period = []
    prev_ew_end = None
    for r in results["window_results"]:
        active = r.active_mask[:prices.shape[1]]
        n_active = int(active.sum())
        if n_active == 0:
            bm_tc_per_period.append(0.0)
            continue
        ew = np.zeros(prices.shape[1])
        ew[active] = 1.0 / n_active
        turnover = np.sum(np.abs(ew - prev_ew_end)) if prev_ew_end is not None else ew.sum()
        bm_tc_per_period.append(turnover * tc_rate_bm)
        cum = (1 + returns.loc[r.oos_start:r.oos_end]).prod().values[:len(ew)]
        end_w = ew * cum
        s = end_w.sum()
        prev_ew_end = end_w / s if s > 1e-10 else ew.copy()
    bm_net_rets = bm_returns - np.array(bm_tc_per_period)

    bm_values = np.cumprod(np.concatenate([[1.0], 1 + bm_returns]))
    pd.DataFrame(bm_values).to_csv("C:\\Users\\lukas\\Downloads\\bm_portfolio_values.csv")
        
    bm_metrics = compute_portfolio_metrics(bm_values, bm_returns, periods_per_year=12)
    print(format_metrics(bm_metrics))

    # Generate HTML dashboard

    # One date per OOS period (monthly data → one date per bar/point)
    oos_dates = pd.DatetimeIndex([r.oos_start for r in results["window_results"]])

    dashboard_kwargs = dict(
        rl_results=results["oos_returns"],
        bm_daily_returns=bm_returns,
        rl_dates=oos_dates,
        weight_history=weight_df if len(weight_df) > 0 else None,
        output_path="rl_backtest_dashboard.html",
        title="RL Portfolio vs Equal-Weight Benchmark",
        periods_per_year=12,
        rl_gross_returns=results["oos_raw_returns"],
        bm_net_returns=bm_net_rets,
    )
 
    # Prefer Plotly (utils.py); fall back to Chart.js (utils_html.py)
    try:
        from rl_agent.utils import generate_dashboard
        generate_dashboard(**dashboard_kwargs)
    except ImportError:
        from rl_agent.utils_html import generate_dashboard
        print("  (Plotly not installed, using Chart.js fallback)")
        generate_dashboard(**dashboard_kwargs)


if __name__ == "__main__":
    main()