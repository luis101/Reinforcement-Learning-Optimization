"""
Utility functions for evaluation, metrics, and visualization.
Dashboard generation uses Plotly.
"""

import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###### Portfolio metrics

def compute_portfolio_metrics(
    portfolio_values: np.ndarray, daily_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.0, periods_per_year: int = 252,
    ) -> dict[str, float]:
    """
    Compute comprehensive portfolio performance metrics

    Args:
        portfolio_values: Array of portfolio values over time.
        daily_returns: Optional pre-computed daily returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Dictionary of metrics.
    """
    if daily_returns is None:
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]

    n = len(daily_returns)
    if n == 0:
        return {"total_return": 0.0}

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = daily_returns - daily_rf

    # Basic return metrics
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    ann_return = (1 + total_return) ** (periods_per_year / n) - 1

    # Risk metrics
    ann_vol = daily_returns.std() * np.sqrt(periods_per_year)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = (
        downside_returns.std() * np.sqrt(periods_per_year)
        if len(downside_returns) > 0
        else 0.0
    )

    # Sharpe and Sortino
    sharpe = (
        (excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year))
        if excess_returns.std() > 1e-10
        else 0.0
    )
    sortino = (
        (excess_returns.mean() / downside_vol * np.sqrt(periods_per_year))
        if downside_vol > 1e-10
        else 0.0
    )

    # Drawdown analysis
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (cummax - portfolio_values) / cummax
    max_drawdown = drawdowns.max()

    # Find max drawdown duration
    in_dd = drawdowns > 0
    dd_durations = []
    current_duration = 0
    for d in in_dd:
        if d:
            current_duration += 1
        else:
            if current_duration > 0:
                dd_durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        dd_durations.append(current_duration)
    max_dd_duration = max(dd_durations) if dd_durations else 0

    # Calmar ratio
    calmar = ann_return / max_drawdown if max_drawdown > 1e-10 else 0.0

    # Tail metrics
    var_95 = np.percentile(daily_returns, 5)
    cvar_95 = daily_returns[daily_returns <= var_95].mean() if (daily_returns <= var_95).any() else var_95

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "max_dd_duration": max_dd_duration,
        "calmar_ratio": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_volatility": downside_vol,
        "n_periods": n,
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Format metrics dictionary into a readable string."""
    lines = []
    fmt = {
        "total_return": ("{:.2%}", "Total Return"),
        "annualized_return": ("{:.2%}", "Annualized Return"),
        "annualized_volatility": ("{:.2%}", "Annualized Volatility"),
        "sharpe_ratio": ("{:.3f}", "Sharpe Ratio"),
        "sortino_ratio": ("{:.3f}", "Sortino Ratio"),
        "max_drawdown": ("{:.2%}", "Max Drawdown"),
        "max_dd_duration": ("{:.0f}", "Max DD Duration (days)"),
        "calmar_ratio": ("{:.3f}", "Calmar Ratio"),
        "var_95": ("{:.4f}", "VaR (95%)"),
        "cvar_95": ("{:.4f}", "CVaR (95%)"),
    }

    for key, (template, label) in fmt.items():
        if key in metrics:
            lines.append(f"  {label:<28s} {template.format(metrics[key])}")

    return "\n".join(lines)


###### Data generation 

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


###### Evaluation and benchmarks

def compute_benchmark_returns(
        prices: pd.DataFrame, rebalance_dates: list[int], returns: pd.DataFrame = None
        ) -> np.ndarray:
    """
    Compute equal-weight benchmark portfolio values at rebalancing points.
    """
    if returns is None:
        returns = prices.pct_change().fillna(0)
    values = [1.0]

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1]
        period_ret = returns.iloc[start + 1 : end + 1].mean(axis=1)
        period_total = np.prod(1 + period_ret.values) - 1
        values.append(values[-1] * (1 + period_total))

    return np.array(values)


def evaluate_agent(agent, env, n_episodes: int = 1) -> dict[str, Any]:
    """
    Evaluate the agent on an environment without training.

    Returns:
        Dictionary with metrics, portfolio values, and weight history.
    """
    all_metrics = []
    all_values = []
    all_weights = []

    for ep in range(n_episodes):
        flat_state, stock_feats, market_feats = env.reset()

        episode_weights = []
        done = False

        while not done:
            action, _, _ = agent.select_action(
                stock_feats, market_feats, deterministic=True
            )

            result = env.step(action)
            stock_feats = result.stock_features
            market_feats = result.market_features
            done = result.done

            episode_weights.append(result.info.get("new_weights", action))

        values = env.portfolio_value_series
        daily_rets = np.diff(values) / values[:-1]
        metrics = compute_portfolio_metrics(values, daily_rets)

        all_metrics.append(metrics)
        all_values.append(values)
        all_weights.append(episode_weights)

    # Average metrics across episodes
    avg_metrics = {}
    for key in all_metrics[0]:
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return {
        "metrics": avg_metrics,
        "portfolio_values": all_values,
        "weight_history": all_weights,
    }


def plot_training_results(
        train_stats: list[dict], eval_results: dict | None = None,
        save_path: str | None = None
        ):
    """
    Plot training curves and evaluation results.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RL Portfolio Training Results", fontsize=14)

    # Policy loss
    ax = axes[0, 0]
    ax.plot([s["policy_loss"] for s in train_stats])
    ax.set_title("Policy Loss")
    ax.set_xlabel("Update")

    # Value loss
    ax = axes[0, 1]
    ax.plot([s["value_loss"] for s in train_stats])
    ax.set_title("Value Loss")
    ax.set_xlabel("Update")

    # Entropy
    ax = axes[0, 2]
    ax.plot([s["entropy"] for s in train_stats])
    ax.set_title("Policy Entropy")
    ax.set_xlabel("Update")

    # KL divergence
    ax = axes[1, 0]
    ax.plot([s["approx_kl"] for s in train_stats])
    ax.set_title("Approx KL Divergence")
    ax.set_xlabel("Update")
    ax.axhline(y=0.03, color="r", linestyle="--", alpha=0.5, label="KL target")
    ax.legend()

    # Clip fraction
    ax = axes[1, 1]
    ax.plot([s["clip_fraction"] for s in train_stats])
    ax.set_title("Clip Fraction")
    ax.set_xlabel("Update")

    # Portfolio value (if eval results provided)
    if eval_results and "portfolio_values" in eval_results:
        ax = axes[1, 2]
        for i, vals in enumerate(eval_results["portfolio_values"]):
            ax.plot(vals, label=f"Episode {i}")
        ax.set_title("Portfolio Value")
        ax.set_xlabel("Rebalancing Step")
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


###### Dashboard

# Color palette — consistent across all charts
_RL_COLOR = "#58a6ff"
_BM_COLOR = "#f0883e"
_GREEN = "#3fb950"
_RED = "#f85149"
_PURPLE = "#8b5cf6"
_BG = "#0e1117"
_BG2 = "#161b22"
_GRID = "rgba(45,51,59,0.6)"
_TEXT = "#e6edf3"
_TEXT2 = "#8b949e"

def generate_dashboard(
    rl_results: np.ndarray,
    bm_daily_returns: np.ndarray,
    rl_dates: pd.DatetimeIndex | None = None,
    weight_history: pd.DataFrame | None = None,
    output_path: str = "dashboard.html",
    title: str = "RL Portfolio vs Equal-Weight Benchmark",
) -> str:
    """
    Generate an interactive HTML dashboard comparing the RL agent
    to an equal-weight benchmark using Plotly.
 
    Panels:
    1. Cumulative returns (RL vs benchmark)
    2. Drawdown chart
    3. Rolling 12-month Sharpe ratio
    4. Monthly returns comparison 
    5. Performance metrics tables (RL and benchmark side by side)
    6. Weight concentration over time (if weight_history provided)
 
    Args:
        rl_results: Dict with at least 'daily_returns' key (1-D array).
        bm_daily_returns: 1-D array of daily benchmark returns.
        rl_dates: DatetimeIndex for RL daily returns.
        weight_history: DataFrame (n_windows, n_stocks), date-indexed.
        output_path: Where to write the HTML file.
        title: Dashboard title.
 
    Returns:
        The output_path string.
    """
 
    rl_daily = np.asarray(rl_results, dtype=np.float64)
    bm_daily = np.asarray(bm_daily_returns, dtype=np.float64)
 
    # Assert that return arrays are 1-D and have compatible lengths
    if rl_daily.ndim != 1 or bm_daily.ndim != 1:
        raise ValueError("Daily returns must be 1-D arrays.")
    if len(rl_daily) != len(bm_daily):
        raise ValueError("Daily returns must have the same length.")
 
    # Date axis
    if rl_dates is not None and len(rl_dates) >= len(rl_daily):
        dates = rl_dates[:len(rl_daily)]
    else:
        dates = pd.RangeIndex(len(rl_daily))
 
    # Cumulative values (n+1 points: starts at 1.0)
    rl_cum = np.cumprod(np.concatenate([[1.0], 1 + rl_daily]))
    bm_cum = np.cumprod(np.concatenate([[1.0], 1 + bm_daily]))
    if isinstance(dates, pd.DatetimeIndex):
        cum_dates = dates.insert(0, dates[0] - pd.Timedelta(days=1))
    else:
        cum_dates = pd.RangeIndex(len(rl_daily) + 1)

    # Compute drawdown from peak for a cumulative value series
    rl_peak = np.maximum.accumulate(rl_cum)
    bm_peak = np.maximum.accumulate(bm_cum)
    rl_dd = (rl_cum - rl_peak) / np.where(rl_peak > 0, rl_peak, 1) 
    bm_dd = (bm_cum - bm_peak) / np.where(bm_peak > 0, bm_peak, 1)
 
    # Sharpe ratio (rolling 12-month window)
    rl_sharpe = rolling_sharpe(rl_daily, window=252)
    bm_sharpe = rolling_sharpe(bm_daily, window=252)
    
    # Monthly returns (calendar month, not rolling 21-day)
    rl_monthly, bm_monthly, month_labels = _build_monthly_returns(
        rl_daily, bm_daily, dates
    )
 
    # Metrics
    rl_metrics = compute_portfolio_metrics(rl_cum, rl_daily)
    bm_metrics = compute_portfolio_metrics(bm_cum, bm_daily)
 
    # Layout
    has_weights = weight_history is not None and len(weight_history) > 0
    n_rows = 4 if has_weights else 3
 
    specs = [
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "table"}, {"type": "table"}],
    ]
    subtitles = [
        "Cumulative returns", "Drawdown",
        "Rolling 12-month Sharpe ratio", "Monthly returns",
        "RL agent metrics", "Benchmark metrics",
    ]
    row_heights = [0.30, 0.30, 0.25]
    if has_weights:
        specs.append([{"type": "xy"}, {"type": "xy"}])
        subtitles.extend(["Weight concentration", "Non-zero positions"])
        row_heights = [0.27, 0.27, 0.22, 0.24]
 
    fig = make_subplots(
        rows=n_rows, cols=2,
        subplot_titles=subtitles,
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        row_heights=row_heights,
    )
 
    # ---- 1. Cumulative returns + Drawdown ----
    fig.add_trace(go.Scatter(
        x=cum_dates, y=rl_cum, name="RL agent",
        line=dict(color=_RL_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cum_dates, y=bm_cum, name="Equal-weight",
        line=dict(color=_BM_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(240,136,62,0.06)",
    ), row=1, col=1)
 
    fig.add_trace(go.Scatter(
        x=cum_dates, y=rl_dd, name="RL agent",
        line=dict(color=_RL_COLOR, width=1.2),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.12)",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=cum_dates, y=bm_dd, name="Equal-weight",
        line=dict(color=_BM_COLOR, width=1.2),
        fill="tozeroy", fillcolor="rgba(240,136,62,0.12)",
        showlegend=False,
    ), row=1, col=2)
 
    # ---- 2. Rolling Sharpe + Monthly returns ----
    fig.add_trace(go.Scatter(
        x=dates, y=rl_sharpe, name="RL agent",
        line=dict(color=_RL_COLOR, width=1.3), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bm_sharpe, name="Equal-weight",
        line=dict(color=_BM_COLOR, width=1.3), showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=_TEXT2,
                  line_width=0.5, row=2, col=1)
 
    rl_bar_colors = [_GREEN if v >= 0 else _RED for v in rl_monthly]
    fig.add_trace(go.Bar(
        x=month_labels, y=rl_monthly, name="RL agent",
        marker_color=rl_bar_colors, opacity=0.75, showlegend=False,
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=month_labels, y=bm_monthly, name="Equal-weight",
        marker_color=_BM_COLOR, opacity=0.4, showlegend=False,
    ), row=2, col=2)
 
    # ---- 3. Metrics tables ----
    rl_table = _metrics_table(rl_metrics)
    bm_table = _metrics_table(bm_metrics)
 
    table_header = dict(
        fill_color=_BG2, font=dict(color=_TEXT, size=11),
        line_color=_GRID, align="left",
    )
    table_cells_base = dict(
        fill_color=_BG,
        font=dict(color=_TEXT, size=11, family="JetBrains Mono, monospace"),
        line_color=_GRID, align=["left", "right"], height=24,
    )
 
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], **table_header),
        cells=dict(values=[rl_table["labels"], rl_table["values"]],
                   **table_cells_base),
    ), row=3, col=1)
 
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], **table_header),
        cells=dict(values=[bm_table["labels"], bm_table["values"]],
                   **table_cells_base),
    ), row=3, col=2)
 
    # ---- 4. Weight concentration (optional) ----
    if has_weights:
        wt = _build_weight_concentration(weight_history)
        wt_dates = weight_history.index
 
        fig.add_trace(go.Scatter(
            x=wt_dates, y=wt["top5"], name="Top 5",
            line=dict(color=_RL_COLOR, width=1.3), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=wt_dates, y=wt["top10"], name="Top 10",
            line=dict(color=_PURPLE, width=1.3), showlegend=False,
        ), row=4, col=1)
        fig.update_yaxes(tickformat=".0%", row=4, col=1)
 
        fig.add_trace(go.Scatter(
            x=wt_dates, y=wt["n_nonzero"], name="Non-zero",
            line=dict(color=_BM_COLOR, width=1.3, dash="dot"),
            showlegend=False,
        ), row=4, col=2)
 
    # ---- Global layout ----
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=_TEXT)),
        height=350 * n_rows,
        template="plotly_dark",
        paper_bgcolor=_BG, plot_bgcolor=_BG2,
        font=dict(family="DM Sans, system-ui, sans-serif",
                  color=_TEXT, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5, font=dict(size=12),
        ),
        margin=dict(l=60, r=30, t=80, b=40),
        hovermode="x unified",
        barmode="group",
    )
 
    axis_style = dict(
        gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(size=10, color=_TEXT2)
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(tickformat=".1%", row=2, col=2)
 
    # Write self-contained HTML
    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
        },
    )
 
    print(f"  Dashboard saved to {output_path}")
    return output_path


###### Internal helper functions

def rolling_sharpe(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """
    Rolling annualized Sharpe ratio.
    """
    result = np.full(len(returns), np.nan)
    if len(returns) < window:
        return result
    
    cs = np.insert(np.cumsum(returns), 0, 0)
    cs2 = np.insert(np.cumsum(returns ** 2), 0, 0)
 
    ends = np.arange(window - 1, len(returns))
    starts = ends - window + 1
 
    # Rolling sums via cumsum differences (vectorized)
    s = cs[ends] - cs[starts]
    s2 = cs2[ends] - cs2[starts]
 
    mean = s / window
    var = np.maximum((s2 / window) - (mean ** 2), 0)
    std = np.sqrt(var)
 
    valid = std > 1e-10
    sharpe = np.where(valid, (mean / std) * np.sqrt(252), 0.0)
    result[window - 1:] = sharpe
 
    return 

def _build_monthly_returns(rl_daily: np.ndarray, bm_daily: np.ndarray,
                           dates: pd.DatetimeIndex | pd.RangeIndex
                           ) -> tuple[list, list, list]:
    """Aggregate daily returns into monthly returns."""
    n = len(rl_daily)
    if isinstance(dates, pd.DatetimeIndex) and len(dates) >= n:
        rl_s = pd.Series(rl_daily, index=dates[:n])
        bm_s = pd.Series(bm_daily, index=dates[:n])
        periods = rl_s.index.to_period("M")
        rl_monthly = (1 + rl_s).groupby(periods).prod() - 1
        bm_monthly = (1 + bm_s).groupby(periods).prod() - 1
        labels = [str(p) for p in rl_monthly.index]
        return rl_monthly.tolist(), bm_monthly.tolist(), labels
    else:
        chunk = 21
        rl_m, bm_m, labels = [], [], []
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            rl_m.append(float(np.prod(1 + rl_daily[i:end]) - 1))
            bm_m.append(float(np.prod(1 + bm_daily[i:end]) - 1))
            labels.append(f"M{len(labels) + 1}")
        return rl_m, bm_m, labels
    
def _metrics_table(metrics: dict[str, float]) -> str:
    """Format metrics dictionary into a list."""
    fmt = {
        "total_return": ("{:.2%}", "Total Return"),
        "annualized_return": ("{:.2%}", "Annualized Return"),
        "annualized_volatility": ("{:.2%}", "Annualized Volatility"),
        "sharpe_ratio": ("{:.3f}", "Sharpe Ratio"),
        "sortino_ratio": ("{:.3f}", "Sortino Ratio"),
        "max_drawdown": ("{:.2%}", "Max Drawdown"),
        "max_dd_duration": ("{:.0f}", "Max DD Duration (days)"),
        "calmar_ratio": ("{:.3f}", "Calmar Ratio"),
        "var_95": ("{:.4f}", "VaR (95%)"),
        "cvar_95": ("{:.4f}", "CVaR (95%)"),
    }
    labels, values = [], []
    for key, (template, label) in fmt.items():
        if key in metrics:
            labels.append(label)
            values.append(template.format(metrics[key]))

    return { "labels": labels, "values": values}

def _build_weight_concentration(weight_history: pd.DataFrame) -> dict:
    """Compute weight concentration metrics from weight history."""
    abs_w = weight_history.abs().values
    n_cols = abs_w.shape[1]
    top5_k = min(5, n_cols)
    top10_k = min(10, n_cols)
    top5 = np.partition(abs_w, -top5_k, axis=1)[:, -top5_k:].sum(axis=1)
    top10 = np.partition(abs_w, -top10_k, axis=1)[:, -top10_k:].sum(axis=1)
    n_nonzero = (abs_w > 0.001).sum(axis=1)
    return {"top5": top5, "top10": top10, "n_nonzero": n_nonzero}