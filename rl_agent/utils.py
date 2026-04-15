"""
Utility functions for evaluation, metrics, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt

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
        "win_rate": ("{:.2%}", "Win Rate"),
        "var_95": ("{:.4f}", "VaR (95%)"),
        "cvar_95": ("{:.4f}", "CVaR (95%)"),
    }

    for key, (template, label) in fmt.items():
        if key in metrics:
            lines.append(f"  {label:<28s} {template.format(metrics[key])}")

    return "\n".join(lines)


def compute_benchmark_returns(prices: pd.DataFrame, rebalance_dates: list[int]
                              ) -> np.ndarray:
    """
    Compute equal-weight benchmark portfolio values at rebalancing points.
    """
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