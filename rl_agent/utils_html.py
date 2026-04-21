"""
Utility functions for evaluation, metrics, and visualization.
"""
import numpy as np
import pandas as pd
from typing import Any

import json


def compute_portfolio_metrics(
    portfolio_values: np.ndarray,
    daily_returns: np.ndarray | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """
    Compute comprehensive portfolio performance metrics.

    Args:
        portfolio_values: Array of portfolio values over time.
        daily_returns: Optional pre-computed daily returns.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Trading periods per year (252 for daily).

    Returns:
        Dictionary of named metrics.
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

    # Win rate
    win_rate = (daily_returns > 0).mean()

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
        "win_rate": win_rate,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_volatility": downside_vol,
        "n_periods": n,
    }

def compute_benchmark_returns(prices: pd.DataFrame,rebalance_dates: list[int]) -> np.ndarray:
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


###### Dashboard

def generate_dashboard(
        rl_results: dict,
        bm_daily_returns: np.ndarray,
        rl_dates: pd.DatetimeIndex | None = None,
        weight_history: pd.DataFrame | None = None,
        output_path: str = "dashboard.html",
        title: str = "RL Portfolio vs Equal-Weight Benchmark"
        ) -> str:
    """
    Generate a self-contained HTML dashboard comparing the RL agent
    strategy to an equal-weight benchmark.

    The dashboard includes:
    - Cumulative returns time-series (RL vs benchmark)
    - Drawdown chart for both strategies
    - Rolling 12-month Sharpe ratio comparison
    - Monthly return bar chart (RL vs benchmark)
    - Side-by-side performance metrics table
    - Portfolio weight concentration over time (if weight_history provided)

    Uses Chart.js via CDN. The output is a single .html file with all
    data embedded as JSON — no external dependencies at runtime.

    Args:
        rl_results: Dictionary from WalkForwardEngine.run() or similar,
                    containing at least 'daily_returns' and optionally
                    'oos_series'.
        bm_daily_returns: 1-D array of daily benchmark returns aligned
                         to the same dates as the RL daily returns.
        rl_dates: DatetimeIndex for RL daily returns. If None, uses
                  integer indices.
        weight_history: DataFrame (n_windows, n_stocks) of portfolio
                       weights, date-indexed.
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
 
    # Cumulative values
    rl_cum = np.concatenate([[1.0], np.cumprod(1 + rl_daily)])
    bm_cum = np.concatenate([[1.0], np.cumprod(1 + bm_daily)])

    # Drawdowns
    rl_peak = np.maximum.accumulate(rl_cum)
    bm_peak = np.maximum.accumulate(bm_cum)
    rl_dd = (rl_cum - rl_peak) / rl_peak
    bm_dd = (bm_cum - bm_peak) / bm_peak

    # Rolling 252-day Sharpe
    window = min(252, len(rl_daily))
    rl_roll_sharpe = rolling_sharpe(rl_daily, window)
    bm_roll_sharpe = rolling_sharpe(bm_daily, window)

    # Date labels
    if rl_dates is not None:
        date_labels = [d.strftime("%Y-%m-%d") for d in rl_dates[:len(rl_daily)]]
        # Prepend one label for the initial value
        if len(rl_dates) > 0:
            cum_labels = [rl_dates[0].strftime("%Y-%m-%d")] + date_labels
        else:
            cum_labels = ["0"] + date_labels
    else:
        date_labels = list(range(len(rl_daily)))
        cum_labels = list(range(len(rl_daily) + 1))

    # Monthly returns (from oos_series if available, else aggregate)
    rl_monthly, bm_monthly, monthly_labels = _build_monthly_returns(
        rl_daily, bm_daily, rl_dates
    )

    # Metrics for both
    rl_metrics = compute_portfolio_metrics(rl_cum, rl_daily)
    bm_metrics = compute_portfolio_metrics(bm_cum, bm_daily)

    # Weight concentration data
    wt_data = None
    if weight_history is not None and len(weight_history) > 0:
        wt_data = _build_weight_concentration(weight_history)

    # Build the JSON data structure for embedding in HTML
    data = {
        "cumLabels": _thin_labels(cum_labels, 800),
        "rlCum": _thin_array(rl_cum.tolist(), 800),
        "bmCum": _thin_array(bm_cum.tolist(), 800),
        "ddLabels": _thin_labels(date_labels, 800),
        "rlDD": _thin_array(rl_dd[1:].tolist(), 800),
        "bmDD": _thin_array(bm_dd[1:].tolist(), 800),
        "sharpeLabels": _thin_labels(date_labels, 400),
        "rlSharpe": _thin_array(rl_roll_sharpe.tolist(), 400),
        "bmSharpe": _thin_array(bm_roll_sharpe.tolist(), 400),
        "monthlyLabels": monthly_labels,
        "rlMonthly": rl_monthly,
        "bmMonthly": bm_monthly,
        "rlMetrics": _format_metrics_dict(rl_metrics),
        "bmMetrics": _format_metrics_dict(bm_metrics),
        "title": title,
    }
    if wt_data is not None:
        data["wtLabels"] = wt_data["labels"]
        data["wtTop5"] = wt_data["top5"]
        data["wtTop10"] = wt_data["top10"]
        data["wtNonzero"] = wt_data["n_nonzero"]

    html = _DASHBOARD_HTML.replace("__DATA_JSON__", json.dumps(data))

    with open(output_path, "w") as f:
        f.write(html)

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

def _build_monthly_returns(
    rl_daily: np.ndarray,
    bm_daily: np.ndarray,
    dates: pd.DatetimeIndex | None,
) -> tuple[list, list, list]:
    """Aggregate daily returns into monthly returns."""
    n = len(rl_daily)
    if dates is not None and len(dates) >= n:
        months = pd.Series(rl_daily, index=dates[:n])
        bm_months = pd.Series(bm_daily, index=dates[:n])
        rl_monthly = ((1 + months).groupby(months.index.to_period("M")).prod() - 1)
        bm_monthly = ((1 + bm_months).groupby(bm_months.index.to_period("M")).prod() - 1)
        labels = [str(p) for p in rl_monthly.index]
        return rl_monthly.tolist(), bm_monthly.tolist(), labels
    else:
        # No dates: chunk by ~21 trading days
        chunk = 21
        rl_m, bm_m, labels = [], [], []
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            rl_m.append(float(np.prod(1 + rl_daily[i:end]) - 1))
            bm_m.append(float(np.prod(1 + bm_daily[i:end]) - 1))
            labels.append(f"M{len(labels) + 1}")
        return rl_m, bm_m, labels

def _build_weight_concentration(
    weight_history: pd.DataFrame,
) -> dict:
    """Compute weight concentration metrics from weight history."""
    abs_w = weight_history.abs().values
    # Top-5 and top-10 via partition (vectorized)
    n_cols = abs_w.shape[1]
    top5_k = min(5, n_cols)
    top10_k = min(10, n_cols)
    top5 = np.partition(abs_w, -top5_k, axis=1)[:, -top5_k:].sum(axis=1)
    top10 = np.partition(abs_w, -top10_k, axis=1)[:, -top10_k:].sum(axis=1)
    n_nonzero = (abs_w > 0.001).sum(axis=1)

    if hasattr(weight_history.index, "strftime"):
        labels = [d.strftime("%Y-%m") for d in weight_history.index]
    else:
        labels = list(range(len(weight_history)))

    return {
        "labels": labels,
        "top5": top5.tolist(),
        "top10": top10.tolist(),
        "n_nonzero": n_nonzero.tolist(),
    }

def _thin_labels(labels: list, max_points: int) -> list:
    """Downsample labels for chart rendering."""
    if len(labels) <= max_points:
        return labels
    step = len(labels) / max_points
    return [labels[int(i * step)] for i in range(max_points)]

def _thin_array(arr: list, max_points: int) -> list:
    """Downsample numeric array for chart rendering."""
    if len(arr) <= max_points:
        return [round(v, 6) for v in arr]
    step = len(arr) / max_points
    return [round(arr[int(i * step)], 6) for i in range(max_points)]

def _format_metrics_dict(metrics: dict) -> dict:
    """Format metrics for JSON embedding."""
    fmt_map = {
        "total_return": "{:.2%}",
        "annualized_return": "{:.2%}",
        "annualized_volatility": "{:.2%}",
        "sharpe_ratio": "{:.3f}",
        "sortino_ratio": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "max_dd_duration": "{:.0f}",
        "calmar_ratio": "{:.3f}",
        "var_95": "{:.4f}",
        "cvar_95": "{:.4f}",
        "downside_volatility": "{:.2%}",
    }
    label_map = {
        "total_return": "Total return",
        "annualized_return": "Annualized return",
        "annualized_volatility": "Annualized volatility",
        "sharpe_ratio": "Sharpe ratio",
        "sortino_ratio": "Sortino ratio",
        "max_drawdown": "Max drawdown",
        "max_dd_duration": "Max DD duration (days)",
        "calmar_ratio": "Calmar ratio",
        "var_95": "VaR (95%)",
        "cvar_95": "CVaR (95%)",
        "downside_volatility": "Downside volatility",
    }
    result = {}
    for key in fmt_map:
        if key in metrics:
            result[label_map.get(key, key)] = fmt_map[key].format(metrics[key])
    return result


###### Dashboard HTML template

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Portfolio Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
  :root {
    --bg: #0e1117; --bg2: #161b22; --bg3: #1c2129;
    --border: #2d333b; --text: #e6edf3; --text2: #8b949e;
    --rl: #58a6ff; --rl-bg: rgba(88,166,255,0.08);
    --bm: #f0883e; --bm-bg: rgba(240,136,62,0.08);
    --green: #3fb950; --red: #f85149;
    --accent: #1f6feb;
    --radius: 10px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'DM Sans',system-ui,sans-serif; background:var(--bg);
         color:var(--text); padding:24px; max-width:1400px; margin:0 auto; }
  h1 { font-size:22px; font-weight:600; margin-bottom:6px; letter-spacing:-0.3px; }
  .subtitle { color:var(--text2); font-size:13px; margin-bottom:28px; }
  .grid { display:grid; gap:16px; margin-bottom:16px; }
  .grid-2 { grid-template-columns:1fr 1fr; }
  .grid-3 { grid-template-columns:1fr 1fr 1fr; }
  .card { background:var(--bg2); border:1px solid var(--border);
          border-radius:var(--radius); padding:20px; }
  .card-title { font-size:13px; font-weight:500; color:var(--text2);
                text-transform:uppercase; letter-spacing:0.6px; margin-bottom:14px; }
  .chart-wrap { position:relative; height:280px; }
  .chart-wrap canvas { width:100%!important; height:100%!important; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; padding:8px 10px; color:var(--text2); font-weight:500;
       border-bottom:1px solid var(--border); font-size:11px;
       text-transform:uppercase; letter-spacing:0.5px; }
  td { padding:7px 10px; border-bottom:1px solid var(--border); }
  td:nth-child(2), td:nth-child(3) { text-align:right; font-family:'JetBrains Mono',monospace;
       font-size:12px; }
  .tag { display:inline-block; padding:2px 8px; border-radius:4px; font-size:11px;
         font-weight:500; font-family:'JetBrains Mono',monospace; }
  .tag-rl { background:var(--rl-bg); color:var(--rl); }
  .tag-bm { background:var(--bm-bg); color:var(--bm); }
  .legend { display:flex; gap:20px; margin-bottom:16px; }
  .legend-item { display:flex; align-items:center; gap:6px; font-size:13px; color:var(--text2); }
  .legend-dot { width:10px; height:10px; border-radius:50%; }
  .kpi-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
             gap:12px; margin-bottom:20px; }
  .kpi { background:var(--bg2); border:1px solid var(--border); border-radius:var(--radius);
         padding:16px; }
  .kpi-label { font-size:11px; color:var(--text2); text-transform:uppercase;
               letter-spacing:0.5px; margin-bottom:4px; }
  .kpi-value { font-size:22px; font-weight:600; font-family:'JetBrains Mono',monospace;
               letter-spacing:-0.5px; }
  .kpi-sub { font-size:11px; color:var(--text2); margin-top:2px; }
  .better { color:var(--green); }
  .worse { color:var(--red); }
  @media(max-width:900px) { .grid-2,.grid-3 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<script>
const D = __DATA_JSON__;

// --- Chart.js defaults ---
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#2d333b';
Chart.defaults.font.family = "'DM Sans', system-ui, sans-serif";
Chart.defaults.font.size = 11;
Chart.defaults.plugins.legend.display = false;
Chart.defaults.elements.point.radius = 0;
Chart.defaults.elements.point.hoverRadius = 4;
Chart.defaults.elements.line.borderWidth = 1.8;
Chart.defaults.animation = false;
const gridOpts = { color:'rgba(45,51,59,0.6)' };
const tickOpts = { maxTicksLimit:8, font:{size:10} };

// --- KPI row ---
const rlRet = D.rlMetrics['Annualized return'] || '—';
const bmRet = D.bmMetrics['Annualized return'] || '—';
const rlSharpe = D.rlMetrics['Sharpe ratio'] || '—';
const bmSharpe = D.bmMetrics['Sharpe ratio'] || '—';
const rlDD = D.rlMetrics['Max drawdown'] || '—';
const bmDD = D.bmMetrics['Max drawdown'] || '—';
const rlVol = D.rlMetrics['Annualized volatility'] || '—';
const bmVol = D.bmMetrics['Annualized volatility'] || '—';

function kpiColor(rl, bm, lowerBetter) {
  const a = parseFloat(rl), b = parseFloat(bm);
  if (isNaN(a)||isNaN(b)) return '';
  return lowerBetter ? (a<b?'better':'worse') : (a>b?'better':'worse');
}

document.write(`
<h1>${D.title}</h1>
<div class="subtitle">Out-of-sample walk-forward backtest</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:var(--rl)"></div>RL agent</div>
  <div class="legend-item"><div class="legend-dot" style="background:var(--bm)"></div>Equal-weight benchmark</div>
</div>
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Ann. return</div>
    <div class="kpi-value ${kpiColor(rlRet,bmRet,false)}">${rlRet}</div>
    <div class="kpi-sub">Benchmark: ${bmRet}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Sharpe ratio</div>
    <div class="kpi-value ${kpiColor(rlSharpe,bmSharpe,false)}">${rlSharpe}</div>
    <div class="kpi-sub">Benchmark: ${bmSharpe}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Max drawdown</div>
    <div class="kpi-value ${kpiColor(rlDD,bmDD,true)}">${rlDD}</div>
    <div class="kpi-sub">Benchmark: ${bmDD}</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Volatility</div>
    <div class="kpi-value ${kpiColor(rlVol,bmVol,true)}">${rlVol}</div>
    <div class="kpi-sub">Benchmark: ${bmVol}</div>
  </div>
</div>
`);

// --- Cumulative returns ---
document.write(`
<div class="grid grid-2">
<div class="card">
  <div class="card-title">Cumulative returns</div>
  <div class="chart-wrap"><canvas id="cumChart"></canvas></div>
</div>
<div class="card">
  <div class="card-title">Drawdown</div>
  <div class="chart-wrap"><canvas id="ddChart"></canvas></div>
</div>
</div>
<div class="grid grid-2">
<div class="card">
  <div class="card-title">Rolling 12-month Sharpe ratio</div>
  <div class="chart-wrap"><canvas id="sharpeChart"></canvas></div>
</div>
<div class="card">
  <div class="card-title">Monthly returns</div>
  <div class="chart-wrap"><canvas id="monthlyChart"></canvas></div>
</div>
</div>
`);

// Metrics table + optional weight concentration
const hasWeights = D.wtLabels && D.wtLabels.length > 0;
const gridClass = hasWeights ? 'grid-2' : 'grid-2';
document.write(`<div class="grid ${gridClass}">`);

// Metrics table
const metricKeys = Object.keys(D.rlMetrics);
let tableRows = metricKeys.map(k => {
  const rv = D.rlMetrics[k], bv = D.bmMetrics[k];
  return `<tr><td>${k}</td><td><span class="tag tag-rl">${rv}</span></td><td><span class="tag tag-bm">${bv}</span></td></tr>`;
}).join('');
document.write(`
<div class="card">
  <div class="card-title">Performance metrics</div>
  <table><thead><tr><th>Metric</th><th>RL agent</th><th>Benchmark</th></tr></thead>
  <tbody>${tableRows}</tbody></table>
</div>
`);

if (hasWeights) {
  document.write(`
  <div class="card">
    <div class="card-title">Weight concentration</div>
    <div class="chart-wrap"><canvas id="wtChart"></canvas></div>
  </div>
  `);
}
document.write('</div>');

// --- Create charts ---
new Chart(document.getElementById('cumChart'), {
  type:'line',
  data:{
    labels:D.cumLabels,
    datasets:[
      { label:'RL agent', data:D.rlCum, borderColor:'#58a6ff',
        backgroundColor:'rgba(88,166,255,0.05)', fill:true },
      { label:'Equal-weight', data:D.bmCum, borderColor:'#f0883e',
        backgroundColor:'rgba(240,136,62,0.05)', fill:true },
    ]
  },
  options:{ scales:{ x:{grid:gridOpts,ticks:tickOpts}, y:{grid:gridOpts,ticks:tickOpts} },
            interaction:{mode:'index',intersect:false},
            plugins:{tooltip:{callbacks:{label:ctx=>ctx.dataset.label+': '+ctx.parsed.y.toFixed(3)}}} }
});

new Chart(document.getElementById('ddChart'), {
  type:'line',
  data:{
    labels:D.ddLabels,
    datasets:[
      { label:'RL agent', data:D.rlDD, borderColor:'#58a6ff', backgroundColor:'rgba(88,166,255,0.12)', fill:true },
      { label:'Equal-weight', data:D.bmDD, borderColor:'#f0883e', backgroundColor:'rgba(240,136,62,0.12)', fill:true },
    ]
  },
  options:{ scales:{ x:{grid:gridOpts,ticks:tickOpts}, y:{grid:gridOpts,ticks:{...tickOpts,callback:v=>(v*100).toFixed(0)+'%'}} },
            interaction:{mode:'index',intersect:false},
            plugins:{tooltip:{callbacks:{label:ctx=>ctx.dataset.label+': '+(ctx.parsed.y*100).toFixed(2)+'%'}}} }
});

new Chart(document.getElementById('sharpeChart'), {
  type:'line',
  data:{
    labels:D.sharpeLabels,
    datasets:[
      { label:'RL agent', data:D.rlSharpe, borderColor:'#58a6ff' },
      { label:'Equal-weight', data:D.bmSharpe, borderColor:'#f0883e' },
    ]
  },
  options:{ scales:{ x:{grid:gridOpts,ticks:tickOpts}, y:{grid:gridOpts,ticks:tickOpts} },
            interaction:{mode:'index',intersect:false} }
});

// Monthly returns — color RL bars by sign
const rlColors = D.rlMonthly.map(v => v >= 0 ? 'rgba(63,185,80,0.7)' : 'rgba(248,81,73,0.7)');
new Chart(document.getElementById('monthlyChart'), {
  type:'bar',
  data:{
    labels:D.monthlyLabels,
    datasets:[
      { label:'RL agent', data:D.rlMonthly, backgroundColor:rlColors, borderRadius:2 },
      { label:'Equal-weight', data:D.bmMonthly, backgroundColor:'rgba(240,136,62,0.35)',
        borderColor:'rgba(240,136,62,0.6)', borderWidth:1, borderRadius:2 },
    ]
  },
  options:{ scales:{ x:{grid:gridOpts,ticks:{...tickOpts,maxTicksLimit:12}},
                     y:{grid:gridOpts,ticks:{...tickOpts,callback:v=>(v*100).toFixed(1)+'%'}} },
            interaction:{mode:'index',intersect:false},
            plugins:{tooltip:{callbacks:{label:ctx=>ctx.dataset.label+': '+(ctx.parsed.y*100).toFixed(2)+'%'}}} }
});

// Weight concentration chart
if (hasWeights) {
  new Chart(document.getElementById('wtChart'), {
    type:'line',
    data:{
      labels:D.wtLabels,
      datasets:[
        { label:'Top 5 weight', data:D.wtTop5, borderColor:'#58a6ff', fill:false },
        { label:'Top 10 weight', data:D.wtTop10, borderColor:'#8b5cf6', fill:false },
        { label:'Non-zero positions', data:D.wtNonzero, borderColor:'#f0883e',
          yAxisID:'y2', borderDash:[4,3] },
      ]
    },
    options:{ scales:{
      x:{grid:gridOpts,ticks:tickOpts},
      y:{grid:gridOpts,ticks:{...tickOpts,callback:v=>(v*100).toFixed(0)+'%'},position:'left',
         title:{display:true,text:'Weight share',color:'#8b949e',font:{size:10}}},
      y2:{grid:{display:false},ticks:tickOpts,position:'right',
          title:{display:true,text:'Count',color:'#8b949e',font:{size:10}}}
    }, interaction:{mode:'index',intersect:false} }
  });
}
</script>
</body>
</html>"""