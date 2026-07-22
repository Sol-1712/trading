import pandas as pd
import numpy as np
from pathlib import Path
import json

from IPython.display import display

from trading.backtester.engine import BacktestConfig
from trading.backtester.portfolio import TradeLog
from .metrics import CoreStats, ReturnMetrics, RiskMetrics, CostMetrics, TradeMetrics
from .metrics.utils import METRICS, SECONDS_TO_PERIODS_24_7

_SECTION_TITLES = {
    "returns": "Returns",
    "risk": "Risk",
    "cost": "Costs",
    "trade": "Trade",
}

_DISPLAY_LABELS: dict[str, str] = {
    "cagr": "CAGR",
    "net_return": "Net Return",
    "gross_return": "Gross Return",
    "mean_bar_return": "Mean Bar Return",
    "median_bar_return": "Median Bar Return",
    "hit_rate_all": "Hit Rate (Bar)",
    "avg_bar_win": "Avg Bar Win",
    "avg_bar_loss": "Avg Bar Loss",
    "avg_bar_win_loss_ratio": "Avg Bar Win/Loss",
    "largest_bar_return": "Largest Bar Return",
    "smallest_bar_return": "Smallest Bar Return",
    "skew": "Skew",
    "kurtosis": "Kurtosis",
    "volatility": "Volatility",
    "downside_deviation": "Downside Deviation",
    "max_drawdown": "Max Drawdown",
    "avg_drawdown": "Avg Drawdown",
    "max_drawdown_duration": "Max DD Duration",
    "avg_drawdown_duration": "Avg DD Duration",
    "time_in_drawdown": "Time in Drawdown",
    "longest_losing_streak": "Longest Losing Streak",
    "sharpe": "Sharpe (per bar)",
    "annualised_sharpe": "Sharpe (annualised)",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "var_95": "VaR 95%",
    "var_99": "VaR 99%",
    "cvar_95": "CVaR 95%",
    "cvar_99": "CVaR 99%",
    "long_sharpe": "Long Sharpe",
    "short_sharpe": "Short Sharpe",
    "total_fee_return": "Total Fee Drag",
    "total_funding_return": "Total Funding Drag",
    "total_cost_return": "Total Costs",
    "total_fee_pct_of_net": "Fee % of Net",
    "funding_pct_of_net": "Funding % of Net",
    "total_cost_pct_of_net": "Cost % of Net",
    "cost_to_gross_ratio": "Cost to Gross Ratio",
    "fee_drag_sharpe": "Fee Drag (Sharpe)",
    "funding_drag_sharpe": "Funding Drag (Sharpe)",
    "avg_fee_per_bar": "Avg Fee per Bar",
    "avg_fee_per_trade": "Avg Fee per Trade",
    "annualised_turnover": "Annualised Turnover",
    "pct_bars_paying_funding": "% Bars Paying Funding",
    "num_trades": "Num Trades",
    "hit_rate_trade": "Hit Rate (Trade)",
    "expectancy": "Expectancy",
    "profit_factor": "Profit Factor",
    "avg_position_size": "Avg Position Size",
    "max_position_size": "Max Position Size",
    "avg_long_size": "Avg Long Size",
    "avg_short_size": "Avg Short Size",
    "time_long": "Time Long",
    "time_short": "Time Short",
    "time_flat": "Time Flat",
    "time_in_market": "Time in Market",
    "avg_holding_period": "Avg Holding Period",
    "largest_win": "Largest Win",
    "largest_loss": "Largest Loss",
    "long_pnl_pct": "Long PnL %",
    "short_pnl_pct": "Short PnL %",
}

_FORMAT: dict[str, str] = {
    "cagr": "pct_dollar",
    "net_return": "pct_dollar",
    "gross_return": "pct_dollar",
    "mean_bar_return": "pct_dollar",
    "median_bar_return": "pct_dollar",
    "avg_bar_win": "pct_dollar",
    "avg_bar_loss": "pct_dollar",
    "largest_bar_return": "pct_dollar",
    "smallest_bar_return": "pct_dollar",
    "expectancy": "pct_dollar",
    "max_drawdown": "pct_dollar",
    "avg_drawdown": "pct_dollar",
    "var_95": "pct_dollar",
    "var_99": "pct_dollar",
    "cvar_95": "pct_dollar",
    "cvar_99": "pct_dollar",
    "total_fee_return": "pct_dollar",
    "total_funding_return": "pct_dollar",
    "total_cost_return": "pct_dollar",
    "avg_fee_per_bar": "pct_dollar",
    "largest_win": "pct_dollar",
    "largest_loss": "pct_dollar",
    "volatility": "pct",
    "downside_deviation": "pct",
    "time_in_drawdown": "pct",
    "hit_rate_all": "pct",
    "hit_rate_trade": "pct",
    "total_fee_pct_of_net": "pct",
    "funding_pct_of_net": "pct",
    "total_cost_pct_of_net": "pct",
    "pct_bars_paying_funding": "pct",
    "avg_position_size": "pct",
    "max_position_size": "pct",
    "avg_long_size": "pct",
    "avg_short_size": "pct",
    "time_long": "pct",
    "time_short": "pct",
    "time_flat": "pct",
    "time_in_market": "pct",
    "long_pnl_pct": "pct",
    "short_pnl_pct": "pct",
    "max_drawdown_duration": "bars",
    "avg_drawdown_duration": "bars",
    "avg_holding_period": "bars",
    "longest_losing_streak": "bars",
    "num_trades": "count",
    "annualised_turnover": "turnover",
}

class PerformanceReport:
    """
    Aggregates return, risk, cost, and trade metrics for a backtest.

    Builds a shared CoreStats substrate from portfolio history, then
    exposes metric groups and helpers to summarise, save, or display.

    Parameters
    ----------
    portfolio_history : pd.DataFrame
        Bar-indexed portfolio snapshots (equity, PnL components, etc.).
    trade_log : TradeLog
        Closed/open trade records used by TradeMetrics.
    rf : float, default 0.0
        Annualised simple risk-free rate (e.g. 0.05 = 5%).
    """

    def __init__(
        self, portfolio_history: pd.DataFrame,
        trade_log: TradeLog,
        rf: float = 0.0,
    ) -> None:

        self.core = CoreStats(portfolio_history, rf=rf)

        self.returns  = ReturnMetrics(self.core)
        self.risk     = RiskMetrics(self.core)
        self.cost     = CostMetrics(self.core)
        self.trade    = TradeMetrics(self.core, trade_log)
        

    def summary(self) -> dict[str, float]:
        """
        Flatten every exported scalar metric across all metric groups.

        Returns
        -------
        dict[str, float]
            Mapping of metric name → value from returns, risk, cost, and trade.
        """
        return {
            **self.returns.to_dict(),
            **self.risk.to_dict(),
            **self.cost.to_dict(),
            **self.trade.to_dict(),
        }

    def save(self, run_dir: Path) -> None:
        """
        Write ``summary()`` to ``run_dir / report.json``.

        Parameters
        ----------
        run_dir : Path
            Destination run directory.
        """
        with open(run_dir / "report.json", "w") as f:
            json.dump(self.summary(), f, indent=2)

    def display(self, backtest_config: BacktestConfig) -> None:
        """
        Render a formatted metrics report in the notebook / IPython frontend.

        Parameters
        ----------
        backtest_config : BacktestConfig
            Used for core metadata (symbol, dates, capital) and dollar scaling.
        """
        display_report(self.summary(), backtest_config)


def display_report(report: dict[str, float], backtest_config: BacktestConfig) -> None:
    """
    Display core run metadata plus metric sections from a flat report dict.

    Parameters
    ----------
    report : dict[str, float]
        Flat metric map as produced by ``PerformanceReport.summary()``.
    backtest_config : BacktestConfig
        Supplies symbol, interval, date range, and initial capital.
    """
    capital = backtest_config.initial_capital
    display(_core_section(report, backtest_config))
    for section_key, metric_keys in METRICS.items():
        title = _SECTION_TITLES[section_key]
        display(_metric_section(title, metric_keys, report, capital))


# ---------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------

def _interval_label(interval_min: int) -> str:
    seconds = interval_min * 60
    return SECONDS_TO_PERIODS_24_7.get(seconds, ("1H", 365 * 24))[0]


def _core_section(report: dict[str, float], config: BacktestConfig) -> pd.DataFrame:
    data = config.data
    start_ts = pd.Timestamp(data.start)
    end_ts = pd.Timestamp(data.end)
    interval_min = int(data.interval)
    starting_capital = float(config.initial_capital)
    net_return = report.get("net_return", 0.0)
    final_capital = starting_capital * (1 + net_return)
    n_bars = int((end_ts - start_ts).total_seconds() / (interval_min * 60))

    return _make_df("Core", {
        "Symbol": str(data.symbol),
        "Interval": _interval_label(interval_min),
        "Start": start_ts.strftime("%Y-%m-%d %H:%M"),
        "End": end_ts.strftime("%Y-%m-%d %H:%M"),
        "Duration": f"{(end_ts - start_ts).days + 1} days",
        "Bars": str(n_bars),
        "Starting Capital": _fmt_dollar(starting_capital),
        "Final Capital": _fmt_dollar(final_capital),
    })


def _metric_section(
    title: str,
    keys: list[str],
    report: dict[str, float],
    capital: float,
) -> pd.DataFrame:
    rows = {
        _DISPLAY_LABELS.get(key, key): _format_metric(key, report[key], capital)
        for key in keys
        if key in report
    }
    return _make_df(title, rows)


def _make_df(title: str, data: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame(data.items(), columns=["Metric", "Value"]).set_index("Metric")
    df.index.name = title
    return df


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _isnan(v: float) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}" if not _isnan(v) else "N/A"


def _fmt_float(v: float) -> str:
    return f"{v:.3f}" if not _isnan(v) else "N/A"


def _fmt_bars(v: float) -> str:
    return f"{v:.2f} bars" if not _isnan(v) else "N/A"


def _fmt_dollar(v: float) -> str:
    return f"${v:,.2f}" if not _isnan(v) else "N/A"


def _fmt_count(v: float) -> str:
    return f"{int(v)}" if not _isnan(v) else "N/A"


def _fmt_turnover(v: float) -> str:
    return f"{v:.1f}x" if not _isnan(v) else "N/A"


def _pct_and_dollar(v: float, capital: float) -> str:
    if _isnan(v):
        return "N/A"
    return f"{v:.2%}  (${v * capital:,.2f})"


def _format_metric(key: str, value: float, capital: float) -> str:
    fmt = _FORMAT.get(key, "float")
    if fmt == "pct":
        return _fmt_pct(value)
    if fmt == "pct_dollar":
        return _pct_and_dollar(value, capital)
    if fmt == "bars":
        if _isnan(value):
            return "N/A"
        if key == "longest_losing_streak":
            return f"{int(value)} bars"
        return _fmt_bars(value)
    if fmt == "count":
        return _fmt_count(value)
    if fmt == "turnover":
        return _fmt_turnover(value)
    return _fmt_float(value)