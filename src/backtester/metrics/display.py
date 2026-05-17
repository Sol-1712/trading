import numpy as np
import pandas as pd
from backtester.metrics.performance_report import PerformanceReport
from IPython.display import display


def display_report(report: PerformanceReport, symbol: str | None) -> None:
    """
    Displays a formatted PerformanceReport in a Jupyter notebook.

    Parameters
    ----------
    report : PerformanceReport
        Instantiated PerformanceReport object.
    """
    cr = report.core
    r  = report.returns
    rk = report.risk
    c  = report.cost
    p  = report.position


    capital   = cr.equity[0]

    avg_win_loss, expectancy, hit_rate_trade = r.avg_win_loss_ratio_expectancy

    def _isnan(v):     return isinstance(v, float) and np.isnan(v)
    def _fmt_pct(v):   return f"{v:.2%}"      if not _isnan(v) else "N/A"
    def _fmt_float(v): return f"{v:.3f}"      if not _isnan(v) else "N/A"
    def _fmt_bars(v):  return f"{v:.2f} bars" if not _isnan(v) else "N/A"
    def _fmt_dollar(v): return f"${v:,.2f}"    if not _isnan(v) else "N/A"

    def _pct_and_dollar(v):
        """Format as both % and $."""
        if _isnan(v):
            return "N/A"
        return f"{v:.2%}  (${v * capital:,.2f})"
    
    def _make_df(title: str, data: dict) -> pd.DataFrame:
        df = pd.DataFrame(
            data.items(),
            columns=["Metric", "Value"]
        ).set_index("Metric")
        df.index.name = title
        return df

    sections = [
        _make_df("Core", {
            "Symbol":           str(symbol if symbol else None),
            "Interval":         cr.freq,
            "Start":            str(cr.pnl_df.index[0].strftime("%Y-%m-%d %H:%M")),
            "End":              str(cr.pnl_df.index[-1].strftime("%Y-%m-%d %H:%M")),
            "Duration":         f"{(cr.pnl_df.index[-1] - cr.pnl_df.index[0]).days + 1} days",
            "Bars":             f"{cr.n_bars}",
            "Starting Capital": _fmt_dollar(cr.equity[0]),
            "Final Capital":    _fmt_dollar(cr.equity[-1])

        }),

        _make_df("Returns", {
            "CAGR":                _pct_and_dollar(r.cagr),
            "Net Return":          _pct_and_dollar(r.net_return),
            "Gross Return":        _pct_and_dollar(r.gross_return),
            "Sharpe (per bar)":    _fmt_float(r.sharpe),
            "Sharpe (annualised)": _fmt_float(r.annualised_sharpe),
            "Sortino":             _fmt_float(r.sortino),
            "Hit Rate (Trade)":    _fmt_pct(hit_rate_trade),
            "Avg Win/Loss":        _fmt_float(avg_win_loss),
            "Expectancy":          _pct_and_dollar(expectancy),
            "Profit Factor":       _fmt_float(r.profit_factor),
            "Skew":                _fmt_float(r.skew),
            "Kurtosis":            _fmt_float(r.kurtosis),
        }),

        _make_df("Risk", {
            "Volatility":            _fmt_pct(rk.volatility),
            "Max Drawdown":          _pct_and_dollar(rk.max_drawdown),
            "Max DD Duration":       _fmt_bars(rk.max_drawdown_duration),
            "Avg DD Duration":       _fmt_bars(rk.avg_drawdown_duration),
            "Time in Drawdown":      _fmt_pct(rk.time_in_drawdown),
            "Calmar":                _fmt_float(rk.calmar),
            "VaR 95%":               _pct_and_dollar(rk.var_95),
            "VaR 99%":               _pct_and_dollar(rk.var_99),
            "CVaR 95%":              _pct_and_dollar(rk.cvar_95),
            "CVaR 99%":              _pct_and_dollar(rk.cvar_99),
            "Downside Deviation":    _fmt_pct(rk.downside_deviation),
            "Longest Losing Streak": f"{rk.longest_losing_streak} bars",
        }),

        _make_df("Costs", {
            "Total Fee Drag":        _pct_and_dollar(c.total_fee_drag),
            "Total Funding Drag":    _pct_and_dollar(c.total_funding_drag),
            "Total Costs":           _pct_and_dollar(c.total_cost_drag),
            "Fee % of Net":          _fmt_pct(c.total_fee_pct_of_net),
            "Funding % of Net":      _fmt_pct(c.total_funding_pct_of_net),
            "Cost % of Net":         _fmt_pct(c.total_cost_pct_of_net),
            "Cost to Gross Ratio":   _fmt_float(c.cost_to_gross_ratio),
            "Fee Drag (Sharpe)":     _fmt_float(c.fee_drag_on_sharpe),
            "Funding Drag (Sharpe)": _fmt_float(c.funding_drag_on_sharpe),
            "Avg Fee per Bar":       _pct_and_dollar(c.avg_fee_per_bar),
            "Annualised Turnover":   f"{c.annualized_turnover:.1f}x",
            "% Bars Paying Funding": _fmt_pct(c.pct_bars_paying_funding),
        }),

        _make_df("Position", {
            "Avg Position Size":  _fmt_pct(p.avg_position_size),
            "Max Position Size":  _fmt_pct(p.max_position_size),
            "Avg Long Size":      _fmt_pct(p.avg_long_size),
            "Avg Short Size":     _fmt_pct(p.avg_short_size),
            "Time Long":          _fmt_pct(p.time_long),
            "Time Short":         _fmt_pct(p.time_short),
            "Time Flat":          _fmt_pct(p.time_flat),
            "Time in Market":     _fmt_pct(p.time_in_market),
            "Avg Holding Period": _fmt_bars(p.avg_holding_period),
            "Largest Win":        _pct_and_dollar(p.largest_win),
            "Largest Loss":       _pct_and_dollar(p.largest_loss),
            "Long PnL %":         _fmt_pct(p.long_pnl_pct),
            "Short PnL %":        _fmt_pct(p.short_pnl_pct),
            "Long Sharpe":        _fmt_float(p.long_sharpe),
            "Short Sharpe":       _fmt_float(p.short_sharpe),
        }),
    ]

    for df in sections:
        display(df)