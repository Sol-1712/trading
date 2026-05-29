from .core_stats import CoreStats
from .return_metrics import ReturnMetrics
from .risk_metrics import RiskMetrics
from .cost_metrics import CostMetrics
from .position_metrics import PositionMetrics

class PerformanceReport:
    """
    Master orchestrator for all performance metrics.

    Owns:
    - CoreStats
    - ReturnMetrics
    - RiskMetrics
    - CostMetrics
    - PositionMetrics
    """

    def __init__(self, pnl_df):
        """
        Parameters
        ----------
        pnl_df : pd.DataFrame
            Must include at least:
            - 'timestamp' (pd.DatetimeIndex or column)
            - 'pnl' (strategy PnL per bar)
            Optional:
            - 'position', 'fee', 'costs', etc.
        """
        # Ensure timestamp index
        if "timestamp" in pnl_df.columns:
            pnl_df = pnl_df.set_index("timestamp")
        self.pnl_df = pnl_df.sort_index()

        # Compute core statistics (equity curve, drawdown, etc.)
        self.core = CoreStats(self.pnl_df)

        # Metrics classes
        self.returns = ReturnMetrics(self.core)
        self.risk = RiskMetrics(self.core)
        self.cost = CostMetrics(self.core)
        self.position = PositionMetrics(self.core)


    def to_dict(self) -> dict:
        """
        Returns all metrics as a flat dictionary of raw values.
        Suitable for building comparison DataFrames across parameter sweeps.

        Returns
        -------
        dict
            Flat dictionary of metric_name -> raw float value.
        """
        r  = self.returns
        rk = self.risk
        c  = self.cost
        p  = self.position

        avg_win_loss, expectancy, hit_rate_trade = r.avg_win_loss_ratio_expectancy

        return {
            # ── Returns ──────────────────────────────────────────────────
            "cagr":               r.cagr,
            "net_return":         r.net_return,
            "gross_return":       r.gross_return,
            "sharpe":             r.sharpe,
            "annualised_sharpe":  r.annualised_sharpe,
            "sortino":            r.sortino,
            "hit_rate_trade":     hit_rate_trade,
            "avg_win_loss":       avg_win_loss,
            "expectancy":         expectancy,
            "profit_factor":      r.profit_factor,
            "skew":               r.skew,
            "kurtosis":           r.kurtosis,

            # ── Risk ─────────────────────────────────────────────────────
            "volatility":            rk.volatility,
            "max_drawdown":          rk.max_drawdown,
            "max_drawdown_duration": rk.max_drawdown_duration,
            "avg_drawdown_duration": rk.avg_drawdown_duration,
            "time_in_drawdown":      rk.time_in_drawdown,
            "calmar":                rk.calmar,
            "var_95":                rk.var_95,
            "var_99":                rk.var_99,
            "cvar_95":               rk.cvar_95,
            "cvar_99":               rk.cvar_99,
            "downside_deviation":    rk.downside_deviation,
            "longest_losing_streak": rk.longest_losing_streak,

            # ── Costs ────────────────────────────────────────────────────
            "total_fee_return":       c.total_fee_drag,
            "total_funding_return":   c.total_funding_drag,
            "total_cost_return":      c.total_cost_drag,
            "total_fee_pct_of_net":   c.total_fee_pct_of_net,
            "funding_pct_of_net":     c.total_funding_pct_of_net,
            "total_cost_pct_of_net":  c.total_cost_pct_of_net,
            "cost_to_gross_ratio":    c.cost_to_gross_ratio,
            "fee_drag_sharpe":        c.fee_drag_on_sharpe,
            "funding_drag_sharpe":    c.funding_drag_on_sharpe,
            "avg_fee_per_bar":        c.avg_fee_per_bar,
            "annualised_turnover":    c.annualized_turnover,
            "pct_bars_paying_funding": c.pct_bars_paying_funding,

            # ── Position ─────────────────────────────────────────────────
            "avg_position_size":    p.avg_position_size,
            "max_position_size":    p.max_position_size,
            "avg_long_size":        p.avg_long_size,
            "avg_short_size":       p.avg_short_size,
            "time_long":            p.time_long,
            "time_short":           p.time_short,
            "time_flat":            p.time_flat,
            "time_in_market":       p.time_in_market,
            "avg_holding_period":   p.avg_holding_period,
            "largest_win":          p.largest_win,
            "largest_loss":         p.largest_loss,
            "long_pnl_pct":         p.long_pnl_pct,
            "short_pnl_pct":        p.short_pnl_pct,
            "long_sharpe":          p.long_sharpe,
            "short_sharpe":         p.short_sharpe,
        }