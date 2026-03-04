from .core_stats import CoreStats
from .return_metrics import ReturnMetrics
from .risk_metrics import RiskMetrics
# from .cost_metrics import CostMetrics
# from .position_metrics import PositionMetrics


# Costs

# Total Fees ($)
# Total Fees (% of Gross PnL)
# Total Funding ($) — can be positive or negative, important alpha/cost for perps
# Net PnL ($)
# Gross PnL ($)
# Fee Drag on Sharpe
# Funding Drag on Sharpe — unique to perps, tells you if funding is helping or hurting
# Annualised Turnover
# Avg Fee Per Trade ($)


# Position

# Avg Position Size (% equity)
# Time Long / Time Short / Time Flat
# Avg Long Size / Avg Short Size
# Avg Holding Period (bars)
# Avg Trade Size ($)
# Max Position Size (% equity)
# Avg Leverage Used
# Largest Single Win / Largest Single Loss
# Long PnL vs Short PnL split — tells you which side of the book is generating alpha
# Long Sharpe vs Short Sharpe — whether your edge is asymmetric between long and short

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
        self.risk = RiskMetrics(self.core, self.returns)
        # self.cost = CostMetrics(self.core)
        # self.position = PositionMetrics(self.core)

    @property
    def equity(self):
        return self.core.equity
    
    @property
    def mdd(self):
        return self.core.mdd
    
    @property
    def sd_returns(self):
        """
        Annualised
        """
        return self.core.sd * self.core.ann_factor