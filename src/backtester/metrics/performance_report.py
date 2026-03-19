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


    