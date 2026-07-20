import numpy as np
import pandas as pd
from pathlib import Path
import json
from IPython.display import display

from trading.backtester.portfolio import TradeLog
from .core_stats import CoreStats
from .returns    import ReturnMetrics
from .risk       import RiskMetrics
from .cost       import CostMetrics
from .trade      import TradeMetrics


class PerformanceReport:
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
        """Flat dict of every exported scalar metric across all groups."""
        return {
            **self.returns.to_dict(),
            **self.risk.to_dict(),
            **self.cost.to_dict(),
            **self.trade.to_dict(),
        }

    def save(self, run_dir: Path) -> None:
        with open(run_dir / "report.json", "w") as f:
            json.dump(self.summary(), f, indent=2)

    def display_report(self, symbol: str | None) -> None:
        """
        Displays a formatted PerformanceReport in a Jupyter notebook.

        Parameters
        ----------
        symbol : str | None
            Optional asset symbol
        """
        pass