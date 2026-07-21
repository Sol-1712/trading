import pandas as pd
from pathlib import Path
import json

from trading.backtester.portfolio import TradeLog
from trading.data_utils.core import PROJECT_ROOT
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


    def display_report(
        self,
        metrics: dict[str, float],
        config: dict,           # parsed from config.yaml (or already in memory post-run)
        starting_capital: float,
        final_capital: float,
    ) -> None:
        pass
    
    
def load_report(run_dir: str | Path) -> dict[str, float]:
    path = PROJECT_ROOT / "runs" / run_dir if isinstance(run_dir, str) else run_dir
    with open(path / "report.json", "r") as f:
        return json.load(f)