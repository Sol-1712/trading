import logging
from dataclasses import dataclass
import pandas as pd
from strategy_engine.core import Signal
from backtester.metrics import PerformanceReport

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Complete backtest output container.
    
    Contains all input data, generated signals, position targets, and resulting
    portfolio performance history. Used to initialize metrics computation and
    performance reporting.
    
    Attributes
    ----------
    data : pd.DataFrame
        Market data with computed features indexed by timestamp.
    signals : list[Signal]
        Signal objects or None entries for each bar (length = len(data)).
    targets : list[float]
        Target position fractions output by position sizer (length = len(data)).
    portfolio_history : pd.DataFrame
        Portfolio state snapshots indexed by timestamp with columns: equity,
        position_units, bar_pnl, funding_pnl, fees, leverage, etc.
    """
    data:                 pd.DataFrame
    signals:              list[Signal]
    targets:              list[float]
    portfolio_history:    pd.DataFrame
    
    def __post_init__(self):
        if self.data is None or self.data.empty:
            raise ValueError("data cannot be None or empty")
        if self.portfolio_history is None or self.portfolio_history.empty:
            raise ValueError("portfolio_history cannot be None or empty")
        if len(self.signals) != len(self.data):
            raise ValueError(f"Signal count {len(self.signals)} != data length {len(self.data)}")
        if len(self.targets) != len(self.data):
            raise ValueError(f"Target count {len(self.targets)} != data length {len(self.data)}")
        
        logger.debug("BacktestResults created: %d bars, %d signals, final equity=%.2f",
                    len(self.data),
                    sum(1 for s in self.signals if s is not None),
                    self.portfolio_history['equity'].iloc[-1])