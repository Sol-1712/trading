import logging
from dataclasses import dataclass
import pandas as pd
from trading.strategy_engine.core import Signal

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Complete backtest output container.
    
    Contains generated signals, position targets, and resulting
    portfolio performance history. Used to initialize metrics computation and
    performance reporting.
    
    Attributes
    ----------

    signals : list[Signal]
        Signal objects or None entries for each bar (length = len(data)).
    targets : list[float]
        Target position fractions output by position sizer (length = len(data)).
    portfolio_history : pd.DataFrame
        Portfolio state snapshots indexed by timestamp with columns: equity,
        position_units, position_pnl, funding_pnl, fees, leverage, etc.
    """
    signals:              list[Signal]
    targets:              list[float | None]
    portfolio_history:    pd.DataFrame
    
    def __post_init__(self):

        if self.portfolio_history is None or self.portfolio_history.empty:
            raise ValueError("portfolio_history cannot be None or empty")
        if len(self.signals) != len(self.portfolio_history):
            raise ValueError(f"Signal count {len(self.signals)} != history length {len(self.portfolio_history)}")
        if len(self.targets) != len(self.portfolio_history):
            raise ValueError(f"Target count {len(self.targets)} != history length {len(self.portfolio_history)}")
        
        logger.debug("BacktestResults created: %d signals, final equity=%.2f",
                    sum(1 for s in self.signals if s is not None),
                    self.portfolio_history['equity'].iloc[-1])