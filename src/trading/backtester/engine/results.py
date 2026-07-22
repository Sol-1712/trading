import logging
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path

from trading.strategy_engine.core import Signal
from trading.backtester.portfolio.trade import TradeLog

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
    targets : list[float | None]
        Target position fractions from the position sizer / risk step
        (length = len(data)); None means no action that bar.
    trade_log : TradeLog
        Trade log containing open and closed trades.
    portfolio_history : pd.DataFrame
        Portfolio state snapshots indexed by timestamp with columns such as
        equity, position_units, position_pnl, funding_pnl, fees.
    """
    signals:              list[Signal]
    targets:              list[float | None]
    trade_log:            TradeLog
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

    
    def save(self, run_dir: Path) -> None:
        """
        Persist portfolio history and closed trades to ``run_dir``.

        Writes ``portfolio_history.parquet`` and ``trades.parquet``.
        Signals are not saved.

        Parameters
        ----------
        run_dir : Path
            Destination directory for result artifacts.
        """
        self.portfolio_history.to_parquet(run_dir / "portfolio_history.parquet")

        trades_df = pd.DataFrame([
            {**asdict(t), "direction": t.direction.name} for t in self.trade_log.closed_trades
        ])
        trades_df.to_parquet(run_dir / "trades.parquet")
    
        # Maybe save signals later if I want.
        logger.info("Successfully saved backtest results to %s", run_dir)