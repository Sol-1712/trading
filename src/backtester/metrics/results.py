from dataclasses import dataclass
import pandas as pd
from strategy_engine.core import Signal
from backtester.metrics import PerformanceReport



@dataclass
class BacktestResults:
    data:                 pd.DataFrame
    signals:              list[Signal]
    targets:              pd.Series
    portfolio_history:    pd.DataFrame
    #report:               PerformanceReport