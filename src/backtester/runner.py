import numpy as np
import pandas as pd

from backtester.data_loader import load_backtest_data
from backtester.pnl import pnl
from backtester.metrics.performance_report import PerformanceReport


class BacktestRunner:
    """
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Must contain: symbol, interval, start, end,
                         capital, leverage, fee_rate, delay_bars
        """
### Leverage is just exchange cap -> 100
        self.config = config

        self.data   = None
        self.pnl_df = None
        self.report = None


    def load_data(self, cols: list[str] = ["mark_close"]):
        """ Load and format market data"""
        self.data = load_backtest_data()


    def run(self, signals: np.ndarray):
        if self.data is None:
            raise RuntimeError("Call load_data() before run()")

        self.pnl_df = pnl(
            data_df    = self.data,
            signals    = signals,
            capital    = self.config["capital"],
            leverage   = self.config["leverage"],
            fee_rate   = self.config["fee_rate"],
            delay_bars = self.config["delay_bars"],
        )
        self.report = PerformanceReport(self.pnl_df)
