import numpy as np
import pandas as pd

from backtester.data_loader import prepare_data
from backtester.engine.pnl import run_backtest
from backtester.metrics.performance_report import PerformanceReport
from backtester.config import Config
from backtester.metrics.display import display_report

### This should be combinded into a run function, that does all these things
class BacktestRunner:
    """
    """


    def __init__(self, config: Config):
        """
        Parameters
        ----------
        config : Config object
            Must contain: symbol, interval, start, end,
                         capital, leverage, fee_rate, delay_bars
        """
        self.config  = config

        self.data    = None
        self.signals = None
        self.pnl_df  = None
        self.report  = None


    def run(self):
        print('test')


    def load_data(self):
        """ Load requested market data as a pandas dataframe.
        Currently loads all columns.
        """
        self.data = prepare_data(self.config)


    ### FUNCTION THAT GETS SIGNALS
    def generate_signals(self):
        pass


    def run_backtest(self):
        if self.data is None:
            raise RuntimeError("Call load_data() before running")
        
        if self.signals is None:
            raise RuntimeError("Call generate_signals() before running")
        
        self.pnl_df = run_backtest(
            data_df    = self.data,
            positions  = self.signals,
            capital    = self.config.capital,
            fee_rate   = self.config.fee_rate,
            delay_bars = self.config.delay_bars,
            price_col  = self.config.price_column
        )
        

    def generate_report(self):
        self.report = PerformanceReport(self.pnl_df)
        display_report(self.report, symbol = self.config.symbol)


    def generate_plots(self):
        pass