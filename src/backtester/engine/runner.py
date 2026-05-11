import numpy as np
import pandas as pd

from backtester.data_loader import prepare_data
from backtester.engine.pnl import run_backtest
from backtester.metrics.performance_report import PerformanceReport
from backtester.engine.config import Config
from backtester.metrics.display import display_report
from strategy_engine.strategies import StrategyBase
from strategy_engine.features import FeatureRegistry

### This should be combinded into a run function, that does all these things
class BacktestRunner:
    """
    """


    def __init__(self, config: Config, strategy = StrategyBase):
        """
        Parameters
        ----------
        config : Config object
            Must contain: symbol, interval, start, end,
                         capital, leverage, fee_rate, delay_bars
        """
        self.config    = config
        self.strategy  = self.strategy
        self.registry  = FeatureRegistry()

        self.data    = None
        self.signals = None
        self.pnl_df  = None
        self.report  = None


    def run(self):
        self._load_data()
        self._compute_features()
        self._generate_signals()
        self._generate_report()


    def _load_data(self):
        """ Load requested market data as a pandas dataframe.
        Currently loads all columns.
        """
        self.data = prepare_data(self.config)


    ### FUNCTION THAT GETS SIGNALS
    def _generate_signals(self):
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
        

    def _generate_report(self):
        if self.report is None:
            self.report = PerformanceReport(self.pnl_df)
        display_report(self.report, symbol = self.config.symbol)


    def generate_plots(self):
        pass