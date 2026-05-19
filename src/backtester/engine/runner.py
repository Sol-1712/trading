import numpy as np
import pandas as pd

from data_utils.prepare                      import prepare_data
from backtester.metrics.performance_report   import PerformanceReport
from backtester.engine.config                import BacktestConfig
from backtester.engine.execution             import ExecutionConfig
from strategy_engine.strategies              import StrategyBase
from strategy_engine.strategies.directional  import DirectionalStrategy
from strategy_engine.features                import FeatureRegistry
from strategy_engine.core                    import Signal
from backtester.engine.results               import BacktestResults
from risk.temp_sizer                         import simple_size
from backtester.portfolio                    import Portfolio




class BacktestRunner:

    def __init__(self, config: BacktestConfig, strategy: StrategyBase):
        """
        Parameters
        ----------
        config : BacktestConfig

        strategy : StrategyBase


        """
        self.config           = config
        self.execution_config = config.execution
        self.data_config      = config.data
        self.strategy         = strategy
        self._registry        = FeatureRegistry()


    def run(self) -> BacktestResults:
        if isinstance(self.strategy, DirectionalStrategy):
            return self._run_directional()

        else:
            raise NotImplementedError


    def _run_directional(self) -> BacktestResults:
        data      = self._load_data()
        data_rich = self._compute_features(data)
        signals   = self._generate_signals(data_rich)

        # ------------------------- #
        # State dependent from here #
        # ------------------------- #
        pos       = self._size_pos(signals)

        portfolio = self._run_backtest(
            data    = data_rich,
            targets = pos
        )
        history = portfolio.history()

        return BacktestResults(
            data    = data_rich,
            signals = signals,
            targets = pos,
            portfolio_history = history,
            report = PerformanceReport(history)
        )


    def _load_data(self) -> pd.DataFrame:
        """ Load requested market data as a pandas dataframe.
        """
        requirements = self.strategy.data_requirements()
        return prepare_data(
            config      = self.data_config,         # symbol, interval, dates
            price_types = requirements.price_types, # which klines file to load
            columns     = requirements.columns,     # optional column filter
        )


    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(self.strategy, DirectionalStrategy):
            raise NotImplementedError(
                f"Feature computation not supported for {type(self.strategy).__name__}"
            )
        self.strategy.register_features(self._registry)
        required_features = self.strategy.required_features()
        return self._registry.compute_batch(data, required_features)


    # Directional method
    def _generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        
        if not isinstance(self.strategy, DirectionalStrategy):
            raise NotImplementedError(
                f"Signal generation not supported for {type(self.strategy).__name__}"
            )

        return self.strategy.generate_signals(data)     


    def _size_pos(self, signals: list[Signal]) -> pd.Series:
        return simple_size(signals)


    def _run_backtest(
        self,
        data:    pd.DataFrame,
        targets: pd.Series,
    ) -> Portfolio:
        

        t = Portfolio(initial_capital=self.config.initial_capital, fee_rate=self.execution_config.fee_rate)
        return t