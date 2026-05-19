import numpy as np
import pandas as pd

from data_utils.prepare                      import prepare_data
from backtester.metrics.performance_report   import PerformanceReport
from backtester.engine.config                import BacktestConfig
from backtester.engine.execution             import ExecutionConfig, PerpDirectionalEngine
from strategy_engine.strategies              import StrategyBase
from strategy_engine.strategies.directional  import DirectionalStrategy
from strategy_engine.features                import FeatureRegistry
from strategy_engine.core                    import Signal
from backtester.metrics.results              import BacktestResults
from backtester.risk.temp_sizer              import simple_size
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
        targets          = self._size_pos(signals)
        targets_adjusted = self._apply_risk(targets, data_rich)

        portfolio = self._run_portfolio(
            data    = data_rich,
            targets = targets_adjusted
        )
        history = portfolio.history()

        return BacktestResults(
            data    = data_rich,
            signals = signals,
            targets = targets_adjusted,
            portfolio_history = history,
            #report = PerformanceReport(history)
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
        """
        Dirty stub for now.
        """
        return simple_size(signals)


    def _apply_risk(
        self, 
        targets: pd.Series, 
        data:    pd.DataFrame,
    ) -> pd.Series:
        """
        Hook for risk engine. Adjusts raw position targets before execution.
        Currently a passthrough — RiskEngine integration added here.
        """
        return targets
    

    def _run_portfolio(
        self,
        data:    pd.DataFrame,
        targets: pd.Series,
    ) -> Portfolio:
    #         → resolves price column
    # → creates PerpDirectionalEngine
    # → calls engine.run()
    #     → validates inputs
    #     → applies delay to targets
    #     → creates Portfolio
    #     → loops bars, calls portfolio.step()
    #         → MTM → funding → sizing → trade → fee
    #         → records PortfolioSnapshot
    #     → returns Portfolio
    # → caller calls portfolio.history()
    #     → returns pd.DataFrame of snapshots

        """
        Selects the correct engine for the strategy type,
        resolves the execution price column, and runs simulation.
        """
        engine    = PerpDirectionalEngine()
        price_col = self._resolve_execution_price_col(data)

        return engine.run(
            targets   = targets,
            data      = data,
            config    = self.config.execution,
            capital   = self.config.initial_capital,
            price_col = price_col,
        )
    

    def _resolve_execution_price_col(self, data: pd.DataFrame) -> str:
        """
        Resolves execution price column from ExecutionConfig.execution_price_type.
        Tries prefixed (mark_close) then unprefixed (close).
        Raises clearly if neither is found.
        """
        pass