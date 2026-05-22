import numpy as np
import pandas as pd

from data_utils.prepare                      import prepare_data
from backtester.metrics.performance_report   import PerformanceReport
from backtester.engine.config                import BacktestConfig
from backtester.engine.execution             import PerpDirectionalEngine
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
        self.strategy         = strategy


        # Initialise objects
        self._data:           pd.DataFrame

        self._registry        = FeatureRegistry()
        self._portfolio       = Portfolio(
            config.initial_capital, 
            self.config.execution.fee_rate
            )
        # Will need to be any engine
        self._engine          = PerpDirectionalEngine(self.config.execution)


    def _load_data(self) -> pd.DataFrame:
        """ Load requested market data as a pandas dataframe.
        """
        requirements = self.strategy.data_requirements()
        return prepare_data(
            config      = self.config.data,         # symbol, interval, dates
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
    
    

    def run_NEW(self) -> BacktestResults:
        ###       STATELESS       ###
        # ------------------------- #
        
        self._data = self._load_data()
        self._data = self._compute_features(self._data)
        ### SIGNALS NEED TO VALIDATE DATA CONTAINS COLUMNS (FEATURES) IT NEEDS
        signals    = self._generate_signals(self._data)

        

        ### TEMPORARILY STATELESS ###
        # ------------------------- #

        targets          = self._size_pos(signals)
        targets_adjusted = self._apply_risk(targets, self._data)

        ###       STATEFULL      ###
        # ------------------------- #

        # Loop
        for t in range(len(self._data)):
            
            ### KEY: Orders submitted at t are NOT stepped until t+1
            bar = self._data.iloc[t]
            timestamp = bar.name


            fills = self._engine.execute_pending(bar) # Needs to be none if no pending


            # Update portfolio with fills, funding, mtm.
            # Returns a portfolio snapshot
            state = self._portfolio.step(fills, bar)


            # Skip for now.
            # signal = signals[t] ?
            # target = self._position_constructor.build(signal, state)
            # target = self._risk.apply(target, state)



            # Submit new order(s) based on targets.
            # Make sure the timesteps line up properly (KEY)
            # -> A target coming at close of bar t is submitted at 
            self._engine.submit(targets[t], state, bar ,t)
            



        # Loop is finished, return results
        history = self._portfolio.history()

        return BacktestResults(
            data    = self._data,
            signals = signals,
            targets = targets_adjusted,
            portfolio_history = history,
            #report = PerformanceReport(history)
        )



    def _resolve_price_column(self):
        pass