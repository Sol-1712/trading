import numpy as np
import pandas as pd

from backtester.data_loader import prepare_data
from backtester.metrics.performance_report import PerformanceReport
from backtester.engine.config import Config
from strategy_engine.strategies import StrategyBase
from strategy_engine.strategies.directional import DirectionalStrategy
from strategy_engine.features import FeatureRegistry
from strategy_engine.core import Signal
from backtester.engine.results import BacktestResults
from risk.temp_sizer import simple_size
from backtester.portfolio import Portfolio




class BacktestRunner:

    def __init__(self, config: Config, strategy: StrategyBase):
        """
        Parameters
        ----------
        config : Config object
        Required attributes:
            "symbol", "interval", "start", "end",
            "initial_capital", "fee_rate", "delay_bars", 
            "signal_price_type", "execution_price_type", "leverage_max"
        """
        self.config    = config
        self.strategy  = strategy
        self._registry = FeatureRegistry()


    def run(self) -> BacktestResults:
        if isinstance(self.strategy, DirectionalStrategy):
            return self._run_directional()

        else:
            raise NotImplementedError


    def _run_directional(self) -> BacktestResults:
        data      = self._load_data()
        data_rich = self._compute_features(data)
        signals   = self._generate_signals(data_rich)
        # --------------------------------- #
        # State dependent from here #
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
        Currently loads all columns.
        """
        return prepare_data(self.config)


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
        """
        Iterate bar-by-bar, advancing the portfolio at each step.
 
        Delay handling
        --------------
        A signal generated at bar T should fill at bar T + delay_bars.
        We implement this by shifting the targets series forward by
        delay_bars positions before the loop. The fill therefore uses
        the price at T + delay_bars, which is the price already present
        in `data` at that row — no lookahead.
 
        The first delay_bars bars will have NaN targets (no signal has
        arrived yet); these are filled with 0.0 (flat).
 
        Parameters
        ----------
        data : pd.DataFrame
            Full enriched DataFrame indexed by timestamp.
        targets : pd.Series
            Signed position fractions from the sizer, indexed by timestamp.
 
        Returns
        -------
        Portfolio
            Portfolio object with full snapshot history populated.
        """

        prices = data[f'{self.config.execution_price_type.value}_open']  # assuming execution at open of next bar after signal

        # Align targets to the price index, then apply execution delay.
        # shift(n) moves each value n positions later in the index, so
        # at bar t we execute the fraction that was set at bar t - delay_bars.
        delayed_targets: pd.Series = (
            targets
            .reindex(prices.index)   # fill bars with no signal as NaN
            .shift(self.config.delay_bars)
            .fillna(0.0)             # pre-signal bars: stay flat
        )

        # Funding rates: use data column if present, else all zeros.
        if "fundingRate" in data.columns:
            funding_rates: pd.Series = (
                data['fundingRate']
                .reindex(prices.index)
                .fillna(0.0)
            )
        else:
            funding_rates = pd.Series(0.0, index=prices.index)

        # Pull to numpy for the loop — avoids per-iteration index lookups
        # and resolves the Pylance overload ambiguity on Series.__getitem__.
        prices_arr:  np.ndarray = prices.to_numpy(dtype=float)
        targets_arr: np.ndarray = delayed_targets.to_numpy(dtype=float)
        funding_arr: np.ndarray = funding_rates.to_numpy(dtype=float)
        timestamps              = prices.index

        portfolio = Portfolio(
            initial_capital = self.config.initial_capital,
            fee_rate        = self.config.fee_rate,
        )
 
        for i, ts in enumerate(timestamps):
            portfolio.step(
                timestamp       = ts,
                price           = prices_arr[i],
                target_fraction = targets_arr[i],
                funding_rate    = funding_arr[i],
            )
 
        return portfolio
    

