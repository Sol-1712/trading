import numpy as np
import pandas as pd
import logging

from trading.data_utils.prepare                      import prepare_data
from trading.backtester.engine.config_bases                import BacktestConfig
from trading.backtester.execution                    import PerpDirectionalEngine
from trading.strategy_engine.core                    import StrategyBase, DataRequirements
from trading.strategy_engine.strategies.directional  import DirectionalStrategy
from trading.strategy_engine.features                import FeatureRegistry
from trading.strategy_engine.core                    import Signal
from trading.backtester.metrics.results              import BacktestResults
from trading.backtester.risk.temp_sizer              import simple_size
from trading.backtester.portfolio                    import Portfolio


logger = logging.getLogger(__name__)

class BacktestRunner:
    """
    Orchestrates a complete backtest: loads data, computes features, generates signals,
    and simulates execution via portfolio and execution engine.
    
    The runner enforces a clear separation between stateless signal generation (data +
    features → signals) and stateful execution (bar-by-bar portfolio updates).
    
    Parameters
    ----------
    config : BacktestConfig
        Backtest configuration including data source, execution parameters, and capital.
    strategy : StrategyBase
        Strategy implementation to generate signals. Must implement DirectionalStrategy
        for current execution path.
        
    Raises
    ------
    ValueError
        If initial_capital is non-positive (validated by BacktestConfig).
    """

    def __init__(self, config: BacktestConfig, strategy: StrategyBase):
        if config is None:
            raise ValueError("config cannot be None")
        if strategy is None:
            raise ValueError("strategy cannot be None")
            
        self.config           = config
        self.strategy         = strategy

        self._data:           pd.DataFrame

        self._registry        = FeatureRegistry()
        self._portfolio       = Portfolio(
                config.initial_capital, 
                self.config.execution.fee_rate
                )
        self._engine          = PerpDirectionalEngine(self.config.execution)
        logger.debug("BacktestRunner initialized with strategy=%s", 
                    type(self.strategy).__name__)


    def _load_data(self) -> pd.DataFrame:
        """
        Load market data for backtest period.
        
        Requests data based on strategy requirements (price types and columns).
        Raises on missing or invalid data.
        
        Returns
        -------
        pd.DataFrame
            Market data indexed by timestamp with requested columns.
            
        Raises
        ------
        ValueError
            If returned data is empty or has zero rows.
        """

        signal_type = self.strategy.data_requirements().price_type
        execution_type = self.config.execution.price_type
        price_types = tuple({signal_type, execution_type})
        
        data = prepare_data(
            config      = self.config.data,
            price_types = price_types,
            columns     = self.strategy.data_requirements().columns,
        )
        
        if data is None or data.empty:
            raise ValueError("Data loader returned empty DataFrame")
        
        logger.info("Loaded %d bars from %s to %s", 
                   len(data), data.index[0], data.index[-1])
        return data


    def _compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute strategy-required features from raw market data.
        
        Registers strategy features and computes batch for all bars.
        Currently only supports DirectionalStrategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw market data indexed by timestamp.
            
        Returns
        -------
        pd.DataFrame
            Input data plus computed feature columns.
            
        Raises
        ------
        NotImplementedError
            If strategy type is not DirectionalStrategy.
        ValueError
            If data is None or empty.
        """
        if data is None or data.empty:
            raise ValueError("Cannot compute features on empty data")
            
        if not isinstance(self.strategy, DirectionalStrategy):
            raise NotImplementedError(
                f"Feature computation not supported for {type(self.strategy).__name__}"
            )
        self.strategy.register_features(self._registry)
        required_features = self.strategy.required_features()
        result = self._registry.compute_batch(data, required_features)
        
        logger.debug("Computed %d features across %d bars", 
                    len(required_features), len(result))
        return result


    def _generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate trading signals from market data and computed features.
        
        Delegates to strategy's signal generator. Signals must be timestamped
        at or before corresponding bar timestamp (no lookahead).
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with all required features already computed.
            
        Returns
        -------
        list[Signal]
            List of Signal objects (or None entries for bars with no signal).
            Length equals data row count.
            
        Raises
        ------
        NotImplementedError
            If strategy type is not DirectionalStrategy.
        ValueError
            If data is None or empty.
        """
        if data is None or data.empty:
            raise ValueError("Cannot generate signals from empty data")
            
        if not isinstance(self.strategy, DirectionalStrategy):
            raise NotImplementedError(
                f"Signal generation not supported for {type(self.strategy).__name__}"
            )

        signals = self.strategy.generate_signals(data)
        logger.debug("Generated %d signal events from %d bars", 
                    sum(1 for s in signals if s is not None), len(signals))
        return signals


    def _validate_signals(self, signals: list[Signal], data: pd.DataFrame) -> None:
        """
        Validate signals for correctness and lookahead bias.
        
        Checks:
        - Signal count matches data length
        - Signal timestamps are at or before corresponding bar
        - Signal timestamps fall within dataset range
        
        Parameters
        ----------
        signals : list[Signal]
            Signal list to validate (may contain None entries).
        data : pd.DataFrame
            Market data signals were generated from.
            
        Raises
        ------
        ValueError
            If signal count mismatches data length.
        ValueError
            If lookahead is detected (signal timestamp after bar).
        ValueError
            If signal timestamp predates dataset start.
        """
        if signals is None or data is None or data.empty:
            raise ValueError("Signals and data must be non-None and non-empty")
            
        if len(signals) != len(data):
            raise ValueError(f"Signal count {len(signals)} != data rows {len(data)}")
        
        for t, signal in enumerate(signals):
            if signal is None:
                continue

            bar_timestamp = data.index[t]

            if signal.timestamp > bar_timestamp:
                raise ValueError(
                    f"Lookahead detected at bar {t}: "
                    f"signal timestamp {signal.timestamp} is after "
                    f"bar timestamp {bar_timestamp}. "
                    f"Signal was generated using future data."
                )

            if signal.timestamp < data.index[0]:
                raise ValueError(
                    f"Signal at bar {t} has timestamp {signal.timestamp} "
                    f"which predates the dataset start {data.index[0]}. "
                    f"Check signal generation logic."
                )
            
        logger.info("Validated %d signals — no lookahead detected.", len(signals))


    def _size_pos(self, signal: Signal) -> float | None:
        """
        Convert signal to a signed target position fraction.
        
        Delegates to position sizer. Returns None if no sizing action needed.
        
        Parameters
        ----------
        signal : Signal | None
            Signal to size. None signals return None (no action).
            
        Returns
        -------
        float | None
            Signed target position as fraction of equity (e.g., 0.5 = 50% long).
            None means no action (keep current position).
            
        Raises
        ------
        ValueError
            If signal produces invalid position (NaN or ±inf).
        """
        result = simple_size(signal)
        
        if result is not None and (np.isnan(result) or np.isinf(result)):
            raise ValueError(f"Position sizer returned invalid position: {result}")
        
        return result


    def _apply_risk(self, target: float) -> float | None:
        """
        Apply risk adjustments to target position.
        
        Currently a passthrough (no risk limits). Designed as integration point
        for future RiskEngine that may reduce, block, or adjust position targets.
        
        Parameters
        ----------
        target : float
            Raw target position from position sizer as fraction of equity.
            
        Returns
        -------
        float | None
            Risk-adjusted target position. May differ from input if risk engine
            is active. Returns None if position should be skipped.
        """
        return target
    
    

    def run_NEW(self) -> BacktestResults:
        """
        Execute full backtest: data load → feature computation → signal generation
        → bar-by-bar execution simulation.
        
        PHASE 1 (Stateless): Load data, compute features, generate signals.
        PHASE 2 (Stateful): Loop through bars, execute orders, track portfolio state.
        
        Returns
        -------
        BacktestResults
            Complete backtest output including price data, signals, targets, and
            portfolio history (equity, positions, PnL components by bar).
            
        Raises
        ------
        ValueError
            If validation fails (empty data, signal/data mismatch, lookahead).
        RuntimeError
            If portfolio or engine enters invalid state during execution.
        """
        # PHASE 1: Stateless
        logger.info("Starting backtest run")
        self._data = self._load_data()
        self._data = self._compute_features(self._data)

        signals = self._generate_signals(self._data)
        self._validate_signals(signals, self._data)

        targets = []
        logger.info("Phase 1 complete: data and signals ready, entering execution loop")
    
        # PHASE 2: Stateful
        for t in range(len(self._data)):
            
            bar = self._data.iloc[t]

            fills = self._engine.execute_pending(bar, t)

            state = self._portfolio.step(fills, bar)

            signal = signals[t] 
            target = self._size_pos(signal)
            targets.append(target)

            if target is None:
                continue

            self._engine.submit(target, state, bar)

        history = self._portfolio.history()
        logger.info("Backtest complete: %d bars processed, final equity=%.2f",
                   len(self._data), self._portfolio.equity)

        return BacktestResults(
            data    = self._data,
            signals = signals,
            targets = targets,
            portfolio_history = history,
        )
