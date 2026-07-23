from trading.data_utils.core.enums import PriceType
import pandas as pd
import logging
import numpy as np
import math

from trading.data_utils.prepare                      import prepare_data
from trading.backtester.engine.config_bases          import BacktestConfig
from trading.backtester.execution                    import PerpDirectionalEngine
from trading.strategy_engine.strategies.directional  import DirectionalStrategy
from trading.strategy_engine.features                import FeatureRegistry
from trading.strategy_engine.core                    import Signal, SignalDirection, StrategyBase
from trading.backtester.risk.temp_sizer              import simple_size
from trading.backtester.portfolio                    import Portfolio, PortfolioSnapshot
from .results                                        import BacktestResults

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
            
        self.config           = config
        self.strategy         = strategy

        self._registry        = FeatureRegistry()
        self._portfolio       = Portfolio(initial_capital=config.initial_capital)
        self._engine          = PerpDirectionalEngine(config.execution)
        self._mtm_col         = f"{config.execution.mtm_price_type.value}_close" 

        logger.debug("BacktestRunner initialized with strategy=%s", 
                    type(self.strategy).__name__)


    def run(self) -> BacktestResults:
        """
        Execute full backtest: data load → feature computation → signal generation
        → bar-by-bar execution simulation.
        
        PHASE 1 (Stateless): Load data, compute features, generate signals.
        PHASE 2 (Stateful): Loop through bars, execute orders, track portfolio state.
        
        Returns
        -------
        BacktestResults
            Complete backtest output including signals, targets, trade log and
            portfolio history (equity, positions, PnL components by bar).
            
        Raises
        ------
        ValueError
            If validation fails (empty data, signal/data mismatch, lookahead).
        RuntimeError
            If portfolio or engine enters invalid state during execution.
        """

        # ── Phase 1: Stateless ────────────────────────────────────────
        logger.info("Starting backtest run: %s", self.strategy.config.name)

        data = self._load_data()
        data = self._compute_features(data)

        signals = self._generate_signals(data)
        self._validate_signals(signals, data)

        logger.info(
            "Phase 1 complete — %d bars, %d signals.",
            len(data),
            sum(1 for s in signals if s is not None),
        )
    
        
        # ── Phase 2: Stateful ─────────────────────────────────────────
        targets:  list[float | None] = []
        index     = data.index
        n         = len(data)
        t         = -1

        for t in range(n):

            bar          = data.iloc[t]
            timestamp    = index[t]
            mark_close   = float(bar[self._mtm_col])
            funding_rate = float(bar.get("funding_rate", 0.0))

            # Accrue portfolio
            self._portfolio.accrue_bar(
                timestamp    = timestamp,
                mtm_price    = mark_close,
                funding_rate = funding_rate)

            if self._portfolio.is_ruined():
                logger.info("Portfolio reached ruin at bar %d — stopping backtest", t)
                break

            # Attempt fills on pending orders
            fills = self._engine.execute_pending(bar, t)

            # Apply fills
            self._portfolio.apply_fills(
                fills        = fills,
            )

            # --- Non-final Bar ----------------------------------------
            if t < n-1: 

                state = self._portfolio.commit_snapshot()

                # Signal -> Target -> Risk
                signal = signals[t] 
                target = self._size_pos(signal)
                target = self._apply_risk(target, state)
                targets.append(target)

                if target is not None:
                    delta = self._compute_delta_notional(
                        target_fraction=target,
                        equity=state.equity,
                        position_units=state.position_units,
                        price=mark_close,
                    )
                    if delta is not None:
                        self._engine.submit(timestamp, delta_notional=delta)
        # --- After the loop -----------------------------------------

        exit_bar = bar
        exit_t = t
        exit_ts = timestamp
        exit_price = mark_close

        # Cancel leftover queue, flatten residual, commit exit bar once.
        self._engine.cancel_all_pending()
        if not self._portfolio.is_flat():
            self._flatten(
                timestamp=exit_ts,
                bar=exit_bar,
                t=exit_t,
                price=exit_price,
            )

        self._portfolio.commit_snapshot()

        history = self._portfolio.history()
        # Exit bar (last bar or ruin) has a snapshot but no sizing step.
        targets.append(None)

        logger.info(
            "Backtest complete: %d bars processed, final equity=%.2f",
            self._portfolio.n_bars,
            self._portfolio.equity,
        )

        # Align to run horizon (truncate signals if ruined early).
        n_hist = len(history)
        return BacktestResults(
            signals=signals[:n_hist],
            targets=targets,
            trade_log=self._portfolio.trade_log,
            portfolio_history=history,
        )


    def _load_data(self) -> pd.DataFrame:
        """
        Load market data for backtest period.
        
        Requests data based on strategy requirements (price types and columns).
        Raises on missing or invalid data.
        
        Returns
        -------
        pd.DataFrame
            Market data indexed by timestamp with requested columns.
            
        """

        signal_type    = self.strategy.data_requirements().price_type
        execution_type = self.config.execution.price_type
        mtm_type       = self.config.execution.mtm_price_type
        price_types    = tuple[PriceType, ...]({signal_type, execution_type, mtm_type})
        
        data = prepare_data(
            config      = self.config.data,
            price_types = price_types,
            columns     = self.strategy.data_requirements().columns,
        )
                
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
        """

            
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


    def _size_pos(self, signal: Signal | None) -> float | None:
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
            Signed target size as fraction of equity (e.g., 0.5 = 50% long).
            None means no signal based action.
            
        """
        if signal is None:
            return None   # No new signal

        if signal.direction == SignalDirection.FLAT:
            return 0.0

        return simple_size(signal) 


    def _apply_risk(
            self, 
            target: float | None,
            state: PortfolioSnapshot) -> float | None:
        """
        Apply risk adjustments to a target position.

        Currently a passthrough (no risk limits). Designed as the integration
        point for a future RiskEngine that may reduce, block, or adjust targets.

        Parameters
        ----------
        target : float or None
            Raw target position from the position sizer as a fraction of
            equity. None means no sizing action.
        state : PortfolioSnapshot
            Current portfolio state available for risk checks.

        Returns
        -------
        float or None
            Risk-adjusted target. May differ from ``target`` once a risk
            engine is wired; None skips submitting an order.
        """
        return target
    
    def _compute_delta_notional(
        self,
        target_fraction: float,
        equity: float,
        position_units: float,
        price: float,
        ) -> float:
        """
        Convert a target fraction into a signed notional delta for the engine.
        Accounts for current position and engine.pending_notional. On a direction
        reversal vs pending, cancels pending and resizes against the flat book.

        Parameters
        ----------
        target_fraction : float
            The desired signed position as a fraction of equity.
        equity : float
            The current equity of the portfolio.
        position_units : float
            The current position units of the portfolio.
        price : float
            The mtm price of the asset.

        Returns
        -------
        float | None
            Delta to pass to ``submit``, or None to skip (ruin / no-op).
        """

        tol = self._engine._UNITS_TOLERANCE

        if not math.isfinite(target_fraction) or not math.isfinite(price) or price <= 0:
            raise ValueError(
                f"Invalid target_fraction/price: {target_fraction}, {price}"
            )
        # Can't open / increase when ruined; flatten is handled separately in _flatten
        if equity <= tol and abs(target_fraction) > tol:
            logger.warning(
                "Skipping submit: equity %.4g with non-flat target %.4g",
                equity, target_fraction,
            )
            return None

        current_notional = position_units * price
        pending = self._engine.pending_notional
        target_notional = target_fraction * max(equity, 0.0)  # flat→0 if ruined
        delta = target_notional - current_notional - pending

        # Reversal vs in-flight orders: cancel, then size vs book only
        if (
            abs(pending) > tol
            and abs(delta) > tol
            and np.sign(delta) != np.sign(pending)
        ):
            self._engine.cancel_all_pending()
            delta = target_notional - current_notional

        if abs(delta) <= tol:
            return None

        return float(delta)

    def _flatten(
        self,
        timestamp: pd.Timestamp,
        bar: pd.Series,
        t: int,
        price: float,
    ) -> None:
        """
        Immediately submit and fill a flat target; apply fills into the open bar.

        Sets the engine bar index before submit so ``immediate`` queues at ``t``.
        """
        self._engine._current_bar = t

        delta = -self._portfolio.position_units * price
        if abs(delta) > self._engine._UNITS_TOLERANCE:
            self._engine.submit(timestamp, delta_notional=delta, immediate=True)

        fills = self._engine.execute_pending(bar, t)
        self._portfolio.apply_fills(fills)

        if not self._portfolio.is_flat():
            logger.warning(
                "Flatten at %s left residual position_units=%.6g "
                "(MTM vs fill-price sizing dust is possible)",
                timestamp,
                self._portfolio.position_units,
            )
        else:
            logger.info("Flattened portfolio at %s", timestamp)

