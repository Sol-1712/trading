from trading.backtester.execution      import ExecutionEngine
from trading.backtester.fill           import Fill, Order
from trading.backtester.portfolio.base import PortfolioSnapshot
from trading.backtester.engine.config_bases  import ExecutionConfig

import pandas as pd
import numpy as np
from typing import ClassVar
import logging
import math

logger = logging.getLogger(__name__)

class PerpDirectionalEngine(ExecutionEngine):
    """
    Perpetual directional execution engine for single-asset / multi-asset backtesting.

    Inherits
    --------
    ExecutionEngine
        Core execution infrastructure including:
        - config (ExecutionConfig)
        - fill model interface (MarketFillModel or custom)
        - order queueing system
        - pending order management state

        Specifically inherited attributes:
        - self.config
        - self._fill_model
        - self._queue (bar-indexed order queue)
        - self._pending (actively processed orders)
        - self._pending_fraction (execution state tracking)
        - self._current_bar
        - self._active

    Extends
    -------
    This class implements perpetual-specific execution logic, including:
    - order lifecycle management under continuous trading assumptions
    - fill simulation for perp-style markets
    - handling of partial fills
    - conversion of strategy targets into executable orders

    Notes
    -----
    This engine assumes a time-discrete simulation model (bar-based),
    but treats execution as continuous within each bar via a fill model.

    State in this engine is strictly execution-layer state and does not
    contain strategy or portfolio logic.
    """

    _FRACTION_TOLERANCE: ClassVar[float] = 1e-9

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__(config)


    def submit(
        self,
        target_fraction: float,
        state:           PortfolioSnapshot,
        price:           float,
    ) -> None:
        """
        Convert target position to a concrete order and queue for execution.
        
        PRICE SEMANTICS:
        - target_fraction is calculated using mark_close (MTM price)
        - This represents desired exposure in mark value
        - The order will execute at last_open (fill model price)
        - This is intentional: target is mark-based, execution is at market
        
        Called at end of bar t. Attempt order fill at bar t + delay_bars.

        Parameters
        ----------
        target_fraction : float
            Desired position as signed fraction of equity.
            Received from position constructor + risk engine.
        state : PortfolioSnapshot
            Current portfolio state. Provides equity and position_fraction.
        price : float
            Execution price
        """

        if state.equity <= 0.0:
            logger.warning(
                "Cannot submit order: equity %.2f at %s — skipping.",
                state.equity, state.timestamp,
            )
            return
    
        if not math.isfinite(target_fraction):
            raise ValueError(f"Non-finite target_fraction: {target_fraction}")
            
        current_fraction  = (state.position_units * price) / state.equity
        pending_fraction  = self._pending_notional / state.equity
        expected_fraction = current_fraction + pending_fraction
        delta_fraction    = target_fraction - expected_fraction

        if self._is_reversal(delta_fraction):
            self._cancel_all_pending()
            delta_fraction = target_fraction - current_fraction

        if abs(delta_fraction) <= self._FRACTION_TOLERANCE:
            return  # pending orders will get us close enough

        delta_notional = delta_fraction * state.equity
        exec_bar = self._current_bar + self.config.delay_bars

        order = Order(
            placed_at          = state.timestamp,
            exec_bar           = exec_bar,
            delta_notional     = delta_notional,
            remaining_notional = delta_notional,
        )

        self._queue[exec_bar] = order
        self._pending_notional += delta_notional


    def execute_pending(self, bar: pd.Series , t: int) -> list[Fill]:
        """
        Attempt fills on all pending orders due at or before bar t.

        Called at the START of each timestep, before signal generation
        and before submit(). Sets _current_bar so submit() knows
        which bar it is operating on.

        Parameters
        ----------
        bar : pd.Series
            Current bar's full OHLCV data. Passed directly to fill model.
        t : int
            Current bar index.

        Returns
        -------
        list[Fill]
            All fills that occurred this bar. May be empty.
        """
        self._current_bar = t

        # Move all orders due at or before t from queue → active
        # <= t handles data gaps — no order silently expires
        due = [k for k in self._queue if k <= t]
        for k in sorted(due):
            self._active.append(self._queue.pop(k))

        if not self._active:
            return []

        fills     = []
        remaining = []

        for order in self._active:
            fill = self._fill_model.attempt_fill(order, bar)

            if fill is None:
                # No fill this bar — order stays active
                remaining.append(order)
                continue
            
            signed_notional_filled = fill.units_filled * fill.fill_price
            notional_filled = abs(signed_notional_filled)

            if notional_filled >= abs(order.remaining_notional) - self._FRACTION_TOLERANCE:
                # Fully filled — remove from active, clear pending tracking
                fills.append(fill)
                self._pending_notional -= order.delta_notional

            else:
                # Partial fill: create new order with remaining as new delta
                fills.append(fill)
                remaining_notional = order.remaining_notional - signed_notional_filled
                remaining.append(Order(
                    placed_at          = order.placed_at,
                    exec_bar           = order.exec_bar,
                    delta_notional     = remaining_notional,  # CHANGED: remaining becomes new delta
                    remaining_notional = remaining_notional
                ))
                self._pending_notional -= signed_notional_filled

        self._active = remaining
        return fills


    def _is_reversal(self, delta_fraction: float) -> bool:
        if abs(self._pending_notional) <= self._FRACTION_TOLERANCE:
            return False
        # True if delta_fraction and _pending_notional point opposite directions
        pending_sign = np.sign(self._pending_notional)
        target_sign = np.sign(delta_fraction)
        
        return pending_sign != target_sign  