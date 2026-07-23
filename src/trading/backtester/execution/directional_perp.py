from trading.backtester.execution      import ExecutionEngine
from trading.backtester.fill           import Fill, Order

from trading.backtester.engine.config_bases  import ExecutionConfig

import pandas as pd
import numpy as np
from typing import ClassVar
import logging
import math

logger = logging.getLogger(__name__)

class PerpDirectionalEngine(ExecutionEngine):
    """
    Perpetual directional execution engine for bar-based backtests.

    Converts signed target position fractions into notional Orders, queues
    them with ``delay_bars``, and drives fill attempts via the configured
    FillModel. Handles partial fills and cancels pending orders on
    direction reversals.

    Notes
    -----
    Simulation is discrete by bar; fill models decide intra-bar execution.
    Engine state is execution-layer only — no strategy or portfolio logic.
    """

    _UNITS_TOLERANCE: ClassVar[float] = 1e-9

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__(config)


    def submit(
        self,
        timestamp:       pd.Timestamp,
        target_fraction: float,
        equity:          float,
        position_units:  float,
        price:           float,
        immediate:       bool = False,
    ) -> None:
        """
        Convert a target position into an Order and queue it for execution.

        Computes the notional delta between ``target_fraction`` and the
        position implied by current holdings plus pending notionals. On a
        direction reversal, cancels pending orders and recomputes against
        the flat current position. The order becomes eligible at
        ``current_bar + delay_bars``.

        Price semantics: ``target_fraction`` / sizing use the MTM price
        passed as ``price``; the fill model later executes at its own
        price series (e.g. last open).

        Parameters
        ----------
        timestamp : pd.Timestamp
            Timestamp of the current bar.
            Converted to datetime for the order.
        target_fraction : float
            Desired signed position as a fraction of equity.
        equity : float
            Current equity.
        position_units : float
            Current position units.
        price : float
            MTM price used to convert units ↔ fraction of equity.
        immediate:       bool = False,
            Whether to submit the order immediately.
        Raises
        ------
        ValueError
            If ``target_fraction`` is non-finite.
        """
        
        if not math.isfinite(target_fraction):
            raise ValueError("Non-finite target_fraction")

        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # --- Flat / flatten: never divide by equity ---
        if abs(target_fraction) <= self._UNITS_TOLERANCE:
            self.cancel_all_pending()  # don't leave size-ups hanging
            if abs(position_units) <= self._UNITS_TOLERANCE:
                return
            delta_notional = -position_units * price
            exec_bar = (
                self._current_bar if immediate
                else self._current_bar + self.config.delay_bars
            )
            order = Order(
                placed_at=timestamp,
                exec_bar=exec_bar,
                delta_notional=delta_notional,
                remaining_notional=delta_notional,
            )
            self._queue[exec_bar] = order
            self._pending_notional += delta_notional
            return



        if equity <= 0.0:
            logger.warning("Cannot submit: equity %.2f — skipping.", equity)
            return

        # --- Submit order --------------------------------------------
        current_fraction  = (position_units * price) / equity
        pending_fraction  = self._pending_notional / equity
        expected_fraction = current_fraction + pending_fraction
        delta_fraction    = target_fraction - expected_fraction

        if self._is_reversal(delta_fraction):
            self.cancel_all_pending()
            delta_fraction = target_fraction - current_fraction

        if abs(delta_fraction) <= self._UNITS_TOLERANCE:
            return  # pending orders will get us close enough

        delta_notional = delta_fraction * equity

        exec_bar = (
            self._current_bar if immediate
            else self._current_bar + self.config.delay_bars
        )

        order = Order(
            placed_at          = timestamp,
            exec_bar           = exec_bar,
            delta_notional     = delta_notional,
            remaining_notional = delta_notional,
        )

        self._queue[exec_bar] = order
        self._pending_notional += delta_notional


    def execute_pending(self, bar: pd.Series , t: int) -> list[Fill]:
        """
        Attempt fills on all orders due at or before bar ``t``.

        Sets ``_current_bar`` so subsequent ``submit`` calls know the
        simulation index. Orders with ``exec_bar <= t`` move from the
        queue into the active set (``<=`` so gaps do not silently drop
        orders). Partial fills leave a residual Order active.

        Parameters
        ----------
        bar : pd.Series
            Current bar data, passed through to the fill model.
        t : int
            Current bar index.

        Returns
        -------
        list[Fill]
            Fills that occurred this bar (may be empty).
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
            ### Need some sort of fee check here -> Based on available cash
            
            if fill is None:
                # No fill this bar — order stays active
                remaining.append(order)
                continue
            
            signed_notional_filled = fill.units_filled * fill.fill_price
            notional_filled = abs(signed_notional_filled)

            if notional_filled >= abs(order.remaining_notional) - self._UNITS_TOLERANCE:
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
        """
        Return True if ``delta_fraction`` opposes outstanding pending notional.

        Used to decide whether to cancel pending orders before submitting
        a new target that flips direction relative to in-flight exposure.
        """
        if abs(self._pending_notional) <= self._UNITS_TOLERANCE:
            return False
        # True if delta_fraction and _pending_notional point opposite directions
        pending_sign = np.sign(self._pending_notional)
        target_sign = np.sign(delta_fraction)
        
        return pending_sign != target_sign  