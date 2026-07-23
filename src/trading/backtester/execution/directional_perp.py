from trading.backtester.execution      import ExecutionEngine
from trading.backtester.fill           import Fill, Order

from trading.backtester.engine.config_bases  import ExecutionConfig

import pandas as pd
from typing import ClassVar
import logging
from datetime import datetime

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
        timestamp:       pd.Timestamp | datetime,
        delta_notional:  float,
        immediate:       bool = False,
    ) -> None:
        """
        Add a new order to the queue.

        Parameters
        ----------
        timestamp : pd.Timestamp | datetime
            Timestamp of the current bar.
            Converted to datetime for the order.
        delta_notional:  float
            The desired signed notional value of the order.
        immediate:       bool = False,
            Whether to submit the order immediately.
        """
        
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        if abs(delta_notional) <= self._UNITS_TOLERANCE:
            return

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

        self._queue.setdefault(exec_bar, []).append(order)
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

        # 1. Promote due orders (key = exec bar)
        due = sorted(k for k in self._queue if k <= t)
        for k in due:
            self._active.extend(self._queue.pop(k))

        if not self._active:
            return []

        #2 Attempt fills
        fills: list[Fill] = []
        still_active: list[Order] = []

        for order in self._active:
            fill = self._fill_model.attempt_fill(order, bar)
            
            if fill is None:
                # No fill this bar — order stays active
                still_active.append(order)
                continue
            
            filled_notional = fill.units_filled * fill.fill_price  
            fills.append(fill)
            self._pending_notional -= filled_notional

            leftover = order.remaining_notional - filled_notional
            if abs(leftover) <= self._UNITS_TOLERANCE:
                # Fully filled — leave off still_active
                continue

            still_active.append(Order(
                placed_at          = order.placed_at,
                exec_bar           = order.exec_bar,
                delta_notional     = leftover,
                remaining_notional = leftover
            ))


        self._active = still_active
        return fills

