from backtester.engine.execution      import ExecutionConfig, ExecutionEngine
from backtester.engine.execution.fill import Fill, Order
from backtester.portfolio.base        import Portfolio, PortfolioSnapshot

import pandas as pd
from datetime import datetime
from typing import ClassVar

class PerpDirectionalEngine(ExecutionEngine):
    """
    Execution engine for directional perpetual futures strategies.

    Responsibilities
    ----------------
    - Convert target fractions → concrete orders in units
    - Queue orders for execution after delay_bars
    - Attempt fills each bar via fill model
    - Handle partial fills — order stays pending until fully filled
    - Manages stale orders

    Not responsible for
    -------------------
    - Portfolio accounting (Portfolio.step)
    - Signal generation or position sizing
    - Price column resolution (passed in by runner)
    """

    _FRACTION_TOLERANCE: ClassVar[float] = 1e-9

    def __init__(self, config: ExecutionConfig) -> None:
        super().__init__(config)


    def submit(
        self,
        target_fraction: float,
        state:           PortfolioSnapshot,
        bar:             pd.Series,
        t:               int,
    ) -> None:
        """
        Convert target position to a concrete order and queue for execution.

        Called at end of bar t. Order executes at bar t + delay_bars.

        Parameters
        ----------
        target_fraction : float
            Desired position as signed fraction of equity.
            Received from position constructor + risk engine.
        state : PortfolioSnapshot
            Current portfolio state. Provides equity and position_fraction.
        price : float
            Current bar's execution price.
            Used for fraction → units conversion only.
            Actual fill price is determined by fill model at execution bar.
        timestamp : datetime
            Bar timestamp. Recorded on order for audit trail.
        t : int
            Current bar index. Order scheduled for t + delay_bars.
        """
        if state.equity <= 0.0:
            return

        # Delta computed against expected position, not actual
        expected_fraction = state.target_fraction + self._pending_fraction
        delta_fraction    = target_fraction - expected_fraction

        if abs(delta_fraction) <= self._FRACTION_TOLERANCE:
            return  # pending orders will get us close enough

        delta_units = (delta_fraction * state.equity) / price

        # Cancel all pending if direction reverses or target significantly changes
        if self._sign_changed(delta_fraction) or self._stale_pending():
            self._cancel_all_pending()
            # Recompute delta against actual position only
            delta_fraction = target_fraction - state.position_fraction
            delta_units    = (delta_fraction * state.equity) / price

        order    = Order(
            placed_at       = timestamp,
            delta_units     = delta_units,
            remaining_units = delta_units,
        )
        exec_bar = t + self.config.delay_bars
        self._queue[exec_bar] = order
        self._pending_fraction += delta_fraction



        raise NotImplementedError

    def execute_pending(self, bar: pd.Series) -> list[Fill]:
        raise NotImplementedError


    def _cancel_all_pending(self) -> None:
        self._queue.clear()
        self._pending.clear()
