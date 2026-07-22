from trading.data_utils.core.enums import PriceType

from .base import FillModel, Order, Fill

import pandas as pd
import numpy  as np
from typing import cast


class MarketFillModel(FillModel):
    """
    Immediate full-fill model at the execution bar open.

    Assumes:
    - Fills occur immediately at ``{price_type}_open`` on the bar
    - Always fully filled — no partial fills
    - No slippage beyond what the open price already reflects

    Uses ``PriceType.LAST`` (last-trade open column).
    """
    price_type = PriceType.LAST

    def __init__(self, fee_rate: float) -> None:
        super().__init__(fee_rate)
        self._open_col   = f"{self.price_type.value}_open"


    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:
        """
        Fill the entire order notional at the bar's last-trade open.

        Parameters
        ----------
        order : Order
            Order whose ``delta_notional`` is fully converted to units.
        bar : pd.Series
            Current bar; must include the model's open price column.

        Returns
        -------
        Fill
            Full fill at open, with fees ``abs(units) * fill_price * fee_rate``.

        Raises
        ------
        KeyError
            If the required open price column is missing from ``bar``.
        ValueError
            If fill price is non-positive or unit sign disagrees with
            order notional.
        """

        if self._open_col not in bar.index:
            raise KeyError(
                f"Fill model requires column '{self._open_col}'. "
                f"Available: {bar.index.tolist()}"
            )
        
        fill_price = float(bar[self._open_col])
        
        if fill_price <= 0:
            raise ValueError(
                f"Invalid fill_price {fill_price} at bar {bar.name}. "
                f"Cannot calculate units for order {order.delta_notional}"
    )
        
        units = order.delta_notional / fill_price # Always fill entire order
        expected_sign = np.sign(order.delta_notional)
        actual_sign = np.sign(units)

        if expected_sign != actual_sign and expected_sign != 0:
            raise ValueError(
                f"Fill sign mismatch: order {order.delta_notional} → fill {units}"
    )
        timestamp = cast(pd.Timestamp, bar.name).to_pydatetime()

        fees = abs(units) * fill_price * self.fee_rate
        fill = Fill(
            placed_at=order.placed_at,
            filled_at=timestamp,
            units_filled=units,
            fill_price=fill_price,
            fees=fees
        )
        return fill