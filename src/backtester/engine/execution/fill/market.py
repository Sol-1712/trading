from .base import FillModel, Order, Fill

import pandas as pd
import numpy  as np
from typing import cast


class MarketFillModel(FillModel):
    """
    Simplest possible fill model.

    Assumes:
    - Orders fill immediately at the open of the execution bar
    - Always fully filled — no partial fills
    - No slippage beyond what open price reflects

    """

    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:

        if 'last_open' not in bar.index:
            raise KeyError(
                f"Column 'last_open' not found in bar. Available: {bar.index.tolist()}"
            )
        
        fill_price = bar['last_open']
        
        if fill_price <= 0:
            raise ValueError(
                f"Invalid fill_price {fill_price} at bar {bar.name}. "
                f"Cannot calculate units for order {order.delta_notional}"
    )
        
        units = order.delta_notional / fill_price
        expected_sign = np.sign(order.delta_notional)
        actual_sign = np.sign(units)

        if expected_sign != actual_sign and expected_sign != 0:
            raise ValueError(
                f"Fill sign mismatch: order {order.delta_notional} → fill {units}"
    )
        timestamp = cast(pd.Timestamp, bar.name).to_pydatetime()
        
        fill = Fill(
            placed_at=order.placed_at,
            filled_at=timestamp,
            units_filled=units,
            fill_price=fill_price
        )
        return fill