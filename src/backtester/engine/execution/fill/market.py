from .base import FillModel, Order, Fill

import pandas as pd
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
            raise ValueError(f"Invalid fill_price: {fill_price}")
        
        units = order.delta_notional / fill_price
        timestamp = cast(pd.Timestamp, bar.name).to_pydatetime()
        
        fill = Fill(
            placed_at=order.placed_at,
            filled_at=timestamp,
            units_filled=units,
            fill_price=fill_price
        )
        return fill