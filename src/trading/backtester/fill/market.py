from trading.data_utils.core.enums import PriceType

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

    def __init__(self, price_type: PriceType = PriceType.LAST) -> None:
        self._price_type = price_type
        self._open_col   = f"{price_type.value}_open"


    @property
    def price_type(self) -> PriceType:
        return self._price_type


    def attempt_fill(self, order: Order, bar: pd.Series) -> Fill:

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