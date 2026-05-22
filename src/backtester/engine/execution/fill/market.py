
from .base import FillModel, Order, Fill

class MarketFillModel(FillModel):
    """
    Simplest possible fill model.

    Assumes:
    - Orders fill immediately at the open of the execution bar
    - Always fully filled — no partial fills
    - No slippage beyond what open price reflects

    This is the correct default for bar-level backtesting with delay_bars >= 1.
    Filling at close would introduce look-ahead bias.
    """

    def fill(self, order: Order) -> Fill:

        fill = Fill(
            order.timestamp,
            )



        pass