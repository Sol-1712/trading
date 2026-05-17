from dataclasses import dataclass
from datetime import datetime
from strategy_engine.core import SignalDirection


@dataclass(frozen=True)
class Trade():

    # Identity
    symbol:        str
    strategy_id:   str

    # Entry
    entry_time:    datetime
    entry_price:   float        # price signal was generated at
    entry_fill:    float        # price actually filled at ← add this

    # Exit
    exit_time:     datetime
    exit_price:    float
    exit_fill:     float        # ← and this

    direction:     SignalDirection
    size_fraction: float
    size_dollars:  float
    gross_pnl:     float
    fees:          float
    net_pnl:       float
    bars_held:     int

    @property
    def slippage(self) -> float:
        """Total slippage cost across entry and exit."""
        entry_slip = abs(self.entry_fill - self.entry_price) * self.size_dollars / self.entry_price
        exit_slip  = abs(self.exit_fill - self.exit_price) * self.size_dollars / self.entry_price
        return entry_slip + exit_slip

    @property
    def return_pct(self) -> float:
        return self.net_pnl / self.size_dollars

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0