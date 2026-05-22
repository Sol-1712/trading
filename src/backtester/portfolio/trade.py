from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Trade:
    entry_time:   datetime
    exit_time:    datetime | None    # None = still open
    side:         Literal["long", "short"]
    units:        float              # absolute
    entry_price:  float
    exit_price:   float | None
    pnl:          float              # cumulative, updated as bars pass
    fees:         float
    duration:     int                # bars