from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class DataConfig:
    """
    Immutable specification of a market-data window.

    Used by the backtester and prepare layer to locate and load klines
    (and optionally funding) for a single symbol over a date range.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. ``"BTCUSDT"``.
    interval : int
        Bar interval in minutes. Must be positive.
    start : datetime
        Inclusive start of the requested window.
    end : datetime
        Exclusive end of the requested window. Must be strictly after ``start``.

    Raises
    ------
    ValueError
        If ``interval`` is not positive, or ``start >= end``.
    """
    symbol:   str
    interval: int
    start:    datetime
    end:      datetime

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start}) must be strictly before end ({self.end})"
            )