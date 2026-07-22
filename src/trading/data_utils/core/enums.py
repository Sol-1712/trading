from enum import Enum


class PriceType(str, Enum):
    """
    Price series used for signal generation and execution simulation.

    Inherits from ``str`` so ``PriceType.LAST == "last"`` is True, and
    YAML / string comparisons work without explicit coercion at each callsite.

    Members
    -------
    LAST
        Last traded price OHLCV.
    MARK
        Mark price OHLC.
    INDEX
        Index price OHLC.
    """
    LAST  = "last"
    MARK  = "mark"
    INDEX = "index"


class DataType(str, Enum):
    """
    Dataset kind used for path construction and loading.

    Members
    -------
    KLINES
        OHLC(V) bar data under ``klines/{interval}m/{price_type}/``.
    FUNDING
        Funding-rate history under ``funding/``.
    """
    KLINES = "klines"
    FUNDING = "funding"