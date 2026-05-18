from enum import Enum


class PriceType(str, Enum):
    """
    The price series to use for execution and returns calculation.
    
    MARK  : Mark price — used for liquidation, typically smoothed.
            Preferred for perp backtests (avoids index/last manipulation).
    INDEX : Index price — underlying spot composite. 
            Useful for funding-related analysis.
    LAST  : Last traded price — raw exchange price.
            Noisier; closer to actual execution for spot.
    """
    MARK  = "mark"
    INDEX = "index"
    LAST  = "last"


class DataType(str, Enum):
    KLINE = "kline"
    FUNDING = "funding"



class Exchange(str, Enum):
    BYBIT = "bybit"
    BINANCE = "binance"