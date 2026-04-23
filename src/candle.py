from dataclasses import dataclass
from datetime import datetime



@dataclass(frozen=True)
class Candle:
    symbol:     int
    open_time:  datetime
    close_time: datetime
    interval:   int
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float
    turnover:   float
    complete:   bool = True   # False if bar is still forming


    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


    @property
    def body(self) -> float:
        return abs(self.close - self.open)


    @property
    def range(self) -> float:
        return self.high - self.low


    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3


    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open
    

    @property
    def is_bearish(self) -> bool:
        return self.close <= self.open
    

