from dataclasses import dataclass
from strategy_engine.strategies import StrategyConfig


@dataclass
class MACrossoverConfig(StrategyConfig):
    strategy_id: str
    fast_window: int
    slow_window: int
    ma_type:     str
    

    def __post_init__(self):
        if self.fast_window <= 0:
            raise ValueError(f"short_window must be > 0, got {self.fast_window}")
        
        if self.slow_window <= 0:
            raise ValueError(f"long_window must be > 0, got {self.slow_window}")
        
        if self.fast_window >= self.slow_window:
            raise ValueError(
                f"fast_window ({self.fast_window}) must be "
                f"less than slow_window ({self.slow_window})"
            )