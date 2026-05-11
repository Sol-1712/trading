from strategy_engine.strategies.directional import DirectionalStrategy
from strategy_engine.features.registry      import FeatureRegistry
from strategy_engine.features.trend         import MAType, MovingAverage, SMA, EMA
from strategy_engine.core.signal            import Signal
from strategy_engine.strategies             import StrategyConfig

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class MACrossoverConfig(StrategyConfig):
    """
    Args
        fast_period: int
        fast_type:   MAType
        slow_period: int
        slow_type:   MAType
    
    """
    fast_period: int
    fast_type:   MAType
    slow_period: int
    slow_type:   MAType
    
    def __post_init__(self):
        if self.fast_period <= 0:
            raise ValueError(f"short_window must be > 0, got {self.fast_period}")
        
        if self.slow_period <= 0:
            raise ValueError(f"long_window must be > 0, got {self.slow_period}")
        
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_window ({self.fast_period}) must be "
                f"less than slow_window ({self.slow_period})"
            )


class MACrossover(DirectionalStrategy[MACrossoverConfig]):

    def __init__(self, config: MACrossoverConfig) -> None:
        super().__init__(config)
        """
        Allocates fast and slow MA objects. 
        e.g self.fast = SMA(30)
        """
        self.fast = self._build(config.fast_type, config.fast_period)
        self.slow = self._build(config.slow_type, config.slow_period)


    def _build(self, ma_type: MAType, period: int) -> MovingAverage:
        match ma_type:
            case MAType.SMA: return SMA(period)
            case MAType.EMA: return EMA(period)
            case _: raise ValueError(f"Unsupported MA type: {ma_type}")


    def required_features(self) -> list[str]:
        # List of all the features the strategy needs, as strings
        return [self.fast.name, self.slow.name]


    def register_features(self, registry: FeatureRegistry) -> None:
        # .register gets the name, just need to initialise objects.
        registry.register(self.fast)
        registry.register(self.slow)


    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        fast = df[self.fast.name]
        slow = df[self.slow.name]

        signals = []

        for ts, f, s in zip(df.index, fast, slow):
            if pd.isna(f) or pd.isna(s):
                continue

            strength = min(abs(f - s) / s * 100, 1.0) ### Have a look at this

            if f > s:
                signals.append(self._long(strength, ts))
            elif f < s:
                signals.append(self._short(strength, ts))    
            else:
                signals.append(self._flat(ts))

        return signals
    

    def on_start(self) -> None:
        raise NotImplementedError


    def on_stop(self) -> None:
        raise NotImplementedError