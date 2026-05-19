from strategy_engine.strategies.directional import DirectionalStrategy
from strategy_engine.features.trend         import MAType, MovingAverage
from strategy_engine.core.signal            import Signal
from strategy_engine.strategies             import StrategyConfig

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class MACrossoverConfig(StrategyConfig):
    """
    Configuration for a dual moving average crossover strategy.

    Inherits
    --------
    strategy_id        : str
    signal_price_types : tuple[PriceType, ...]

    Parameters
    ----------
    fast_period : int
        Lookback period for the fast MA.
    fast_type : MAType
        MA type for the fast line (SMA or EMA).
    slow_period : int
        Lookback period for the slow MA.
    slow_type : MAType
        MA type for the slow line (SMA or EMA).
    """
    fast_period: int    = 30
    fast_type:   MAType = MAType.SMA
    slow_period: int    = 90
    slow_type:   MAType = MAType.SMA

    def __post_init__(self) -> None:
        if self.fast_period <= 0:
            raise ValueError(f"fast_period must be > 0, got {self.fast_period}")
        if self.slow_period <= 0:
            raise ValueError(f"slow_period must be > 0, got {self.slow_period}")
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period ({self.fast_period}) must be "
                f"< slow_period ({self.slow_period})"
            )


class MACrossover(DirectionalStrategy[MACrossoverConfig]):
    """
    Dual moving average crossover strategy.

    Generates a long signal when the fast MA crosses above the slow MA,
    and a short signal when it crosses below. Signal strength is
    proportional to the percentage spread between the two MAs.
    """

    def __init__(self, config: MACrossoverConfig) -> None:
        super().__init__(config)
        col        = self._resolve_column("close")
        self.fast  = config.fast_type.build(config.fast_period, col)
        self.slow  = config.slow_type.build(config.slow_period, col)

    def _build_features(self) -> list[MovingAverage]:
        return [self.fast, self.slow]

    def generate_signals(self, df: pd.DataFrame) -> list[Signal]:
        fast_vals = df[self.fast.name]
        slow_vals = df[self.slow.name]

        signals = []

        for ts, f, s in zip(df.index, fast_vals, slow_vals):
            if pd.isna(f) or pd.isna(s):
                continue

            strength = min(abs(f - s) / s * 100, 1.0)

            if f > s:
                signals.append(self._long(strength, ts))
            elif f < s:
                signals.append(self._short(strength, ts))
            else:
                signals.append(self._flat(ts))

        return signals
