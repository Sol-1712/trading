from crypto_quant.strategy_engine.strategies.directional import DirectionalStrategy
from crypto_quant.strategy_engine.features.trend         import MAType, MovingAverage
from crypto_quant.strategy_engine.core.signal            import Signal
from crypto_quant.strategy_engine.strategies             import StrategyConfig

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
    fast_period: int    = 20
    fast_type:   MAType = MAType.EMA
    slow_period: int    = 50
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
        price_col  = self._resolve_column("close") # We want PriceType_close 
        self.fast  = config.fast_type.build(config.fast_period, price_col)
        self.slow  = config.slow_type.build(config.slow_period, price_col)


    def _build_features(self) -> list[MovingAverage]:
        """
        List of features (MovingAverages)
        """
        return [self.fast, self.slow]


    def generate_signals(self, df: pd.DataFrame) -> list[Signal | None]:

        fast_vals = df[self.fast.name]
        slow_vals = df[self.slow.name]

        signals: list[Signal | None] = [None]

        prev_f = fast_vals.iloc[0]
        prev_s = slow_vals.iloc[0]

        for ts, f, s in zip(
            df.index[1:],
            fast_vals.iloc[1:],
            slow_vals.iloc[1:]
        ):

            if any(pd.isna(x) for x in (prev_f, prev_s, f, s)):
                signals.append(None)
                if not pd.isna(f): prev_f = f   
                if not pd.isna(s): prev_s = s
                continue

            prev_spread = prev_f - prev_s
            curr_spread = f - s

            strength = min(abs(curr_spread) / abs(s), 1.0)

            bull_cross = (
                prev_spread <= 0 and curr_spread > 0
            )

            bear_cross = (
                prev_spread >= 0 and curr_spread < 0
            )

            if bull_cross:
                signals.append(self._long(strength, ts))

            elif bear_cross:
                signals.append(self._short(strength, ts))

            else:
                signals.append(None)

            prev_f, prev_s = f, s

        return signals
