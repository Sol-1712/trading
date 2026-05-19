import pandas as pd
from strategy_engine.core.signal import Signal, SignalDirection


def simple_size(
    signals:         list[Signal],
    max_position:    float = 1.0,    # fraction of capital, 1.0 = fully in
) -> pd.Series:
    """
    Quick and dirty position sizer.
    Converts signals to a signed position series.
    LONG  → +max_position
    SHORT → -max_position
    FLAT  →  0.0

    Signal strength is ignored for now — fully in or fully out.
    Replace with proper PositionSizer when backtester is validated.
    """
    index  = [s.timestamp for s in signals]
    values = []

    for signal in signals:
        if signal.direction is SignalDirection.LONG:
            values.append(max_position * 0.5 * signal.strength)
        elif signal.direction is SignalDirection.SHORT:
            values.append(-max_position * 0.5 * signal.strength)
        else:
            values.append(0.0)

    return pd.Series(values, index=index, name='position')