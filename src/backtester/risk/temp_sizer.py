import pandas as pd
from strategy_engine.core.signal import Signal, SignalDirection


def simple_size(
    signal:          Signal,
    max_position:    float = 1.0,    # fraction of capital, 1.0 = fully in
) -> float | None:
    """
    Quick and dirty position sizer.
    Converts signals to a signed position series.
    LONG  → +max_position
    SHORT → -max_position
    FLAT  →  0.0

    Signal strength is ignored for now — fully in or fully out.
    Replace with proper PositionSizer when backtester is validated.
    """


    if signal is None: # No signal so no change (build logic later)
        return None

    if signal.direction is SignalDirection.LONG:
        return float(max_position * 0.5 * signal.strength)

    elif signal.direction is SignalDirection.SHORT:
        return float(-max_position * 0.5 * signal.strength)

    else: # Flat signal, need to go flat
        return 0.0

