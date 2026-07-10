import logging
from trading.strategy_engine.core.signal import Signal, SignalDirection

logger = logging.getLogger(__name__)

def simple_size(
    signal:          Signal,
    max_position:    float = 1.0,
) -> float:
    """
    Stub sizer. Returns full max_position on any directional signal.
    Strength scaling belongs in a proper volatility-targeting sizer.
    """
    if signal.direction is SignalDirection.LONG:
        return float(max_position)

    elif signal.direction is SignalDirection.SHORT:
        return float(-max_position)

    return 0.0
