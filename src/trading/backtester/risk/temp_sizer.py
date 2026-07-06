import logging
import pandas as pd
from trading.strategy_engine.core.signal import Signal, SignalDirection

logger = logging.getLogger(__name__)

def simple_size(
    signal:          Signal | None,
    max_position:    float = 1.0,
) -> float | None:
    """
    Convert signal to a target position fraction.
    
    Simple sizing: full in (LONG), full out (SHORT), or flat (FLAT/None).
    Signal strength modulates position size. No validation of strength values.
    
    TEMPORARY: Designed as placeholder for proper PositionSizer when backtester
    is validated.
    
    Parameters
    ----------
    signal : Signal | None
        Signal to convert to position. None returns None (no action).
    max_position : float, default 1.0
        Maximum position fraction (1.0 = 100% of equity). Scaled by signal strength.
        Must be positive.
        
    Returns
    -------
    float | None
        Signed target position as fraction of equity:
        - LONG direction: +max_position * 0.5 * strength
        - SHORT direction: -max_position * 0.5 * strength
        - FLAT/None: 0.0 or None
        - None signal: None (no action)
        
    Raises
    ------
    ValueError
        If max_position is not positive.
    """
    if max_position <= 0:
        raise ValueError(f"max_position must be positive, got {max_position}")

    if signal is None:
        return None

    if signal.direction is SignalDirection.LONG:
        position = float(max_position * 0.5 * signal.strength)
        logger.debug("Sized LONG position: %.3f", position)
        return position

    elif signal.direction is SignalDirection.SHORT:
        position = float(-max_position * 0.5 * signal.strength)
        logger.debug("Sized SHORT position: %.3f", position)
        return position

    else:  # FLAT or unknown
        logger.debug("Signal is FLAT, returning 0.0")
        return 0.0

