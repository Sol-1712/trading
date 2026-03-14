import numpy as np

class PositionMetrics:
    """
    Computes position-based metrics from a CoreStats object.

    This class provides key metrics:


    Attributes:
        core (CoreStats): Precomputed core statistics and returns from PnL.
    """

    def __init__(self, core):
        """
        Initializes PositionMetrics with a CoreStats object.

        Args:
            core (CoreStats): Object containing primitive statistics and returns.
        """
        self.core = core
