import logging
import numpy as np
from functools import cached_property

from .base import MetricsGroup
from .utils import compute_sharpe

logger = logging.getLogger(__name__)

class PositionMetrics(MetricsGroup):
    pass