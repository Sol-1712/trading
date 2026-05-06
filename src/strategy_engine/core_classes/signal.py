from dataclasses import dataclass, field
from datetime import datetime
from strategy_engine.core_classes import SignalDirection
from typing import Any


@dataclass
class Signal:
    direction: SignalDirection
    strength:  float
    timestamp: datetime
    source:    str  = ""
    metadata:  dict[str, Any] = field(default_factory=dict)