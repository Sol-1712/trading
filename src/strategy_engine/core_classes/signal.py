from dataclasses import dataclass, field
from datetime import datetime
from .enums import SignalDirection


@dataclass
class Signal:
    direction: SignalDirection
    strength:  float
    timestamp: datetime
    source:    str  = ""
    metadata:  dict[str, any] = field(default_factory=dict)