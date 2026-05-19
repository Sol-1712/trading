from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class DataConfig:
    symbol:   str
    interval: int
    start:    datetime
    end:      datetime

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError(f"interval must be positive, got {self.interval}")
        if self.start >= self.end:
            raise ValueError(
                f"start ({self.start}) must be strictly before end ({self.end})"
            )