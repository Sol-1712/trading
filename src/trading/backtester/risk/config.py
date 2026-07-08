from __future__ import annotations

from dataclasses import dataclass




@dataclass(frozen=True)
class RiskConfig:
    """
    Portfolio-level risk constraints.

    Applied by the risk engine between position construction
    and order submission.
    """
    leverage_max: float = 100.0
    max_drawdown: float = 0.20
    max_position: float = 1.0

    def __post_init__(self) -> None:
        if self.leverage_max < 1.0:
            raise ValueError(f"leverage_max must be >= 1.0, got {self.leverage_max}")
        if not 0.0 < self.max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {self.max_drawdown}")
        if not 0.0 < self.max_position <= self.leverage_max:
            raise ValueError(
                f"max_position must be in (0, leverage_max], got {self.max_position}"
            )