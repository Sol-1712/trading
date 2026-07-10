from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from trading.data_utils.core import PROJECT_ROOT


@dataclass(frozen=True)
class RunContext:
    run_id:   str 
    run_dir: Path

    @classmethod
    def create(cls, base_dir: Path = PROJECT_ROOT / 'runs'):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        uid = uuid4().hex[:6]

        run_id = f"{timestamp}_{uid}"
        run_dir = base_dir / run_id

        run_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            run_id=run_id,
            run_dir=run_dir,
        )