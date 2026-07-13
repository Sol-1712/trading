from dataclasses import dataclass
import yaml
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

from trading.data_utils.core import PROJECT_ROOT
from .serialisation import serialize


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
    

    def save_configs(
        self,
        backtest_config,
        strategy_config
    ):
        self._save_yaml(name = 'backtest_config', obj = backtest_config)
        self._save_yaml(name = 'strategy_config', obj = strategy_config)


    def _save_yaml(self, name: str, obj) -> Path:
        """
        Save an object as a YAML file inside this run directory.
        """

        path = self.run_dir / f"{name}.yaml"

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                serialize(obj),
                f,
                sort_keys=False,
            )

        return path