from __future__ import annotations

from dataclasses import asdict, dataclass
import yaml
from datetime import datetime, timezone
from uuid import uuid4
import json
from pathlib import Path
import subprocess


from trading.data_utils.core import PROJECT_ROOT
from trading.strategy_engine import STRATEGY_REGISTRY
from .serialisation import serialize


@dataclass(frozen=True)
class RunContext:
    run_id:          str 
    run_dir:         Path
    git_commit_hash: str
    git_dirty:       bool
    created_at:      str

    @classmethod
    def create(
        cls, 
        base_dir: Path = PROJECT_ROOT / 'runs'
        ) -> RunContext:
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        uid = uuid4().hex[:6]

        run_id = f"{timestamp}_{uid}"
        run_dir = base_dir / run_id

        run_dir.mkdir(parents=True, exist_ok=True)

        context =  cls(
            run_id=run_id,
            run_dir=run_dir,
            git_commit_hash=_get_git_commit_hash(),
            git_dirty=_get_git_dirty(),
            created_at = now.isoformat()
        )

        context._save_self()
        return context
    

    def save_configs(self, backtest_config, strategy_config) -> Path:
        payload = {
            "backtest": serialize(backtest_config),
            "strategy": {
                "type": self._strategy_type_key(strategy_config),
                **serialize(strategy_config),
            },
        }

        path = self.run_dir / "run_config.yaml"

        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

        return path


    def _save_self(self) -> None:
        manifest = asdict(self)
        manifest["run_dir"] = str(self.run_dir)
        with open(self.run_dir / "context.json", "w") as f:
            json.dump(manifest, f, indent=2)


    def _strategy_type_key(self, strategy_config) -> str:
        for key, (config_cls, _) in STRATEGY_REGISTRY.items():
            if isinstance(strategy_config, config_cls):
                return key

        raise ValueError(
            f"Unknown strategy config type: {type(strategy_config).__name__}"
        )


def _get_git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def _get_git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True, cwd=PROJECT_ROOT,
    )
    return bool(result.stdout.strip())