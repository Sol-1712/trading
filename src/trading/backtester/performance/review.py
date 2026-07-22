from __future__ import annotations

import pandas as pd
import json
import yaml
from pathlib import Path
from dataclasses import dataclass

from trading.data_utils.core import PROJECT_ROOT
from trading.backtester.performance import display_report
from trading.backtester.engine import BacktestConfig, build_config


@dataclass(frozen=True)
class RunReview:
    """
    Loaded artifacts from a completed run directory for offline review.

    Attributes
    ----------
    run_dir : str or Path
        Resolved path to the run folder.
    backtest_config : dict
        Raw ``backtest`` section from ``run_config.yaml``.
    strategy_config : dict
        Raw ``strategy`` section from ``run_config.yaml``.
    metrics : dict[str, float]
        Contents of ``report.json``.
    portfolio_history : pd.DataFrame
        Bar-level portfolio history from parquet.
    trades : pd.DataFrame
        Closed trades from parquet.
    """
    run_dir: str | Path
    backtest_config: dict
    strategy_config: dict
    metrics: dict[str, float]
    portfolio_history:  pd.DataFrame
    trades:             pd.DataFrame

    @classmethod
    def build(cls, run_dir: str | Path) -> RunReview:
        """
        Load configs, metrics, and history from a run directory.

        Parameters
        ----------
        run_dir : str or Path
            Run id relative to ``PROJECT_ROOT / runs`` (str), or an absolute Path.

        Returns
        -------
        RunReview
            Frozen container with all loaded run artifacts.
        """
        # run = RunReview.build(run_dir)

        run_dir = _resolve_run_dir(run_dir)
        with open(run_dir / "run_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        backtest_dict, strategy_dict = _split_config_config(config)
        metrics = _load_report_json(run_dir)
        portfolio_history = _load_portfolio_history(run_dir)
        trades = _load_trades(run_dir)

        return RunReview(
            run_dir=run_dir,
            backtest_config=backtest_dict,
            strategy_config=strategy_dict,
            metrics=metrics,
            portfolio_history=portfolio_history,
            trades=trades
        )

    def display(self) -> None:
        """
        Render the saved metrics report using the run's backtest config.
        """
        config = _build_backtest_config(self.backtest_config)
        display_report(self.metrics, config)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def _load_portfolio_history(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "portfolio_history.parquet")

def _load_trades(run_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(run_dir / "trades.parquet")

def _load_report_json(run_dir: str | Path) -> dict[str, float]:
    with open(run_dir / "report.json", "r") as f:
        return json.load(f)

def _split_config_config(config: dict) -> tuple[dict, dict]:
    return config["backtest"], config["strategy"]

def _build_backtest_config(backtest_dict: dict) -> BacktestConfig:
    return build_config(backtest_dict)

def _resolve_run_dir(run_dir: str | Path) -> Path:
    return PROJECT_ROOT / "runs" / run_dir if isinstance(run_dir, str) else run_dir