import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT_LOGGER = "trading"

def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    overwrite: bool = False
) -> None:
    """
    Configure logging for the entire platform.
    Call once at the top of any script or notebook entry point.
    
    Params
    ------
    console_level : what severity to show in the terminal
    file_level    : what severity to write to disk (usually more verbose)
    log_dir       : if provided, also write logs to this directory
    run_id        : appended to the log filename for traceability
    """
    root = logging.getLogger(PROJECT_ROOT_LOGGER)

    log_path = log_dir / f"run_{run_id}.log"
    if log_path.exists() and not overwrite:
        raise FileExistsError(
            f"Log file already exists for run_id '{run_id}'. "
            f"Pass overwrite=True or use a unique run_id."
        )
    
    if root.handlers:
        # Guard against double-initialisation (e.g. in notebooks)
        return

    root.setLevel(logging.DEBUG)  # let handlers decide what they show

    formatter = _build_formatter()

    root.addHandler(_console_handler(console_level, formatter))

    if log_dir is not None:
        root.addHandler(_file_handler(log_dir, file_level, formatter, run_id))


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _console_handler(level: int, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _file_handler(
    log_dir: Path,
    level: int,
    formatter: logging.Formatter,
    run_id: Optional[str],
) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_id}" if run_id else ""
    log_path = log_dir / f"run{suffix}.log"

    # RotatingFileHandler prevents logs from growing unbounded
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler