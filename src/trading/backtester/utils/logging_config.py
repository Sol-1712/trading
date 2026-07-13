import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT_LOGGER = "trading"

def setup_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    run_dir: Optional[Path] = None,
) -> None:
    """
    Configure logging for the entire platform.
    Call once at the top of any script or notebook entry point.
    
    Params
    ------
    console_level : what severity to show in the terminal
    file_level    : what severity to write to disk
    run_dir       : if provided, also write logs to this directory
    """
    root = logging.getLogger(PROJECT_ROOT_LOGGER)

    if root.handlers:
        # Guard against double-initialisation (e.g. in notebooks)
        return

    root.setLevel(logging.DEBUG)  # let handlers decide what they show

    formatter = _build_formatter()

    root.addHandler(_console_handler(console_level, formatter))

    if run_dir is not None:
        log_path = run_dir / "run.log"

        if log_path.exists():
            raise FileExistsError(
                f"Log file already exists: {log_path}"
            )

        root.addHandler(
            _file_handler(
                log_path,
                file_level,
                formatter,
            )
        )


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
    log_path: Path,
    level: int,
    formatter: logging.Formatter,
) -> logging.Handler:

    # RotatingFileHandler prevents logs from growing unbounded
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    
    return handler