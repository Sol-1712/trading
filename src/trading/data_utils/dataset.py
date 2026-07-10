import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import yaml

from trading.data_utils.fetcher import BybitFetcher
from trading.data_utils.io import get_stored_range, save_partitioned_parquet
from trading.data_utils.core import PriceType, DataType, make_data_path, CONFIGS_ROOT

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class UpdateConfig:
    symbols:   tuple[str, ...]
    intervals: tuple[int, ...]
    start:     int               # ms
    end:       int | None = None


def load_update_config(file: str | Path= "data_update.yaml") -> dict:

    config_path = CONFIGS_ROOT / 'dataset' / file if isinstance(file, str) else file

    if not config_path.exists():
        raise FileNotFoundError(f"Update config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return {
        "symbols":   raw["symbols"],
        "intervals": [int(i) for i in raw["intervals"]],
        "start":     int(pd.Timestamp(raw["start"], tz="UTC").timestamp() * 1000),
        "end":       int(pd.Timestamp(raw["end"], tz="UTC").timestamp() * 1000)
                     if "end" in raw else None,
    }


# ------------------------------------------------------------------
# Update functions
# ------------------------------------------------------------------


def update_klines(
    fetcher:   BybitFetcher,
    symbol:    str,
    intervals: list[int],
    start:     int,
    end:       int | None = None,
) -> None:
    for interval in intervals:
        for pt in PriceType:
            path = make_data_path(
                symbol     = symbol,
                data_type  = DataType.KLINES,
                interval   = interval,
                price_type = pt,
            )

            fetch_start, fetch_end = _resolve_fetch_range(path, start, end, interval)

            if fetch_start is None:
                logger.info(
                    "%s %s %dm — already up to date, skipping.",
                    symbol, pt.value, interval,
                )
                continue

            logger.info(
                "Updating %s %s %dm from %d to %s.",
                symbol, pt.value, interval, fetch_start,
                fetch_end or "now",
            )

            df = fetcher.fetch_klines(
                symbol      = symbol,
                interval    = interval,
                price_type  = pt,
                start       = fetch_start,
                end         = fetch_end,
            )

            if df.empty:
                logger.warning(
                    "No data returned for %s %s %dm — "
                    "API exhausted or data unavailable.",
                    symbol, pt.value, interval,
                )
                continue

            _validate_fetch_coverage(df, fetch_start, fetch_end, symbol, pt, interval)
            save_partitioned_parquet(df, path)


def update_funding(
    fetcher: BybitFetcher,
    symbol:  str,
    start:   int,
    end:     int | None = None,
) -> None:
    
    funding_path = make_data_path(symbol, data_type=DataType.FUNDING)
    stored       = get_stored_range(funding_path)

    if stored is None:
        # No data — fetch full range from start
        pass

    else:
        stored_start_ms     = int(stored[0].timestamp() * 1000)
        stored_end_ms       = int(stored[1].timestamp() * 1000)
        funding_interval_ms = 8 * 60 * 60 * 1000
        server_time_ms      = fetcher._server_time_ms()

        if start < stored_start_ms:
            # Backfill case — fetch from requested start up to stored start
            end   = stored_start_ms - funding_interval_ms
            logger.info("%s funding — backfilling from %d to %d.", symbol, start, end)

        elif stored_end_ms >= server_time_ms - funding_interval_ms:
            # Forward case — already up to date
            logger.info("%s funding — already up to date, skipping.", symbol)
            return

        else:
            # Forward case — fetch from stored end onwards
            start = stored_end_ms + funding_interval_ms
            logger.info("%s funding — forward filling from %d.", symbol, start)

    logger.info("Updating funding for %s from %d.", symbol, start)

    df = fetcher.fetch_funding_rate(symbol, start=start, end=end)

    if df.empty:
        logger.warning(
            "No funding data returned for %s — "
            "API exhausted or data unavailable.", symbol,
        )
        return

    save_partitioned_parquet(df, funding_path)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------


def _resolve_fetch_range(
    path:     Path,
    start:    int,
    end:      int | None,
    interval: int,
) -> tuple[int | None, int | None]:
    """
    Determine the fetch range for a dataset given its stored range.

    Returns
    -------
    tuple[int | None, int | None]
        (fetch_start, fetch_end) in milliseconds.
        Returns (None, None) if no fetch is needed.
    """
    stored = get_stored_range(path)

    if stored is None:
        # No data exists — fetch full requested range
        return start, end

    stored_start_ms = int(stored[0].timestamp() * 1000)
    stored_end_ms   = int(stored[1].timestamp() * 1000)
    interval_ms     = interval * 60 * 1000

    ### There is techincally a hole, if I request earlier data than is there, it won't
    ### collect data newer than what is stored.
    if start < stored_start_ms:
        # Backfill — fetch from requested start up to stored start
        return start, stored_start_ms - interval_ms

    if end is None or end > stored_end_ms:
        # Forward fill — fetch from stored end onwards
        return stored_end_ms + interval_ms, end

    # Requested range fully covered by stored data
    return None, None

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def _validate_fetch_coverage(
    df:       pd.DataFrame,
    start_ms: int,
    end_ms:   int | None,
    symbol:   str,
    pt:       PriceType,
    interval: int,
) -> None:
    """
    Warn if fetched data doesn't cover the expected range.
    Distinguishes API gaps from genuine data unavailability.
    """
    actual_start = int(df.index[0].timestamp()  * 1000)
    actual_end   = int(df.index[-1].timestamp() * 1000)
    tolerance_ms = interval * 60 * 1000 * 2    # 2 bar tolerance

    if actual_start > start_ms + tolerance_ms:
        logger.warning(
            "%s %s %dm — fetched data starts at %s, "
            "requested from %d. Gap of ~%d bars at start.",
            symbol, pt.value, interval,
            df.index[0], start_ms,
            (actual_start - start_ms) // (interval * 60 * 1000),
        )

    if end_ms is not None and actual_end < end_ms - tolerance_ms:
        logger.warning(
            "%s %s %dm — fetched data ends at %s, "
            "requested to %d. Data may be unavailable beyond this point.",
            symbol, pt.value, interval,
            df.index[-1], end_ms,
        )