import os
API_KEY = os.getenv('BYBIT_API_KEY')
API_SECRET = os.getenv('BYBIT_API_SECRET')

import argparse
from pybit.unified_trading import HTTP
import pandas as pd
import logging

from trading.data_utils.core import BYBIT_VALID_INTERVALS
from trading.data_utils.fetcher import BybitFetcher
from trading.data_utils.dataset import update_klines, update_funding, load_update_config


logger = logging.getLogger(__name__)


def run(
    fetcher: BybitFetcher, 
    symbols: list[str], 
    intervals: list[int], 
    start: int, 
    end: int | None = None
) -> None:
    
    _validate_intervals(intervals)

    for symbol in symbols:
        logger.info("Updating data for symbol: %s", symbol)

        try:
            update_klines(fetcher, symbol, intervals, start, end)
            logger.info("Finished updating klines for %s", symbol)
        except Exception as e:
            logger.error("Error updating klines for symbol %s: %s", symbol, e)

        try:
            update_funding(fetcher, symbol, start, end)
            logger.info("Finished updating funding for %s", symbol)
        except Exception as e:
            logger.error("Error updating funding for symbol %s: %s", symbol, e)


def _validate_intervals(intervals: list[int]) -> None:
    invalid = set(intervals) - BYBIT_VALID_INTERVALS
    if invalid:
        raise ValueError(
            f"Invalid intervals: {invalid}. "
            f"Valid Bybit intervals (minutes): {sorted(BYBIT_VALID_INTERVALS)}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Update market data.")
    parser.add_argument(
        "--config",
        default = "config/data_update.yaml",
        help    = "Path to update config YAML.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    cfg     = load_update_config(args.config)
    session = HTTP(testnet=False, api_key=API_KEY, api_secret=API_SECRET)
    fetcher = BybitFetcher(session)

    run(fetcher, **cfg)

# python -m scripts.update                                # default
# python -m scripts.update --config config/eth_only.yaml  # targeted