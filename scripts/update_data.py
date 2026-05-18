from pathlib import Path
from pybit.unified_trading import HTTP
import pandas as pd
import key
import logging

from data_utils_n.io import save_partitioned_parquet, get_stored_range
from data_utils_n.paths import make_data_path
from data_utils_n.fetch import BybitFetcher, KLINE_PRICE_TYPES, PriceType
logger = logging.getLogger(__name__)


SYMBOLS = ["BTCUSDT"]
INTERVALS = [1]  # minutes
START = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)


def update_klines(
        fetcher: BybitFetcher,
        symbol: str,
        intervals: list[int],
        start: int,
        end: int | None = None
        ) -> None:
    """
    Updates kline data for a given symbol.

    Parameters
    ----------
    fetcher : BybitFetcher
        Instance of BybitFetcher to use for API calls.
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    intervals : list[int]
        List of bar intervals in minutes to update, e.g. [1, 5, 15, 60, 240].
    start : int
        Start timestamp in milliseconds since epoch.
    end : int, optional
        End timestamp in milliseconds since epoch. If None, fetches up to current time.
    """
    for interval in intervals:   
        ts = []
        paths: dict[PriceType, Path] = {}

        for price_type in KLINE_PRICE_TYPES:
            path = make_data_path(symbol, "klines", interval=interval, price_type=price_type)
            paths[price_type] = path

            last_ts = get_latest_timestamp(path)
            if last_ts is not None:
                ts.append(int((last_ts + pd.Timedelta(minutes=interval)).timestamp() * 1000))
            else:
                ts.append(start)

        fetch_start = min(ts)
        klines = fetcher.fetch_all_kline_types(
            symbol=symbol,
            interval=interval,
            start=fetch_start,
            end=end
        )

        for price_type, df in klines.items():
            if df.empty:
                logger.warning("No new kline data fetched for symbol %s, interval %dm, price type %s", symbol, interval, price_type)
                continue
            save_partitioned_parquet(df, paths[price_type])


def update_funding(fetcher: BybitFetcher, 
                   symbol: str, 
                   start: int,
                   end: int | None = None
                   ) -> None:
    """
    Updates funding rate data for a given symbol.

    Parameters
    ----------
    fetcher : BybitFetcher
        Instance of BybitFetcher to use for API calls.
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    start : int
        Start timestamp in milliseconds since epoch.
    end : int, optional
        End timestamp in milliseconds since epoch. If None, fetches up to current time.
    """

    funding_path = make_data_path(symbol, "funding")
    last_ts = get_latest_timestamp(funding_path)

    if last_ts is not None:
        start = int((last_ts + pd.Timedelta(hours=8)).timestamp() * 1000)

    funding = fetcher.fetch_funding_rate(symbol, start = start, end = end)
    if not funding.empty:
        save_partitioned_parquet(funding, funding_path)
    else:
        logger.warning("No new funding data fetched for symbol %s", symbol)


def run(fetcher: BybitFetcher, 
        symbols: list[str], 
        intervals: list[int], 
        start: int, 
        end: int | None = None
        ) -> None:
    
    for symbol in symbols:
        logger.info("Updating data for symbol: %s", symbol)
        try:
            update_klines(fetcher, symbol, intervals, start, end)
        except Exception as e:
            logger.error("Error updating klines for symbol %s: %s", symbol, e)

        try:
            update_funding(fetcher, symbol, start, end)
        except Exception as e:
            logger.error("Error updating funding for symbol %s: %s", symbol, e)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    session = HTTP(testnet=False, api_key=key.BYBIT_API_KEY, api_secret=key.BYBIT_API_SECRET)
    fetcher = BybitFetcher(session)
    run(fetcher, SYMBOLS, INTERVALS, START)
