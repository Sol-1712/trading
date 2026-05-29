from pathlib import Path
from pybit.unified_trading import HTTP
import pandas as pd
import key
import logging

from crypto_quant.data_utils.io import save_partitioned_parquet, get_stored_range
from crypto_quant.data_utils.paths import make_data_path
from crypto_quant.data_utils.fetch import BybitFetcher
from crypto_quant.data_utils.enums import PriceType, DataType


logger = logging.getLogger(__name__)
# Can't handle D W M (Also make sure intervals are valid please or it dies)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
INTERVALS = [5, 15, 30, 60]  # minutes
START = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp() * 1000)


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
        starts = []
        ends   = []
        paths: dict[PriceType, Path] = {}

        for pt in PriceType:
            path = make_data_path(symbol, 
                                  DataType.KLINES, 
                                  interval=interval, 
                                  price_type=pt
                                  )
            
            paths[pt] = path

            stored_range = get_stored_range(path)

            if stored_range is None:
                starts.append(start)

            elif start < int(stored_range[0].timestamp() * 1000): # Backfill case
                starts.append(start)
                ends.append(int((stored_range[0] - pd.Timedelta(minutes=interval)).timestamp() * 1000))

            else:
                starts.append(int((stored_range[1] + pd.Timedelta(minutes=interval)).timestamp() * 1000))

        fetch_start = min(starts)
        if ends:
            fetch_end = max(ends)
        else:
            fetch_end = end

        logger.info("Updating kline data for symbol %s, interval %dm from %d", symbol, interval, fetch_start)
        klines = fetcher.fetch_all_kline_types(
            symbol=symbol,
            interval=interval,
            start=fetch_start,
            end=fetch_end
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

    funding_path = make_data_path(symbol, data_type=DataType.FUNDING)
    stored_range = get_stored_range(funding_path)
    
    if stored_range is not None:
        min_ts = int(stored_range[0].timestamp() * 1000)
        if start >= min_ts: # requested start already in stored data
            start = int((stored_range[1] + pd.Timedelta(hours=8)).timestamp() * 1000)
        
    logger.info("Updating funding data for symbol %s from %d", symbol, start)    
    funding = fetcher.fetch_funding_rate(symbol, start = start, end = end)

    if not funding.empty:
        save_partitioned_parquet(funding, funding_path)
    # else:
    #     logger.warning("No new funding data fetched for symbol %s", symbol)


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
            logger.info("Finished updating klines for %s", symbol)
        except Exception as e:
            logger.error("Error updating klines for symbol %s: %s", symbol, e)

        try:
            update_funding(fetcher, symbol, start, end)
            logger.info("Finished updating funding for %s", symbol)
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
     
