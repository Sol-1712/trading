from datetime import datetime, timezone
from pathlib import Path
from pybit.unified_trading import HTTP
import pandas as pd
import key

from data_utils.fetch import fetch_funding_rate, fetch_last_ohlcv, fetch_mark_ohlc
from data_utils.io import save_partitioned_parquet, load_partitioned_parquet
from data_utils.paths import make_data_path

### Requires a full rewrite with threading logging, more rigorous
### Call multiple symbols at once rather than pagination
### Also it gets all the data and then saves, better to get all of one file (say a month), save and continue

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = [1, 5, 15, 60, 240]
START_FALLBACK = 1609459200000 # 01/01/2021 00:00:00



def get_last_timestamp(
        path: Path
        ) -> int | None:
    
    """
    Returns last timestamp (unix ms) in existing parquet files,
    or None if no data exists.
    """
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return None

    df = load_partitioned_parquet(path)

    df.index = df.index.view("int64") // 1_000_000

    return int(df.index.max())


def update_ohlcv(
        session,
        symbol: str,
        interval: int,
        start_fallback: int = START_FALLBACK
    ) -> None:

    """
    Updates last and mark price ohlcv data

    Parameters
    ----------
    session : HTTP
        Bybit API session
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    interval : int
        Bar interval in minutes 
    start_fallback: int (optional)
        Start time fallback if data cannot be found
    """

    path = make_data_path('ohlcv', symbol, interval)

    last_ts = get_last_timestamp(path)
    start = last_ts + 1 if last_ts is not None else start_fallback # No files found

    print(f"Updating Last_OHLCV {symbol} {interval}m from {start}")
    df_last = fetch_last_ohlcv(
        session=session,
        symbol=symbol,
        interval=interval,
        start=start,
    )

    if df_last.empty:
        print("No new OHLCV data.")
        return    

    print(f"Updating Mark_OHLC {symbol} {interval}m from {start}")
    df_mark = fetch_mark_ohlc(
        session=session,
        symbol=symbol,
        interval=interval,
        start=start,
    )

    df_merge = df_last.merge(df_mark, how='left', on='timestamp')
    
    save_partitioned_parquet(df_merge, path)
    

def update_funding(
    session,
    symbol: str,
    start_fallback: int = START_FALLBACK,
    ):
    
    """
    Updates funding rate data

    Parameters
    ----------
    session : HTTP
        Bybit API session
    symbol : str
        Instrument symbol, e.g. "BTCUSDT".
    start_fallback: int (optional)
        Start time fallback if data cannot be found
    """

    path = make_data_path("funding", symbol)

    last_ts = get_last_timestamp(path)
    start = last_ts + 1 if last_ts is not None else start_fallback

    print(f"Updating funding {symbol} from {start}")

    df = fetch_funding_rate(
        session=session,
        symbol=symbol,
        start=start,
    )

    if df.empty:
        print("No new funding data.")
        return

    save_partitioned_parquet(
        df=df,
        dataset_path=path
    )



def main():
    session = HTTP(api_key=key.BYBIT_API_KEY, api_secret=key.BYBIT_API_SECRET)

    for symbol in SYMBOLS:
        for interval in INTERVALS:
            update_ohlcv(session, symbol, interval)

        update_funding(session, symbol)


if __name__ == "__main__":

    main()