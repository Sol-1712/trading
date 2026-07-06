"""
Bybit data fetching layer.

Design principles:
- All fetching is encapsulated in BybitFetcher; no bare session calls elsewhere.
- Pagination is handled by a single generic engine (_fetch_paginated).
- Concurrent fetching (multi-type, multi-symbol) uses ThreadPoolExecutor.
- Retry with exponential backoff is applied at the page level.
- All output DataFrames share a consistent schema: UTC DatetimeIndex,
  snake_case columns, float64 numerics.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable, Literal
import pandas as pd

from trading.data_utils.enums import PriceType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases & schemas
# ---------------------------------------------------------------------------

# Bybit returns kline rows as ordered lists, not dicts, so column order is
# load-bearing. Centralising these schemas here means a Bybit API change
# requires exactly one edit, not a hunt across multiple functions.
_KLINE_SCHEMA: dict[PriceType, dict[str, list[str]]] = {
    PriceType.LAST: {
        "raw_cols": ["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        "numeric_cols": ["open", "high", "low", "close", "volume", "turnover"],
    },
    PriceType.MARK: {
        "raw_cols": ["timestamp", "open", "high", "low", "close"],
        "numeric_cols": ["open", "high", "low", "close"],
    },
    PriceType.INDEX: {
        "raw_cols": ["timestamp", "open", "high", "low", "close"],
        "numeric_cols": ["open", "high", "low", "close"],
    },
}

_FUNDING_COLS: list[str] = ["funding_rate", "timestamp"]


# ---------------------------------------------------------------------------
# Pagination engine
# ---------------------------------------------------------------------------

def _fetch_paginated(
    call_fn: Callable[[int, int], pd.DataFrame],
    time_col: str,
    start: int,
    end: int,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Generic backward-walking pagination engine.

    Repeatedly calls ``call_fn(start, end)`` until the earliest returned
    timestamp reaches ``start``, or no data is returned.


    Parameters
    ----------
    call_fn : Callable[[int, int], pd.DataFrame]
        Function accepting (start_ms, end_ms), returning one page of data.
        Must include ``time_col`` as a column. Must return an empty DataFrame
        (not raise) when no data exists in the given window.
    time_col : str
        Column containing Unix timestamps in milliseconds.
    start : int
        Earliest timestamp to fetch (ms, inclusive).
    end : int
        Latest timestamp to fetch (ms, inclusive).
    max_retries : int
        Retry attempts per page on transient failure.
    retry_delay : float
        Base backoff delay in seconds (doubles each retry).

    Returns
    -------
    pd.DataFrame
        Chronologically sorted, deduplicated result. Empty if nothing returned.
    """
    pages: list[pd.DataFrame] = []
    current_end = end

    while current_end > start:
        page = _call_with_retry(call_fn, start, current_end, max_retries, retry_delay)

        if page is None or page.empty:
            break
        page[time_col] = page[time_col].astype("int64")
        pages.append(page)

        earliest = page[time_col].min()

        # Stop if this page already covers back to (or past) our start.
        if earliest <= start:
            break

        current_end = earliest - 1

    if not pages:
        return pd.DataFrame()

    return (
        pd.concat(pages, ignore_index=True)
        .drop_duplicates(subset=[time_col])
        .sort_values(time_col)
        .reset_index(drop=True)
    )


def _call_with_retry(
    call_fn: Callable[[int, int], pd.DataFrame],
    start: int,
    end: int,
    max_retries: int,
    retry_delay: float,
) -> pd.DataFrame | None:
    """
    Call ``call_fn(start, end)`` with exponential backoff on failure.
    """
    for attempt in range(max_retries):
        try:
            return call_fn(start, end)
        except Exception as exc:
            wait = retry_delay * (2 ** attempt)
            logger.warning(
                "API call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    logger.error("API call abandoned after %d attempts.", max_retries)
    return None


# ---------------------------------------------------------------------------
# BybitFetcher
# ---------------------------------------------------------------------------

class BybitFetcher:
    """
    High-level interface for fetching Bybit market data.

    Provides:
    - Unified kline fetching across price types (last, mark, index)
    - Funding rate fetching
    - Concurrent multi-type and multi-symbol fetching

    All returned DataFrames share a consistent schema:
    - Index: ``pd.DatetimeIndex`` in UTC, named ``"datetime"``
    - Columns: snake_case, float64 for all numeric fields

    Parameters
    ----------
    session :
        Authenticated pybit HTTP session.
    category : str
        Default instrument category (e.g. ``"linear"`` for USDT perps).
    kline_limit : int
        Rows per kline API page. Bybit maximum is 1000.
    funding_limit : int
        Rows per funding rate API page. Bybit maximum is 200.
    max_workers : int
        Thread pool size for concurrent fetching. Keep within Bybit's rate
        limit headroom — 4 is a safe default for authenticated endpoints.
    max_retries : int
        Retry attempts per API page on transient failure.
    retry_delay : float
        Base exponential backoff delay in seconds.
    """

    def __init__(
        self,
        session,
        category: str = "linear",
        kline_limit: int = 1000,
        funding_limit: int = 200,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        
        self._session = session
        self.category = category
        self.kline_limit = kline_limit
        self.funding_limit = funding_limit
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _server_time_ms(self) -> int:
        """Fetch current Bybit server time in milliseconds."""
        return int(self._session.get_server_time()["time"])

    def _resolve_end(self, end: int | None) -> int:
        """Return ``end`` if provided, otherwise current server time."""
        return end if end is not None else self._server_time_ms()

    def _get_kline_api_method(self, price_type: PriceType):
        """Map price_type to the corresponding pybit session method."""
        methods: dict[PriceType, Callable] = {
            PriceType.LAST: self._session.get_kline,
            PriceType.MARK: self._session.get_mark_price_kline,
            PriceType.INDEX: self._session.get_index_price_kline,
        }
        return methods[price_type]

    @staticmethod
    def _to_datetime_index(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Convert a millisecond timestamp column to a UTC DatetimeIndex.

        The timestamp column is dropped after conversion. The index is
        named ``"datetime"`` to make it unambiguous in downstream joins
        (``"timestamp"`` is sometimes also a column name, which creates
        confusing dual-access patterns).
        """
        df = df.copy()
        df.index = pd.to_datetime(df[timestamp_col].astype("int64"), unit="ms", utc=True)
        df.index.name = "datetime"
        return df.drop(columns=[timestamp_col]).sort_index()

    # ------------------------------------------------------------------
    # Kline fetching
    # ------------------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: int,
        price_type: PriceType,
        start: int,
        end: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC(V) kline data for a given symbol and price type.

        Parameters
        ----------
        symbol : str
            Instrument symbol, e.g. ``"BTCUSDT"``.
        interval : int
            Bar interval in minutes.
        price_type : PriceType
            One of ``"last"``, ``"mark"``, ``"index"``.
        start : int
            Start timestamp in milliseconds (inclusive).
        end : int, optional
            End timestamp in milliseconds. Defaults to server time.

        Returns
        -------
        pd.DataFrame
            UTC DatetimeIndex. Columns: ``open, high, low, close``
            (plus ``volume, turnover`` for ``"last"``). All float64.
        """
        schema = _KLINE_SCHEMA[price_type]
        raw_cols: list[str] = schema["raw_cols"]
        numeric_cols: list[str] = schema["numeric_cols"]
        api_method = self._get_kline_api_method(price_type)
        end = self._resolve_end(end)
        symbol = symbol.upper()

        def call_fn(s: int, e: int) -> pd.DataFrame:
            response = api_method(
                category=self.category,
                symbol=symbol,
                interval=interval,
                start=s,
                end=e,
                limit=self.kline_limit,
            )
            data = response["result"]["list"]
            if not data:
                return pd.DataFrame(columns=raw_cols)
            return pd.DataFrame(data, columns=raw_cols)

        logger.info(
            "Fetching %s %s klines at %dm interval. Range: [%d, %d].",
            symbol,
            price_type.value,
            interval,
            start,
            end,
        )

        raw = _fetch_paginated(
            call_fn=call_fn,
            time_col="timestamp",
            start=start,
            end=end,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        if raw.empty:
            logger.warning("No %s kline data returned for %s.", price_type, symbol)
            return raw

        raw[numeric_cols] = raw[numeric_cols].astype("float64")
        return self._to_datetime_index(raw)

    def fetch_all_kline_types(
        self,
        symbol: str,
        interval: int,
        start: int,
        end: int | None = None,
        price_types: list[PriceType] | None = None,
    ) -> dict[PriceType, pd.DataFrame]:
        """
        Fetch multiple kline price types concurrently for a single symbol.

        Threading model
        ---------------
        last/mark/index klines are separate API endpoints with no shared
        state, so they can be fetched in parallel.

        Why as_completed() rather than pool.map()?
        pool.map() raises on the first exception and silently drops results
        from other futures. as_completed() yields each Future as it finishes
        (in completion order, not submission order) and lets us handle each
        result or exception independently — which is why a failed fetch returns
        an empty DataFrame rather than killing the whole call.

        The dict {future: pt} pattern is the standard way to recover which
        argument was associated with each Future after it completes.
        as_completed() yields Future objects; without this dict you cannot tell
        which price_type each result belongs to.

        Parameters
        ----------
        price_types : list[PriceType], optional
            Subset of price types to fetch. Defaults to all three.

        Returns
        -------
        dict[PriceType, pd.DataFrame]
            Maps each requested price type to its DataFrame.
            Failed fetches return an empty DataFrame under their key.
        """
        if price_types is None:
            price_types = [PriceType.LAST, PriceType.MARK, PriceType.INDEX]

        end = self._resolve_end(end)

        def _fetch(pt: PriceType) -> tuple[PriceType, pd.DataFrame]:
            return pt, self.fetch_klines(symbol, interval, pt, start, end)

        results: dict[PriceType, pd.DataFrame] = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(price_types))
        ) as pool:
            # Explicit type annotation so Pylance tracks PriceType through the dict.
            futures: dict[Future[tuple[PriceType, pd.DataFrame]], PriceType] = {
                pool.submit(_fetch, pt): pt for pt in price_types
            }

            for future in as_completed(futures):
                # Retrieve price_type from the tracking dict — do NOT unpack
                # from future.result(), which would rebind pt as str.
                pt: PriceType = futures[future]
                try:
                    _, df = future.result()
                    results[pt] = df
                except Exception as exc:
                    logger.error(
                        "fetch_all_kline_types failed for %s (%s): %s",
                        symbol,
                        pt,
                        exc,
                    )
                    results[pt] = pd.DataFrame()

        return results

    def fetch_klines_multi_symbol(
        self,
        symbols: list[str],
        interval: int,
        price_type: PriceType,
        start: int,
        end: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch klines for multiple symbols concurrently.
        ----------
        symbols : list[str]
        price_type : PriceType
            Price type to fetch for all symbols.

        Returns
        -------
        dict[str, pd.DataFrame]
            Maps each symbol to its DataFrame.
            Failed fetches return an empty DataFrame under their key.
        """
        end = self._resolve_end(end)

        def _fetch(sym: str) -> tuple[str, pd.DataFrame]:
            return sym, self.fetch_klines(sym, interval, price_type, start, end)

        results: dict[str, pd.DataFrame] = {}

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(symbols))
        ) as pool:
            futures: dict[Future[tuple[str, pd.DataFrame]], str] = {
                pool.submit(_fetch, sym): sym for sym in symbols
            }

            for future in as_completed(futures):
                sym: str = futures[future]
                try:
                    _, df = future.result()
                    results[sym] = df
                except Exception as exc:
                    logger.error(
                        "fetch_klines_multi_symbol failed for %s: %s", sym, exc
                    )
                    results[sym] = pd.DataFrame()

        return results

    # ------------------------------------------------------------------
    # Funding rate fetching
    # ------------------------------------------------------------------

    def fetch_funding_rate(
        self,
        symbol: str,
        start: int,
        end: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        Parameters
        ----------
        symbol : str
        start : int
            Start timestamp in milliseconds.
        end : int, optional
            End timestamp in milliseconds. Defaults to server time.

        Returns
        -------
        pd.DataFrame
            UTC DatetimeIndex. Single column: ``"funding_rate"`` (float64).
        """
        end = self._resolve_end(end)
        symbol = symbol.upper()

        def call_fn(s: int, e: int) -> pd.DataFrame:
            response = self._session.get_funding_rate_history(
                category=self.category,
                symbol=symbol,
                startTime=s,
                endTime=e,
                limit=self.funding_limit,
            )
            data = response["result"]["list"]
            if not data:
                return pd.DataFrame(columns=_FUNDING_COLS)
            # Rename Bybit's camelCase fields to snake_case immediately,
            # before any data leaves this closure.
            df = pd.DataFrame(data)
            df = df[["fundingRate", "fundingRateTimestamp"]].rename(columns={
                "fundingRate": "funding_rate",
                "fundingRateTimestamp": "timestamp"
            })
            return df

        logger.info("Fetching funding rate history for %s.", symbol)

        raw = _fetch_paginated(
            call_fn=call_fn,
            time_col="timestamp",
            start=start,
            end=end,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )

        if raw.empty:
            logger.warning("No funding rate data returned for %s.", symbol)
            return raw

        raw["funding_rate"] = raw["funding_rate"].astype("float64")
        return self._to_datetime_index(raw)