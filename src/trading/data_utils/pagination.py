import pandas as pd
from typing import Callable
import logging
import time

logger = logging.getLogger(__name__)

def fetch_paginated(
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
    call_fn:     Callable[[int, int], pd.DataFrame],
    start:       int,
    end:         int,
    max_retries: int,
    retry_delay: float,
) -> pd.DataFrame | None:
    """
    Call call_fn(start, end) with exponential backoff on failure.

    Distinguishes rate limit errors (429) from transient failures —
    rate limits get a longer minimum wait.

    Returns
    -------
    pd.DataFrame | None
        None means retries exhausted — caller treats as API failure.
        Empty DataFrame means API returned successfully with no data.
    """
    for attempt in range(max_retries):
        try:
            return call_fn(start, end)

        except Exception as exc:
            exc_str = str(exc).lower()
            is_rate_limit = (
                "429"         in exc_str or
                "10006"       in exc_str or
                "rate limit"  in exc_str
            )

            if is_rate_limit:
                wait = max(retry_delay * (2 ** attempt), 10.0)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Waiting %.1fs.",
                    attempt + 1, max_retries, wait,
                )
            else:
                wait = retry_delay * (2 ** attempt)
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt + 1, max_retries, exc, wait,
                )

            time.sleep(wait)

    logger.error("API call abandoned after %d attempts.", max_retries)
    return None