from __future__ import annotations

from dataclasses import dataclass

from trading.data_utils.core import PriceType

BYBIT_VALID_INTERVALS: frozenset[int] = frozenset({1, 3, 5, 15, 30, 60, 120, 240, 360, 720, 1440})


@dataclass(frozen=True)
class KlineSchema:
    """
    Column layout for a Bybit kline API response.

    Parameters
    ----------
    raw_cols : tuple[str, ...]
        Column names in the order returned by the Bybit API.
    numeric_cols : tuple[str, ...]
        Subset of ``raw_cols`` to cast to ``float64`` after fetch.
    """
    raw_cols:     tuple[str, ...]
    numeric_cols: tuple[str, ...]


@dataclass(frozen=True)
class FundingSchema:
    """
    Column layout for a Bybit funding-rate API response.

    Parameters
    ----------
    raw_cols : tuple[str, ...]
        Bybit camelCase field names to select from the response.
    rename_map : dict[str, str]
        Mapping from raw camelCase names to snake_case column names.
    numeric_cols : tuple[str, ...]
        Columns to cast to ``float64`` after rename.
    """
    raw_cols:     tuple[str, ...]
    rename_map:   dict[str, str]
    numeric_cols: tuple[str, ...]


KLINE_SCHEMAS: dict[PriceType, KlineSchema] = {
    PriceType.LAST: KlineSchema(
        raw_cols     = ("timestamp", "open", "high", "low", "close", "volume", "turnover"),
        numeric_cols = ("open", "high", "low", "close", "volume", "turnover"),
    ),
    PriceType.MARK: KlineSchema(
        raw_cols     = ("timestamp", "open", "high", "low", "close"),
        numeric_cols = ("open", "high", "low", "close"),
    ),
    PriceType.INDEX: KlineSchema(
        raw_cols     = ("timestamp", "open", "high", "low", "close"),
        numeric_cols = ("open", "high", "low", "close"),
    ),
}

FUNDING_SCHEMA = FundingSchema(
    raw_cols     = ("fundingRate", "fundingRateTimestamp"),
    rename_map   = {"fundingRate": "funding_rate", "fundingRateTimestamp": "timestamp"},
    numeric_cols = ("funding_rate",),
)