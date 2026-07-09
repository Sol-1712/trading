from __future__ import annotations

from dataclasses import dataclass

from trading.data_utils.core import PriceType

BYBIT_VALID_INTERVALS: frozenset[int] = frozenset({1, 3, 5, 15, 30, 60, 120, 240, 360, 720, 1440})


@dataclass(frozen=True)
class KlineSchema:
    raw_cols:     tuple[str, ...]   # column order matches Bybit API response
    numeric_cols: tuple[str, ...]   # subset to cast to float64


@dataclass(frozen=True)
class FundingSchema:
    raw_cols:     tuple[str, ...]   # Bybit camelCase field names
    rename_map:   dict[str, str]    # raw → clean column names
    numeric_cols: tuple[str, ...]   # columns to cast to float64


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