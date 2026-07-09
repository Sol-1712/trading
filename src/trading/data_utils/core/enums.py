from enum import Enum


class PriceType(str, Enum):
    """
    Valid price fields for signal generation and execution simulation.
    
    Inheriting from str means PriceType.LAST == "last" is True,
    so existing string comparisons and YAML values work without 
    explicit coercion at every callsite.
    """
    LAST  = "last"
    MARK  = "mark"
    INDEX = "index"


class DataType(str, Enum):
    """
    Valid data types for path construction and loading.
    """
    KLINES = "klines"
    FUNDING = "funding"