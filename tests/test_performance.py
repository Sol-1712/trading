import pandas as pd

from trading.backtester.portfolio import TradeLog
from trading.backtester.performance import PerformanceReport, _DISPLAY_LABELS

def test_all_metrics_have_display_labels():
    report = PerformanceReport(pd.DataFrame(), TradeLog())
    missing = set(report.summary()) - set(_DISPLAY_LABELS)
    assert not missing, f"Missing display labels for: {missing}"