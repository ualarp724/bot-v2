import numpy as np
import pandas as pd

from phoenix.checks import lookahead_violations
from phoenix.timeframes import higher_tf_closed


def _random_bars(n=400, start="2025-01-06 00:00"):
    rng = np.random.default_rng(1)
    idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
    close = 2600 + np.cumsum(rng.normal(0, 1, n))
    open_ = np.concatenate([[2600], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "tick_volume": 1.0, "spread_points": 7.0}, index=idx
    )


def test_h1_is_assigned_only_when_closed():
    bars = _random_bars()
    h1 = higher_tf_closed(bars, "1h")
    # La vela H1 de las 10:00 (base 10:00-10:45) aparece en la fila de las 10:45, no antes
    t0 = pd.Timestamp("2025-01-06 10:00", tz="UTC")
    expected_close = bars.loc[t0 + pd.Timedelta("45min"), "close"]
    assert h1.loc[t0 + pd.Timedelta("45min"), "close"] == expected_close
    assert h1.loc[t0 + pd.Timedelta("30min"), "close"] != expected_close
    assert h1.loc[t0 + pd.Timedelta("30min"), "close"] == bars.loc[t0 - pd.Timedelta("15min"), "close"]


def test_incomplete_last_candle_is_not_used():
    bars = _random_bars()
    cut = bars.loc[:"2025-01-06 10:30"]  # la H1 de las 10:00 está a medias
    h1 = higher_tf_closed(cut, "1h")
    assert h1["close"].iloc[-1] == cut.loc[pd.Timestamp("2025-01-06 09:45", tz="UTC"), "close"]


def test_checker_passes_correct_mtf():
    bars = _random_bars()

    def fn(b):
        return higher_tf_closed(b, "1h")["close"]

    assert lookahead_violations(fn, bars, cut_points=[50, 101, 202, 303]) == []


def test_checker_catches_legacy_mtf_leak():
    bars = _random_bars()

    def legacy(b):  # lo que hacía legacy/core/mtf.py
        return b["close"].resample("1h").last().reindex(b.index, method="ffill")

    assert len(lookahead_violations(legacy, bars, cut_points=[50, 101, 202, 303])) > 0
