import itertools

import numpy as np
import pandas as pd

from phoenix.backtest import run_backtest
from phoenix.checks import lookahead_violations
from phoenix.features import FEATURES, build_features
from phoenix.labels import tp_first_labels
from phoenix.settings import load_settings
from phoenix.walkforward import make_windows, train_rows

S = load_settings()


def _bars(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-06 00:00", periods=n, freq="15min", tz="UTC")
    close = 2600 + np.cumsum(rng.normal(0, 1.5, n))
    open_ = np.concatenate([[2600], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 1.5, n)
    low = np.minimum(open_, close) - rng.uniform(0, 1.5, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": rng.integers(50, 500, n).astype(float),
            "spread_points": 12.0,
        },
        index=idx,
    )


def test_features_have_no_lookahead():
    bars = _bars()
    bad = lookahead_violations(lambda b: build_features(b)[FEATURES], bars, cut_points=[700, 901, 1100, 1350])
    assert bad == []


def test_labels_match_backtest_engine():
    bars = _bars(600)
    feats = build_features(bars)
    sl, tp = 1.0 * feats["atr"], 1.5 * feats["atr"]
    lab = tp_first_labels(bars, sl, tp, 16, S)
    checked = 0
    for t in range(100, 580, 13):
        for side, col in ((1, "long"), (-1, "short")):
            o = pd.DataFrame({"side": 0, "sl_dist": sl, "tp_dist": tp, "max_bars": 16}, index=bars.index)
            o.iloc[t, 0] = side
            tr = run_backtest(bars, o, S, risk_usd=1e6).trades
            if tr.empty or np.isnan(lab[col].iloc[t]):
                continue
            assert (tr.iloc[0].reason == "tp") == bool(lab[col].iloc[t]), (t, side)
            checked += 1
    assert checked > 40


def test_windows_and_purge():
    ws = make_windows(pd.Timestamp("2021-11-15", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC"))
    assert ws[0].val_start == pd.Timestamp("2023-05-15", tz="UTC")
    assert all(a.val_end <= b.val_start or a.val_end == b.val_start for a, b in itertools.pairwise(ws))
    idx = pd.date_range("2021-11-15", "2023-06-01", freq="15min", tz="UTC")
    m = train_rows(idx, ws[0], purge_bars=16)
    assert idx[m].max() < ws[0].train_end - pd.Timedelta(minutes=15 * 15)
