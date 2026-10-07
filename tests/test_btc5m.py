import numpy as np
import pandas as pd
import pytest

from phoenix.btc5m.backtest import fee_per_share, fill_prices, simulate, summary
from phoenix.btc5m.features import features_at, proxy_target, window_starts


def _minutes(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq="1min", tz="UTC")
    close = 80000 + np.cumsum(rng.normal(0, 20, n))
    open_ = np.concatenate([[80000], close[:-1]])
    high = np.maximum(open_, close) + rng.uniform(0, 10, n)
    low = np.minimum(open_, close) - rng.uniform(0, 10, n)
    vol = rng.uniform(5, 50, n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "taker_buy_base": vol * rng.uniform(0.3, 0.7, n),
        },
        index=idx,
    )


def test_fee_formula():
    assert fee_per_share(0.5) == pytest.approx(0.0175)
    assert fee_per_share(0.9) == pytest.approx(0.0063)


def test_features_do_not_look_ahead():
    m = _minutes()
    starts = window_starts(m)
    X = features_at(m, starts, lag=1)
    rng = np.random.default_rng(1)
    for cut in (starts[300], starts[350]):
        fut = m.index >= cut  # la vela que empieza en el inicio y todas las posteriores
        alt = m.copy()
        alt.loc[fut, ["open", "high", "low", "close"]] *= rng.normal(1, 0.01, (fut.sum(), 1))
        alt.loc[fut, "volume"] *= 3
        Xa = features_at(alt, starts, lag=1)
        before = X.index <= cut
        pd.testing.assert_frame_equal(X[before], Xa[before])


def test_proxy_target_uses_minute_before_and_last_minute():
    m = _minutes(30)
    s = pd.DatetimeIndex([m.index[10]])
    typ = (m.high + m.low + m.close) / 3
    y = proxy_target(m, s)
    assert y.iloc[0] == float(typ.iloc[14] >= typ.iloc[9])


def test_fill_prices_and_simulation():
    markets = pd.DataFrame({"slug": ["a", "b"], "start_ts": [1000, 1300], "outcome_up": [True, False]})
    trades = pd.DataFrame(
        {
            "slug": ["a", "a", "a", "b", "b"],
            "ts": [1001, 1003, 990, 1302, 1302],
            "side": ["BUY", "BUY", "BUY", "BUY", "SELL"],
            "outcome": ["Up", "Up", "Up", "Down", "Down"],
            "price": [0.50, 0.54, 0.40, 0.45, 0.10],
            "size": [10, 10, 100, 20, 50],
        }
    )
    px = fill_prices(trades, markets, 1, 5)
    assert px.loc["a", "px_up"] == pytest.approx(0.52)  # la operación de t=990 queda fuera
    assert px.loc["b", "px_down"] == pytest.approx(0.45)  # las ventas no cuentan
    p_up = pd.Series({"a": 0.60, "b": 0.30})
    bets = simulate(markets, p_up, px, margin=0.02, stake=5)
    a = bets.set_index("slug").loc["a"]
    assert a.side == "Up" and a.won
    shares = 5 / 0.52
    assert a.pnl == pytest.approx(shares - 5 - shares * fee_per_share(0.52))
    b = bets.set_index("slug").loc["b"]
    assert b.side == "Down" and b.won  # p_down 0,70 - 0,45 - comisión > margen
    assert summary(bets)["bets"] == 2


def test_paper_decide_matches_backtest_rule():
    from phoenix.btc5m.paper import decide

    side, price, edge = decide(0.60, 0.52, 0.49, margin=0.02)
    assert side == "Up" and price == 0.52
    assert edge == pytest.approx(0.60 - 0.52 - fee_per_share(0.52))
    side, _, _ = decide(0.53, 0.52, 0.49, margin=0.02)
    assert side is None
    side, price, _ = decide(0.30, 0.72, 0.45, margin=0.02)
    assert side == "Down" and price == 0.45
