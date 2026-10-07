import numpy as np
import pandas as pd
import pytest

from phoenix.backtest import run_backtest
from phoenix.metrics import max_drawdown, monte_carlo_dd, summarize
from phoenix.settings import load_settings

S = load_settings()
RISK = 5.0  # USD; con SL de 5 $/oz -> 0,01 lotes (1 oz)


def _bars(rows, start="2025-01-06 10:00", spread_points=12):
    idx = pd.date_range(start, periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    df["tick_volume"] = 1.0
    df["spread_points"] = float(spread_points)  # 0,12 $/oz = el suelo
    return df


def _orders(bars, signals):
    o = pd.DataFrame({"side": 0, "sl_dist": np.nan, "tp_dist": np.nan, "max_bars": 0}, index=bars.index)
    for i, (side, sl, tp, mb) in signals.items():
        o.iloc[i] = [side, sl, tp, mb]
    return o


def test_long_hits_tp_with_exact_costs():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 111, 99, 110], [110, 110, 110, 110]])
    res = run_backtest(bars, _orders(bars, {0: (1, 5, 10, 10)}), S, risk_usd=RISK)
    t = res.trades.iloc[0]
    # entrada en la apertura de la vela 1: 100 + spread 0,12 + desliz 0,05
    assert t.entry == pytest.approx(100.17)
    assert t.reason == "tp" and t.exit == pytest.approx(110.17)
    # 10 $/oz * 1 oz - comisión 0,06
    assert t.pnl_usd == pytest.approx(10 - 0.06)
    assert t.r == pytest.approx((10 - 0.06) / 5)


def test_same_bar_sl_and_tp_assumes_sl():
    bars = _bars([[100, 100, 100, 100], [100, 120, 80, 100], [100, 100, 100, 100]])
    t = run_backtest(bars, _orders(bars, {0: (1, 5, 10, 10)}), S, risk_usd=RISK).trades.iloc[0]
    assert t.reason == "sl"
    assert t.exit == pytest.approx(100.17 - 5 - 0.05)


def test_short_uses_ask_for_exits():
    bars = _bars([[100, 100, 100, 100], [100, 100.5, 99.5, 100], [100, 100.2, 89.0, 90], [90, 90, 90, 90]])
    t = run_backtest(bars, _orders(bars, {0: (-1, 5, 10, 10)}), S, risk_usd=RISK).trades.iloc[0]
    assert t.entry == pytest.approx(99.95)  # vende al bid menos desliz
    assert t.reason == "tp" and t.exit == pytest.approx(89.95)  # ask (bid 89 + 0,12) <= 89,95
    assert t.pnl_usd == pytest.approx(10 - 0.06)


def test_gap_through_stop_exits_at_open():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [90, 91, 89, 90], [90, 90, 90, 90]])
    t = run_backtest(bars, _orders(bars, {0: (1, 5, 10, 10)}), S, risk_usd=RISK).trades.iloc[0]
    assert t.reason == "sl_gap" and t.exit == pytest.approx(90 - 0.05)
    assert t.r < -1  # el hueco hace perder más de 1 R


def test_time_exit_and_one_position_at_a_time():
    bars = _bars([[100, 100, 100, 100]] + [[100, 101, 99, 100.5]] * 5)
    orders = _orders(bars, {0: (1, 5, 10, 3), 1: (1, 5, 10, 3), 2: (-1, 5, 10, 3)})
    tr = run_backtest(bars, orders, S, risk_usd=RISK).trades
    assert len(tr) == 1  # las señales 1 y 2 llegan con la posición abierta
    assert tr.iloc[0].reason == "time" and tr.iloc[0].bars == 3
    assert tr.iloc[0].exit == pytest.approx(100.5 - 0.05)


def test_closes_before_market_break_and_never_enters_across_it():
    idx = pd.DatetimeIndex(["2025-01-06 21:30", "2025-01-06 21:45", "2025-01-06 23:00", "2025-01-06 23:15"], tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "tick_volume": 1.0, "spread_points": 12.0},
        index=idx,
    )
    orders = _orders(bars, {0: (1, 5, 10, 10), 1: (1, 5, 10, 10)})
    tr = run_backtest(bars, orders, S, risk_usd=RISK).trades
    assert len(tr) == 1
    assert tr.iloc[0].reason == "break" and tr.iloc[0].exit_time == idx[1]


def test_signal_too_wide_for_min_lot_is_skipped():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 101, 99, 100]])
    res = run_backtest(bars, _orders(bars, {0: (1, 6, 10, 5)}), S, risk_usd=RISK)
    assert res.trades.empty and res.skipped_min_lot == 1


def test_future_bars_do_not_change_past_trades():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    rows = [[c, c + 1, c - 1, c] for c in close]
    bars = _bars(rows)
    sig = {i: (1 if i % 2 else -1, 2, 3, 8) for i in range(0, 300, 7)}
    full = run_backtest(bars, _orders(bars, sig), S, risk_usd=RISK).trades
    half = run_backtest(
        bars.iloc[:150], _orders(bars.iloc[:150], {k: v for k, v in sig.items() if k < 150}), S, risk_usd=RISK
    ).trades
    done = half[half.exit_time < bars.index[149]]
    pd.testing.assert_frame_equal(full.iloc[: len(done)].reset_index(drop=True), done.reset_index(drop=True))


def test_metrics():
    assert max_drawdown(np.array([10, -5, -10, 20]), 100) == (15.0, pytest.approx(100 * 15 / 110))
    assert monte_carlo_dd(np.array([-1.0] * 10), 100) == pytest.approx(10.0)
    bars = _bars([[100, 100, 100, 100], [100, 111, 99, 110], [110, 110, 110, 110]])
    s = summarize(run_backtest(bars, _orders(bars, {0: (1, 5, 10, 10)}), S, risk_usd=RISK).trades, 228)
    assert s["trades"] == 1 and s["win_rate_pct"] == 100
