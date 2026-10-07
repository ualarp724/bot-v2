import numpy as np
import pandas as pd
import pytest

from phoenix.intraday.engine import Costs, Rules, Signals, simulate

C0 = Costs(taker=0.0005, maker=0.0002, slippage=0.0002, funding_long_per_hour=0.0)
R0 = Rules(risk_frac=0.01, max_leverage=10.0, max_open=1, max_trades_per_day=None, daily_loss_limit=None)


def _bars(rows, start="2026-01-05 10:00", freq="15min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _sig(n, entries=None, **kw):
    side, dist = np.zeros(n), np.full(n, np.nan)
    for i, (s, d) in (entries or {}).items():
        side[i], dist[i] = s, d
    return Signals(side=side, stop_dist=dist, **kw)


FLAT = [100, 100, 100, 100]


def test_long_stopped_out_matches_hand_calculation():
    # Señal al cierre de la vela 0 -> entrada a la APERTURA de la vela 1 (100) con 0,02 % de deslizamiento.
    # Riesgo 1 % de 10.000 = 100 $ con stop a 1 $ -> 100 BTC. Entrada 100,02; stop 99 tocado (no hay hueco).
    bars = _bars([FLAT, [100, 100.5, 98.5, 99], FLAT])
    r = simulate(bars, _sig(3, {0: (1, 1.0)}), C0, R0)
    t = r.trades.iloc[0]
    assert len(r.trades) == 1 and t.reason == "stop" and t.qty == pytest.approx(100.0)
    assert t.entry == pytest.approx(100.02) and t.exit == pytest.approx(99 * 0.9998)
    assert t.fees == pytest.approx(100 * 100.02 * 0.0005 + 100 * 98.9802 * 0.0005)  # 5,001 + 4,94901
    assert t.slippage == pytest.approx(2.0 + 1.98) and t.gross == pytest.approx(-100.0)
    assert t.net == pytest.approx(-113.93001, abs=1e-4) and t.r == pytest.approx(-1.1393001, abs=1e-5)
    assert t.net == pytest.approx(t.gross - t.fees - t.slippage - t.funding)


def test_signal_executes_at_next_open_never_at_the_signal_close():
    bars = _bars([[100, 101, 99, 100.5], [103, 104, 102.5, 103.5], FLAT])  # la apertura 1 (103) difiere del cierre 0
    r = simulate(bars, _sig(3, {0: (1, 10.0)}), C0, R0)
    assert r.trades.iloc[0].entry == pytest.approx(103 * 1.0002)


def test_take_profit_is_maker_without_slippage_and_needs_price_to_trade_through():
    # Objetivo = entrada + 2 x 1 $ = 102. Maker 0,02 %, sin deslizamiento en la salida.
    bars = _bars([FLAT, [100, 102.5, 99.5, 102], FLAT])
    r = simulate(bars, _sig(3, {0: (1, 1.0)}, tp_r=2.0), C0, R0)
    t = r.trades.iloc[0]
    assert t.reason == "objetivo" and t.exit == pytest.approx(102.0)
    assert t.net == pytest.approx(100 * (102 - 100.02) - 100 * 100.02 * 0.0005 - 100 * 102 * 0.0002)  # 190,959
    assert t.slippage == pytest.approx(2.0)  # solo la entrada
    touch = _bars([FLAT, [100, 102.0, 99.5, 101.9], FLAT])  # high == objetivo: solo lo toca, no se da por ejecutado
    r2 = simulate(touch, _sig(3, {0: (1, 1.0)}, tp_r=2.0), C0, R0)
    assert r2.trades.iloc[0].reason == "fin"


def test_stop_wins_when_stop_and_target_are_both_inside_the_bar():
    bars = _bars([FLAT, [100, 103, 98.5, 100], FLAT])
    r = simulate(bars, _sig(3, {0: (1, 1.0)}, tp_r=2.0), C0, R0)
    assert r.trades.iloc[0].reason == "stop"


def test_gap_through_the_stop_fills_at_the_open():
    # Entra a 100 (vela 1, stop 99) y la vela 2 abre por debajo del stop (97): se ejecuta a la apertura, no en 99.
    bars = _bars([FLAT, [100, 100.3, 99.5, 100], [97.0, 98.0, 96.0, 97.5]])
    r = simulate(bars, _sig(3, {0: (1, 1.0)}), C0, R0)
    t = r.trades.iloc[0]
    assert t.reason == "stop" and t.exit == pytest.approx(97.0 * 0.9998)


def test_short_is_symmetrical():
    bars = _bars([FLAT, [100, 101.5, 99.8, 101], FLAT])
    r = simulate(bars, _sig(3, {0: (-1, 1.0)}), C0, R0)
    t = r.trades.iloc[0]
    assert t.entry == pytest.approx(100 * 0.9998) and t.reason == "stop"
    assert t.exit == pytest.approx(101 * 1.0002) and t.net < 0 and t.gross == pytest.approx(-100.0)


def test_session_flat_closes_before_midnight_and_blocks_late_entries():
    # Velas de 15 min desde las 22:00 UTC hasta las 00:30. Cierre obligatorio a las 23:45.
    bars = _bars([FLAT] * 11, start="2026-01-05 22:00")
    rules = Rules(
        risk_frac=0.01, max_leverage=10.0, max_trades_per_day=None, daily_loss_limit=None,
        flat_at_min=23 * 60 + 45, entry_until_min=22 * 60 + 30,
    )  # fmt: skip
    sig = _sig(
        11, {0: (1, 5.0), 2: (1, 5.0)}
    )  # señal 22:00 -> entra 22:15; señal 22:30 -> entraría 22:45 (fuera de ventana)
    r = simulate(bars, sig, C0, rules)
    assert len(r.trades) == 1 and r.skipped["window"] == 1
    t = r.trades.iloc[0]
    assert t.reason == "sesion" and t.exit_time == pd.Timestamp("2026-01-05 23:45", tz="UTC")


def test_two_losses_trigger_the_daily_halt_and_trading_resumes_next_day():
    # Velas de 1 h. Cada pérdida ~1,1 % del equity; la segunda lleva el día a ~-2,2 % y detiene el día.
    down = [100, 100, 98.5, 99]
    rows = [FLAT, down, FLAT, down, FLAT, FLAT] + [FLAT] * 18 + [FLAT, down, FLAT]
    bars = _bars(rows, start="2026-01-05 00:00", freq="1h")
    n = len(rows)
    entries = {0: (1, 1.0), 2: (1, 1.0), 4: (1, 1.0), 24: (1, 1.0)}
    rules = Rules(risk_frac=0.01, max_leverage=10.0, max_trades_per_day=None, daily_loss_limit=0.02)
    r = simulate(bars, _sig(n, entries), C0, rules)
    assert r.halted_days == 1
    assert len(r.trades) == 3 and r.skipped["halted"] == 1  # la entrada de la vela 4 se bloquea; la del día 2 sí entra
    assert r.trades.iloc[2].entry_time.day == 6


def test_open_position_is_flattened_when_the_daily_limit_is_breached():
    rows = [FLAT, [100, 100, 99.5, 100], [100, 100, 96.5, 97], [97, 97.5, 96.8, 97.2], [97.2, 97.5, 97, 97.3]]
    bars = _bars(rows, freq="1h")
    rules = Rules(risk_frac=0.05, max_leverage=10.0, max_trades_per_day=None, daily_loss_limit=0.02)
    r = simulate(bars, _sig(5, {0: (1, 5.0)}), C0, rules)  # 1x de apalancamiento, stop a 95 (no se toca)
    t = r.trades.iloc[0]
    assert t.reason == "limite_diario" and t.exit_time == bars.index[3]
    assert t.exit == pytest.approx(97 * 0.9998)  # a la apertura de la vela siguiente a la que rompió el límite


def test_max_open_max_trades_per_day_and_no_hedging():
    bars = _bars([FLAT] * 8)
    two = Rules(risk_frac=0.01, max_leverage=10.0, max_open=2, max_trades_per_day=None, daily_loss_limit=None)
    r = simulate(bars, _sig(8, {0: (1, 5.0), 1: (1, 5.0), 2: (1, 5.0), 3: (-1, 5.0)}), C0, two)
    assert len(r.trades) == 2 and r.skipped["no_slot"] == 2  # el tercero no cabe y el corto no se abre en contra
    capped = Rules(risk_frac=0.01, max_leverage=10.0, max_open=3, max_trades_per_day=1, daily_loss_limit=None)
    r2 = simulate(bars, _sig(8, {0: (1, 5.0), 2: (1, 5.0)}), C0, capped)
    assert len(r2.trades) == 1 and r2.skipped["day_limit"] == 1


def test_funding_is_paid_by_longs_only():
    costs = Costs(taker=0.0005, maker=0.0002, slippage=0.0, funding_long_per_hour=0.001)
    bars = _bars([FLAT] * 4)
    lg = simulate(bars, _sig(4, {0: (1, 5.0)}), costs, R0).trades.iloc[0]
    sh = simulate(bars, _sig(4, {0: (-1, 5.0)}), costs, R0).trades.iloc[0]
    assert lg.funding == pytest.approx(lg.qty * 100 * 0.001 * 0.25 * 3)  # 3 velas de 15 min con la posición abierta
    assert sh.funding == 0.0


def test_leverage_cap_limits_the_size():
    bars = _bars([FLAT] * 3)
    rules = Rules(risk_frac=0.01, max_leverage=5.0, max_trades_per_day=None, daily_loss_limit=None)
    t = simulate(bars, _sig(3, {0: (1, 0.01)}), C0, rules).trades.iloc[0]  # sin tope serían 10.000 BTC
    assert t.notional <= 5.0 * 10_000 + 1e-6 and t.qty == pytest.approx(500.0)


def test_signal_before_the_window_is_not_executed_and_open_trades_close_at_the_end():
    bars = _bars([FLAT] * 6)
    r = simulate(bars, _sig(6, {1: (1, 5.0), 2: (1, 5.0)}), C0, R0, i0=2, i1=5)
    assert (
        len(r.trades) == 1 and r.trades.iloc[0].entry_time == bars.index[3]
    )  # la señal de la vela 1 (fuera) se ignora
    assert r.trades.iloc[0].reason == "fin" and r.trades.iloc[0].exit_time == bars.index[4]
    assert len(r.equity) == 3


def test_exit_signal_time_stop_and_cooldown():
    bars = _bars([FLAT] * 8)
    xl = np.zeros(8, bool)
    xl[2] = True
    r = simulate(bars, _sig(8, {0: (1, 5.0)}, exit_long=xl), C0, R0)
    t = r.trades.iloc[0]
    assert t.reason == "senal" and t.exit_time == bars.index[3]
    held = Rules(risk_frac=0.01, max_leverage=10.0, max_trades_per_day=None, daily_loss_limit=None, max_hold_bars=2)
    r2 = simulate(bars, _sig(8, {0: (1, 5.0)}), C0, held)
    assert r2.trades.iloc[0].reason == "tiempo" and r2.trades.iloc[0].bars == 2
    lose = [FLAT, [100, 100, 98.5, 99], FLAT, FLAT, FLAT, FLAT]
    cool = Rules(risk_frac=0.01, max_leverage=10.0, max_trades_per_day=None, daily_loss_limit=None, cooldown_bars=3)
    r3 = simulate(_bars(lose), _sig(6, {0: (1, 1.0), 1: (1, 1.0)}), C0, cool)
    assert len(r3.trades) == 1 and r3.skipped["cooldown"] == 1


def test_accounting_identity_on_random_walks():
    """Equity final = capital + suma de las operaciones netas, y neto = bruto - comisiones - deslizamiento - funding."""
    rng = np.random.default_rng(7)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 1500
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.0005, n))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
        bars = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close},
            index=pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC"),
        )
        side = rng.choice([-1, 0, 0, 1], n)
        sig = Signals(
            side=side,
            stop_dist=close * 0.006,
            exit_long=rng.random(n) < 0.05,
            exit_short=rng.random(n) < 0.05,
            tp_r=1.5,
        )
        costs = Costs(funding_long_per_hour=0.0000125)
        rules = Rules(
            risk_frac=0.01,
            max_open=2,
            max_trades_per_day=8,
            daily_loss_limit=0.02,
            flat_at_min=23 * 60 + 45,
            max_hold_bars=40,
        )
        r = simulate(bars, sig, costs, rules)
        t = r.trades
        assert len(t) > 10
        assert np.allclose(t.net, t.gross - t.fees - t.slippage - t.funding, atol=1e-6)
        assert r.equity.iloc[-1] == pytest.approx(10_000 + t.net.sum(), abs=1e-6)
        assert (t.notional <= rules.max_leverage * 20_000).all() and r.equity.notna().all()


# ------------------------------------------------------------------ neutralidad del motor
def _gross_bps(seeds, minutes, leak=False, dof=None):
    from phoenix.intraday.data import resample, synthetic_1m
    from phoenix.intraday.features import atr

    zero = Costs(taker=0.0, maker=0.0, slippage=0.0, funding_long_per_hour=0.0)
    rules = Rules(risk_frac=0.005, max_trades_per_day=None, daily_loss_limit=None, max_hold_bars=12)
    out = []
    for seed in seeds:
        bars = resample(synthetic_1m(days=40, seed=1000 + seed, start="2024-01-01", dof=dof), minutes)
        rng = np.random.default_rng(5000 + seed)
        if leak:  # CONTROL POSITIVO: la señal conoce la dirección de la vela siguiente (no causal a propósito)
            fwd = np.sign(bars["close"].shift(-1).to_numpy() - bars["open"].shift(-1).to_numpy())
            side = np.where(rng.random(len(bars)) < 0.04, np.nan_to_num(fwd), 0).astype(int)
        else:
            side = rng.choice([-1, 0, 1], len(bars), p=[0.04, 0.92, 0.04])
        sig = Signals(side=side, stop_dist=2.0 * atr(bars, 14).to_numpy())
        t = simulate(bars, sig, zero, rules).trades
        out.append((t["gross"] / t["notional"] * 1e4).to_numpy())
    g = np.concatenate(out)
    return g.mean(), g.std(ddof=1) / np.sqrt(len(g)), len(g)


def test_engine_is_neutral_on_a_gaussian_random_walk_and_the_test_has_power():
    """Sin ventaja y sin costes, cualquier estrategia debe ganar 0 en bruto: si no, el motor regala (o quita) dinero.
    Con retornos gaussianos y extremos de puente browniano el sesgo medido es +0,3 pb (z = 0,4 con 80 semillas)."""
    mean, se, n = _gross_bps(range(40), 15)
    assert n > 5000 and abs(mean / se) < 3.0, f"sesgo bruto {mean:+.2f} pb (z = {mean / se:+.1f})"
    # Control positivo: con una señal que mira el futuro, el mismo test tiene que dispararse.
    lmean, lse, _ = _gross_bps(range(6), 15, leak=True)
    assert lmean / lse > 5.0


def test_load_1m_wires_the_binance_loader(monkeypatch):
    from phoenix.btc5m import binance
    from phoenix.intraday.data import load_1m

    calls = []
    monkeypatch.setattr(binance, "download", lambda start: calls.append(("download", start)))
    monkeypatch.setattr(binance, "load", lambda start: calls.append(("load", start)) or "df")
    assert load_1m("2024-01", download=True) == "df" and calls == [("download", "2024-01"), ("load", "2024-01")]
    calls.clear()
    assert load_1m("2024-02") == "df" and calls == [("load", "2024-02")]
