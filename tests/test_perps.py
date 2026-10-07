import json

import pandas as pd
import pytest

from phoenix.perps.sim import Costs, Plan, liquidation_price, run

C = Costs(taker=0.0005, slippage=0.0, funding_long_per_hour=0.0)


def _bars(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _once(plan):
    done = {"x": False}

    def strat(st):
        if not done["x"]:
            done["x"] = True
            return plan
        return None
    return strat


def test_liquidation_price_10x():
    assert liquidation_price(100.0, 1, 10, C) == pytest.approx(95.0)
    assert liquidation_price(100.0, -1, 10, C) == pytest.approx(105.0)


def test_long_reaches_target_and_stops():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 150, 99, 140], [140, 140, 140, 140]])
    r = run(bars, _once(Plan(1, 97.0, 10.0)), bars.index[0], days=1, capital=100, target=150)
    assert r.hit_target and r.equity_end == pytest.approx(150, abs=0.01)


def test_stop_loss_costs():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [100, 100, 96, 97], [97, 97, 97, 97]])
    r = run(bars, _once(Plan(1, 97.0, 10.0)), bars.index[0], days=1, capital=100, costs=C)
    # 10 BTC-equivalente de 100 $ -> 10x: pierde 3 % * 1000 = 30 $ + comisiones (0,5 + 0,485)
    assert r.equity_end == pytest.approx(100 - 30 - 0.5 - 0.485, abs=0.01)
    assert not r.hit_target and r.trades == 1


def test_gap_through_liquidation():
    bars = _bars([[100, 100, 100, 100], [100, 101, 99, 100], [90, 91, 89, 90], [90, 90, 90, 90]])
    r = run(bars, _once(Plan(1, 94.0, 10.0)), bars.index[0], days=1, capital=100, costs=C)
    assert r.liquidations == 1 and r.equity_end < 10


# --- Bot con cuenta simulada ---
import logging  # noqa: E402

import numpy as np  # noqa: E402

from phoenix.perps.bot import Bot, Config  # noqa: E402
from phoenix.perps.exchange import KrakenPerp, PaperPerp, Pos  # noqa: E402


def _trend_candles(n=1300, breakout_last=True):
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    close = 60000 + np.arange(n) * 20.0 + np.sin(np.arange(n) / 5) * 300
    if breakout_last:
        close[-1] = close[-60:-1].max() + 800  # cierra por encima del máximo de 20 velas
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 50
    low = np.minimum(open_, close) - 50
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)


def _bot(tmp_path, equity=114.0):
    x = PaperPerp(equity)
    return Bot(x, Config(), tmp_path / "state.json", logging.getLogger("test")), x


def test_bot_opens_long_with_stop_and_target(tmp_path):
    bot, x = _bot(tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    assert bot.step(c, now=c.index[-1] + pd.Timedelta(hours=4))
    assert x.pos is not None and x.pos.side == 1
    assert x.pos.qty * x.last <= 9.0 * 114.0 + 1e-6
    assert x.stop < x.last < x.tp
    # el objetivo está donde el capital llegaría a 5 x 114 $
    assert x.equity() + x.pos.qty * (x.tp - x.last) == pytest.approx(570, rel=0.01)


def test_bot_refuses_account_bigger_than_planned(tmp_path):
    bot, x = _bot(tmp_path, equity=1000.0)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    with pytest.raises(RuntimeError):
        bot.step(c)


def test_bot_stops_at_target_and_state_survives_restart(tmp_path):
    bot, x = _bot(tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    bot.step(c, now=c.index[-1] + pd.Timedelta(hours=4))
    x.cash = 600.0  # el capital supera el objetivo
    assert not bot.step(c, now=c.index[-1] + pd.Timedelta(hours=5))
    assert x.pos is None
    bot2 = Bot(x, Config(), tmp_path / "state.json", logging.getLogger("test"))
    assert bot2.st.finished and not bot2.step(c)


# --- Fase 1: la posición nunca puede quedar sin stop en el exchange ---
class FlakyStop(PaperPerp):
    """set_stop falla las primeras `fail_times` veces (red caída); el stop anterior queda intacto."""

    def __init__(self, equity, fail_times=1):
        super().__init__(equity)
        self.fail_times, self.set_stop_calls = fail_times, 0

    def set_stop(self, pos, stop):
        self.set_stop_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("red caída al colocar el stop")
        super().set_stop(pos, stop)


class _Crash(BaseException):
    """El proceso muere: no la captura ningún `except Exception`."""


class CrashOnSetStop(PaperPerp):
    crash = True

    def set_stop(self, pos, stop):
        if self.crash:
            raise _Crash()
        super().set_stop(pos, stop)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr("phoenix.perps.bot.time.sleep", lambda s: None)


def _bot_with(x, tmp_path):
    return Bot(x, Config(), tmp_path / "state.json", logging.getLogger("test"))


def _wick_candles():
    """Tendencia alcista con una mecha profunda hace 5 velas: el canal de 10 velas queda MUY por debajo
    del stop de entrada, así que la estrategia no vuelve a emitir un plan para mover el stop."""
    c = _trend_candles()
    c.iloc[-5, c.columns.get_loc("low")] = c["close"].iloc[-1] - 9000.0
    return c


def _next_candle(c, step=20.0):
    last = c["close"].iloc[-1]
    row = pd.DataFrame({"open": last, "high": last + 50, "low": last - 50, "close": last + step, "volume": 1.0},
                       index=[c.index[-1] + pd.Timedelta(hours=4)])
    return pd.concat([c, row])


def _cycle(bot, candles, now):
    """Un ciclo del bucle de main(): una excepción se registra y el bot sigue."""
    try:
        return bot.step(candles, now=now)
    except Exception:  # noqa: BLE001
        return True


def _open_unprotected_scenario(x, tmp_path):
    bot = _bot_with(x, tmp_path)
    c = _wick_candles()
    x.last = float(c["close"].iloc[-1])
    return bot, c, c.index[-1] + pd.Timedelta(hours=4)


def test_same_candle_retry_restores_stop_after_failed_set_stop(tmp_path, no_sleep):
    """bot.py:112-114: tras un fallo, la vela ya estaba marcada y el reintento era un no-op."""
    x = FlakyStop(114.0, fail_times=1)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    _cycle(bot, c, now)  # abre y set_stop falla
    _cycle(bot, c, now)  # misma vela: debe reponer el stop
    assert x.pos is not None
    assert x.stop is not None, "posición abierta SIN stop tras reintentar en la misma vela"


def test_next_candle_restores_stop_even_if_strategy_believes_it_exists(tmp_path, no_sleep):
    """bot.py:153 + strategies.py:73: st.stop se anotaba antes de set_stop, la estrategia creía que el stop
    ya existía, devolvía None y la posición quedaba sin stop de forma indefinida."""
    x = FlakyStop(114.0, fail_times=1)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    _cycle(bot, c, now)
    c2 = _next_candle(c)
    x.last = float(c2["close"].iloc[-1])
    _cycle(bot, c2, now)
    assert x.pos is not None
    assert x.stop is not None, "posición abierta SIN stop en la vela siguiente (la estrategia no repone)"


def test_transient_set_stop_failure_is_retried_inside_the_step(tmp_path, no_sleep):
    x = FlakyStop(114.0, fail_times=2)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    assert bot.step(c, now=now)
    assert x.stop is not None and x.set_stop_calls == 3
    assert bot.st.stop_synced and bot.st.stop == pytest.approx(x.stop)


def test_persistent_stop_failure_closes_position_and_aborts(tmp_path, no_sleep):
    """Fail-safe: si no se puede asegurar el stop, cierre a mercado inmediato y el bot se detiene."""
    x = FlakyStop(114.0, fail_times=99)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    assert _cycle(bot, c, now) is False
    assert x.pos is None, "la posición sigue abierta sin stop"
    assert bot.st.finished and "stop" in bot.st.finished
    assert x.fills[-1][3] is True  # el último fill es el cierre reduce-only a mercado
    assert x.set_stop_calls == bot.cfg.stop_retries
    n = len(x.fills)
    assert _cycle(bot, c, now) is False and len(x.fills) == n  # abortado: no manda más órdenes


def test_failed_trailing_keeps_old_stop_and_does_not_close(tmp_path, no_sleep):
    """Mover el stop (trailing) falla pero el anterior sigue vivo: la posición está protegida, no se cierra."""
    x = FlakyStop(114.0, fail_times=0)
    bot = _bot_with(x, tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    now = c.index[-1] + pd.Timedelta(hours=4)
    assert bot.step(c, now=now)
    old_stop = x.stop
    x.fail_times = 99
    c2 = _next_candle(c, step=600.0)  # sube más de 1 ATR: piramida y sube el stop
    x.last = float(c2["close"].iloc[-1])
    assert _cycle(bot, c2, now) is True
    assert x.pos is not None and not bot.st.finished
    assert x.stop == old_stop and bot.st.stop > old_stop and bot.st.stop_synced is False
    x.fail_times = 0  # vuelve la red
    assert _cycle(bot, c2, now) is True
    assert x.stop == pytest.approx(bot.st.stop) and bot.st.stop_synced


def test_stop_missing_on_exchange_is_restored_each_cycle(tmp_path, no_sleep):
    """Se reconcilia contra el estado REAL del exchange, no contra lo que el bot cree."""
    x = FlakyStop(114.0, fail_times=0)
    bot = _bot_with(x, tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    now = c.index[-1] + pd.Timedelta(hours=4)
    bot.step(c, now=now)
    wanted = x.stop
    x.stop = None  # el stop desaparece del exchange (cancelado a mano, rechazo asíncrono...)
    assert bot.step(c, now=now)
    assert x.stop == pytest.approx(wanted)


def test_unverifiable_stop_does_not_liquidate_a_protected_position(tmp_path, monkeypatch, no_sleep):
    """Un fallo al CONSULTAR órdenes (red) no es prueba de que falte el stop: no se cierra nada."""
    x = FlakyStop(114.0, fail_times=0)
    bot = _bot_with(x, tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    now = c.index[-1] + pd.Timedelta(hours=4)
    bot.step(c, now=now)

    def boom():
        raise ConnectionError("fetch_open_orders caído")
    monkeypatch.setattr(x, "stop_price", boom, raising=False)
    assert bot.step(c, now=now)
    assert x.pos is not None and not bot.st.finished and x.stop is not None


def test_intent_is_persisted_before_the_fill_and_restart_restores_stop(tmp_path):
    """El proceso muere entre el fill y set_stop: al reiniciar, el stop previsto está en disco y se repone."""
    x = CrashOnSetStop(114.0)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    with pytest.raises(_Crash):
        bot.step(c, now=now)
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["stop_synced"] is False and saved["stop"] is not None
    assert x.pos is not None and x.stop is None
    x.crash = False
    bot2 = _bot_with(x, tmp_path)
    assert bot2.step(c, now=now)
    assert x.stop == pytest.approx(saved["stop"])
    assert json.loads((tmp_path / "state.json").read_text())["stop_synced"] is True


def test_bot_neither_trades_nor_finishes_when_equity_is_unavailable(tmp_path):
    from phoenix.perps.exchange import EquityUnavailable

    class NoEquity(PaperPerp):
        def equity(self):
            raise EquityUnavailable("sin datos flex")

    x = NoEquity(114.0)
    bot, c, now = _open_unprotected_scenario(x, tmp_path)
    with pytest.raises(EquityUnavailable):
        bot.step(c, now=now)
    assert x.pos is None and not bot.st.finished


# --- KrakenPerp contra un ccxt simulado (sin red) ---
class FakeEx:
    def __init__(self, orders=None, fail_create=False, create_returns_id=True, fail_cancel=False, balance=None):
        self.orders = list(orders or [])
        self.fail_create, self.create_returns_id, self.fail_cancel = fail_create, create_returns_id, fail_cancel
        self.balance, self.log = balance, []

    def fetch_open_orders(self, symbol):
        return list(self.orders)

    def create_order(self, symbol, typ, side, amount, price, params):
        self.log.append(("create", params.get("stopLossPrice")))
        if self.fail_create:
            raise ConnectionError("timeout creando el stop")
        if not self.create_returns_id:
            return {}
        o = {"id": f"new{len(self.log)}", "triggerPrice": params["stopLossPrice"]}
        self.orders.append(o)
        return o

    def cancel_order(self, oid, symbol):
        self.log.append(("cancel", oid))
        if self.fail_cancel:
            raise ConnectionError("timeout cancelando")
        self.orders = [o for o in self.orders if o["id"] != oid]

    def fetch_balance(self):
        return self.balance


def _kraken(ex):
    k = KrakenPerp.__new__(KrakenPerp)
    k.ex, k.mode = ex, "demo"
    return k


OLD_STOP = {"id": "old", "triggerPrice": 58000.0}
LONG = Pos(1, 0.01, 60000.0)


def test_set_stop_creates_and_confirms_new_stop_before_cancelling_previous():
    ex = FakeEx([OLD_STOP])
    _kraken(ex).set_stop(LONG, 59000.0)
    kinds = [k for k, _ in ex.log]
    assert kinds == ["create", "cancel"] and ex.log[1] == ("cancel", "old")
    assert [o["id"] for o in ex.orders] != ["old"] and len(ex.orders) == 1  # solo queda el nuevo


def test_set_stop_keeps_previous_stop_when_creation_fails():
    ex = FakeEx([OLD_STOP], fail_create=True)
    with pytest.raises(ConnectionError):
        _kraken(ex).set_stop(LONG, 59000.0)
    assert ("cancel", "old") not in ex.log and [o["id"] for o in ex.orders] == ["old"]


def test_set_stop_does_not_cancel_previous_if_new_order_is_not_confirmed():
    ex = FakeEx([OLD_STOP], create_returns_id=False)
    with pytest.raises(RuntimeError):
        _kraken(ex).set_stop(LONG, 59000.0)
    assert ("cancel", "old") not in ex.log


def test_set_stop_survives_failure_cancelling_the_superseded_stop():
    ex = FakeEx([OLD_STOP], fail_cancel=True)
    _kraken(ex).set_stop(LONG, 59000.0)  # el stop nuevo ya protege: no debe romper el ciclo
    assert any(o["id"] != "old" for o in ex.orders)


def test_stop_price_reads_the_open_stop_order_or_none():
    assert _kraken(FakeEx([OLD_STOP, {"id": "tp", "price": 70000.0}])).stop_price() == pytest.approx(58000.0)
    assert _kraken(FakeEx([{"id": "tp", "price": 70000.0}])).stop_price() is None
    assert _kraken(FakeEx([])).stop_price() is None


def test_equity_reads_flex_portfolio_value():
    bal = {"info": {"accounts": {"flex": {"portfolioValue": "114.5"}}}, "total": {}}
    assert _kraken(FakeEx(balance=bal)).equity() == pytest.approx(114.5)


@pytest.mark.parametrize("bal", [
    {"info": {"accounts": {"flex": {}}}, "total": {}},                    # flex sin ningún valor
    {"info": {"accounts": {}}, "total": {"USD": 50.0}},                   # sin flex: no se improvisa con otro dato
    {"info": {}, "total": {}},                                            # respuesta vacía
    {"info": {"accounts": {"flex": {"portfolioValue": "abc"}}}, "total": {}},   # no numérico
    {"info": {"accounts": {"flex": {"portfolioValue": float("nan")}}}, "total": {}},
    {"info": {"accounts": {"flex": {"portfolioValue": -5.0}}}, "total": {}},
])
def test_equity_raises_instead_of_returning_zero_silently(bal):
    from phoenix.perps.exchange import EquityUnavailable
    with pytest.raises(EquityUnavailable):
        _kraken(FakeEx(balance=bal)).equity()
