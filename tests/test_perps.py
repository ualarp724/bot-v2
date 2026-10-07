import json

import ccxt
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
from phoenix.perps.exchange import (  # noqa: E402
    KrakenPerp,
    OrderNotConfirmed,
    PaperPerp,
    Pos,
)


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

    def set_stop(self, pos, stop, client_id=None):
        self.set_stop_calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("red caída al colocar el stop")
        super().set_stop(pos, stop, client_id)


class _Crash(BaseException):
    """El proceso muere: no la captura ningún `except Exception`."""


class CrashOnSetStop(PaperPerp):
    crash = True

    def set_stop(self, pos, stop, client_id=None):
        if self.crash:
            raise _Crash()
        super().set_stop(pos, stop, client_id)


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
    row = pd.DataFrame(
        {"open": last, "high": last + 50, "low": last - 50, "close": last + step, "volume": 1.0},
        index=[c.index[-1] + pd.Timedelta(hours=4)],
    )
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
    def __init__(
        self,
        orders=None,
        fail_create=False,
        create_returns_id=True,
        fail_cancel=False,
        balance=None,
        create_response=None,
        raise_duplicate=False,
    ):
        self.orders = list(orders or [])
        self.fail_create, self.create_returns_id, self.fail_cancel = fail_create, create_returns_id, fail_cancel
        self.balance, self.log = balance, []
        self.created = []  # (tipo, lado, cantidad, precio, params) de CADA create_order recibido
        self.create_response = create_response  # respuesta fija de create_order (p. ej. un rechazo sin id)
        self.raise_duplicate = raise_duplicate  # create_order responde clientOrderIdAlreadyExist

    def fetch_open_orders(self, symbol):
        return list(self.orders)

    def fetch_ticker(self, symbol):
        return {"last": 60000.0}

    def create_order(self, symbol, typ, side, amount, price, params):
        self.created.append((typ, side, amount, price, dict(params)))
        self.log.append(("create", params.get("stopLossPrice")))
        if self.fail_create:
            raise ConnectionError("timeout creando el stop")
        if self.raise_duplicate:
            raise ccxt.DuplicateOrderId("krakenfutures: createOrder failed due to clientOrderIdAlreadyExist")
        if self.create_response is not None:
            return self.create_response
        if not self.create_returns_id:
            return {}
        o = {
            "id": f"new{len(self.log)}",
            "triggerPrice": params.get("stopLossPrice"),
            "clientOrderId": params.get("clientOrderId"),
        }
        if "stopLossPrice" in params:
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
    with pytest.raises(OrderNotConfirmed):
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


@pytest.mark.parametrize(
    "bal",
    [
        {"info": {"accounts": {"flex": {}}}, "total": {}},  # flex sin ningún valor
        {"info": {}, "total": {}},  # respuesta vacía
        {"info": {"accounts": {"flex": {"portfolioValue": "abc"}}}, "total": {}},  # no numérico
        {"info": {"accounts": {"flex": {"portfolioValue": float("nan")}}}, "total": {}},
        {"info": {"accounts": {"flex": {"portfolioValue": -5.0}}}, "total": {}},
    ],
)
def test_equity_raises_instead_of_returning_zero_silently(bal):
    from phoenix.perps.exchange import EquityUnavailable

    with pytest.raises(EquityUnavailable):
        _kraken(FakeEx(balance=bal)).equity()


def test_equity_falls_back_to_total_usd_with_a_warning_when_flex_is_missing(caplog):
    """Cuentas demo / payload alternativo: hay un valor real computable, se usa pero avisando."""
    bal = {"info": {"accounts": {}}, "total": {"USD": 50.0}}
    with caplog.at_level(logging.WARNING, logger="perps"):
        assert _kraken(FakeEx(balance=bal)).equity() == pytest.approx(50.0)
    assert any("total['USD']" in r.getMessage() and "no incluye" in r.getMessage() for r in caplog.records)


def test_equity_flex_value_takes_precedence_and_does_not_warn(caplog):
    bal = {"info": {"accounts": {"flex": {"portfolioValue": "114.5"}}}, "total": {"USD": 99.0}}
    with caplog.at_level(logging.WARNING, logger="perps"):
        assert _kraken(FakeEx(balance=bal)).equity() == pytest.approx(114.5)
    assert not caplog.records


def test_equity_invalid_flex_value_falls_back_to_total_usd_with_warnings(caplog):
    bal = {"info": {"accounts": {"flex": {"portfolioValue": "abc"}}}, "total": {"USD": 75.0}}
    with caplog.at_level(logging.WARNING, logger="perps"):
        assert _kraken(FakeEx(balance=bal)).equity() == pytest.approx(75.0)
    assert len(caplog.records) == 2  # el valor inválido y el uso del fallback


@pytest.mark.parametrize(
    "bal",
    [
        None,  # balance nulo
        {},  # balance vacío
        {"info": {"accounts": {}}, "total": {"USD": 0.0}},  # 0.0 no es un valor computable (¿colateral en otra divisa?)
        {"info": {"accounts": {}}, "total": {"USD": None}},
        {"info": {"accounts": {}}, "total": {"USD": float("nan")}},
        {"info": {"accounts": {}}, "total": {"USD": -3.0}},
        {"info": {"accounts": {}}, "total": {"USD": "n/a"}},
    ],
)
def test_equity_still_raises_when_nothing_computable(bal):
    from phoenix.perps.exchange import EquityUnavailable

    with pytest.raises(EquityUnavailable):
        _kraken(FakeEx(balance=bal)).equity()


# --- reduceOnly obligatorio en stops y órdenes condicionales de salida ---
_FAKE_MARKET = {
    "id": "PF_XBTUSD",
    "symbol": "BTC/USD:USD",
    "base": "BTC",
    "quote": "USD",
    "settle": "USD",
    "baseId": "BTC",
    "quoteId": "USD",
    "settleId": "usd",
    "type": "swap",
    "spot": False,
    "margin": False,
    "swap": True,
    "future": False,
    "option": False,
    "contract": True,
    "linear": True,
    "inverse": False,
    "active": True,
    "contractSize": 1,
    "precision": {"amount": 0.0001, "price": 1.0},
    "limits": {"amount": {"min": 0.0001}, "price": {}, "cost": {}},
    "info": {},
}


def test_every_stop_and_take_profit_order_is_sent_reduce_only():
    ex = FakeEx([])
    k = _kraken(ex)
    k.set_stop(LONG, 59000.0)
    k.set_take_profit(LONG, 70000.0)
    k.set_stop(Pos(-1, 0.01, 60000.0), 61000.0)  # también en cortos
    k.set_take_profit(Pos(-1, 0.01, 60000.0), 50000.0)
    assert len(ex.created) == 4
    for typ, side, *_, params in ex.created:
        assert params.get("reduceOnly") is True, f"orden de salida sin reduceOnly: {typ} {side} {params}"


def test_close_all_sends_a_reduce_only_market_order():
    class WithPosition(FakeEx):
        def cancel_all_orders(self, symbol):
            self.orders = []

        def fetch_positions(self, symbols):
            return [{"contracts": 0.01, "side": "long", "entryPrice": 60000.0}]

    ex = WithPosition([OLD_STOP])
    _kraken(ex).close_all()
    typ, side, *_, params = ex.created[-1]
    assert (typ, side, params.get("reduceOnly")) == ("market", "sell", True)


def test_stop_and_tp_params_become_reduce_only_orders_in_the_real_ccxt_request():
    """Los params que manda KrakenPerp, pasados por el código REAL de ccxt, llevan reduceOnly en la request."""
    import ccxt

    ex = FakeEx([])
    k = _kraken(ex)
    k.set_stop(LONG, 59000.0)
    k.set_take_profit(LONG, 70000.0)
    c = ccxt.krakenfutures()
    c.set_markets([_FAKE_MARKET])
    stop_req = c.create_order_request("BTC/USD:USD", "market", "sell", 0.01, None, ex.created[0][4])
    tp_req = c.create_order_request("BTC/USD:USD", "limit", "sell", 0.01, 70000.0, ex.created[1][4])
    assert stop_req["reduceOnly"] is True and stop_req["orderType"] == "stp" and stop_req["stopPrice"] == "59000"
    assert tp_req["reduceOnly"] is True and tp_req["orderType"] == "lmt"


# --- id de orden verificable ---
def test_order_id_is_extracted_defensively():
    from phoenix.perps.exchange import _order_id

    assert _order_id({"id": "a"}) == "a"
    assert _order_id({"id": None, "info": {"order_id": "b"}}) == "b"
    assert _order_id({"info": {"orderId": "c"}}) == "c"
    assert _order_id({"info": {"orderEvents": [{"order": {"orderId": "d"}}]}}) == "d"
    assert _order_id({"info": {"orderEvents": [{"orderPriorExecution": {"orderId": "e"}}]}}) == "e"
    for bad in (None, {}, "x", {"id": None, "info": {}}, {"info": {"orderEvents": []}}, {"id": ""}):
        assert _order_id(bad) is None


def test_set_stop_accepts_an_id_found_only_in_the_raw_response():
    ex = FakeEx([OLD_STOP], create_response={"id": None, "status": "open", "info": {"order_id": "raw-1"}})
    _kraken(ex).set_stop(LONG, 59000.0)
    assert ("cancel", "old") in ex.log  # confirmado por info.order_id: ya se puede retirar el anterior


@pytest.mark.parametrize(
    "response",
    [
        {},  # nada
        {
            "id": None,
            "status": "rejected",
            "info": {"status": "invalidPrice", "orderEvents": []},
        },  # rechazo de ccxt sin id
        {"id": "x1", "status": "rejected"},  # id pero rechazada
        {"id": "x2", "status": "canceled"},
    ],
)
def test_unverifiable_stop_is_a_placement_failure_and_keeps_the_previous_stop(response):
    ex = FakeEx([OLD_STOP], create_response=response)
    with pytest.raises(OrderNotConfirmed):
        _kraken(ex).set_stop(LONG, 59000.0)
    assert ("cancel", "old") not in ex.log and [o["id"] for o in ex.orders] == ["old"]


def test_unconfirmed_take_profit_raises():
    with pytest.raises(OrderNotConfirmed):
        _kraken(FakeEx([], create_response={"id": None, "status": "rejected"})).set_take_profit(LONG, 70000.0)


class ExchangeSim(FakeEx):
    """ccxt simulado con posición y órdenes: lo justo para que KrakenPerp trabaje dentro de Bot."""

    def __init__(self, equity=114.0, last=87000.0, stop_response=None, timeout_after_effect=0):
        super().__init__(balance={"info": {"accounts": {"flex": {"portfolioValue": str(equity)}}}, "total": {}})
        self.last, self.pos, self.stop_response = last, None, stop_response
        self.seen_cids, self.timeout_after_effect = set(), timeout_after_effect  # N stops se aplican y luego "timeout"
        self.accepted_stops = 0  # stops que el exchange ACEPTÓ en total (aunque luego se cancelen)

    def fetch_ticker(self, symbol):
        return {"last": self.last}

    def set_leverage(self, lev, symbol):
        pass

    def fetch_positions(self, symbols):
        return [] if self.pos is None else [self.pos]

    def cancel_all_orders(self, symbol):
        self.orders = []

    def create_order(self, symbol, typ, side, amount, price, params):
        self.created.append((typ, side, amount, price, dict(params)))
        n = len(self.created)
        cid = params.get("clientOrderId")
        if cid is not None:
            if cid in self.seen_cids:
                raise ccxt.DuplicateOrderId("krakenfutures: createOrder failed due to clientOrderIdAlreadyExist")
            self.seen_cids.add(cid)
        if "stopLossPrice" in params:
            if self.stop_response is not None:
                return self.stop_response
            o = {"id": f"stop{n}", "status": "open", "triggerPrice": params["stopLossPrice"], "clientOrderId": cid}
            self.orders.append(o)
            self.accepted_stops += 1
            if self.timeout_after_effect > 0:
                self.timeout_after_effect -= 1
                raise ConnectionError("timeout: la orden se aplicó pero la respuesta no llegó")
            return o
        if typ == "limit":
            o = {"id": f"tp{n}", "status": "open", "price": price, "clientOrderId": cid}
            self.orders.append(o)
            return o
        if params.get("reduceOnly"):
            self.pos = None
        else:
            self.pos = {"contracts": amount, "side": "long" if side == "buy" else "short", "entryPrice": self.last}
        return {"id": f"mkt{n}", "status": "closed", "average": self.last}


def _bot_on_sim(sim, tmp_path):
    k = _kraken(sim)
    c = _trend_candles()
    sim.last = float(c["close"].iloc[-1])
    return _bot_with(k, tmp_path), c, c.index[-1] + pd.Timedelta(hours=4)


def test_end_to_end_unverifiable_stop_triggers_retries_and_fail_safe_close(tmp_path, no_sleep):
    """Bot real + KrakenPerp real + ccxt simulado que rechaza el stop SIN id: reintentos y cierre a mercado."""
    sim = ExchangeSim(
        stop_response={"id": None, "status": "rejected", "info": {"status": "invalidPrice", "orderEvents": []}}
    )
    bot, c, now = _bot_on_sim(sim, tmp_path)
    assert bot.step(c, now=now) is False
    assert sim.pos is None and bot.st.finished
    stop_attempts = [x for x in sim.created if "stopLossPrice" in x[4]]
    assert len(stop_attempts) == bot.cfg.stop_retries
    typ, *_, params = sim.created[-1]
    assert (typ, params.get("reduceOnly")) == ("market", True)  # el último envío es el cierre reduce-only


def test_end_to_end_confirmed_stop_keeps_position_and_reconcile_is_idempotent(tmp_path, no_sleep):
    sim = ExchangeSim()
    bot, c, now = _bot_on_sim(sim, tmp_path)
    assert bot.step(c, now=now) is True
    assert sim.pos is not None and len(sim.orders) == 2  # un stop y un take profit
    exits = [x for x in sim.created if "stopLossPrice" in x[4] or x[0] == "limit"]
    assert exits and all(x[4].get("reduceOnly") is True for x in exits)
    n = len(sim.created)
    assert bot.step(c, now=now) is True and len(sim.created) == n  # mismo estado: no se duplica ninguna orden


# =====================================================================================================
# Fase 2: clientOrderId determinista, idempotencia y validaciones de cordura antes de enviar
# =====================================================================================================
CID = "11111111-2222-5333-8444-555555555555"


def _cid(n):
    return CID[:-1] + str(n)


def test_client_order_id_is_forwarded_on_every_order_type():
    ex = FakeEx([])
    k = _kraken(ex)
    k.set_stop(LONG, 59000.0, client_id=_cid(1))
    k.set_take_profit(LONG, 70000.0, client_id=_cid(2))
    k.market(1, 0.01, client_id=_cid(3))
    assert [c[4].get("clientOrderId") for c in ex.created] == [_cid(1), _cid(2), _cid(3)]


def test_orders_without_client_id_do_not_send_the_param():
    ex = FakeEx([])
    _kraken(ex).set_stop(LONG, 59000.0)
    assert "clientOrderId" not in ex.created[0][4]


def test_client_order_id_reaches_the_real_ccxt_request_as_cliOrdId():
    ex = FakeEx([])
    _kraken(ex).set_stop(LONG, 59000.0, client_id=CID)
    c = ccxt.krakenfutures()
    c.set_markets([_FAKE_MARKET])
    req = c.create_order_request("BTC/USD:USD", "market", "sell", 0.01, None, ex.created[0][4])
    assert req["cliOrdId"] == CID and req["reduceOnly"] is True


def test_duplicate_stop_id_is_success_when_that_stop_is_already_open_and_older_ones_are_cleaned():
    mine = {"id": "mine", "triggerPrice": 59000.0, "clientOrderId": CID}
    ex = FakeEx([OLD_STOP, mine], raise_duplicate=True)
    _kraken(ex).set_stop(LONG, 59000.0, client_id=CID)  # el primer intento sí llegó: no es un error
    assert ("cancel", "old") in ex.log and ("cancel", "mine") not in ex.log


def test_duplicate_stop_id_without_that_order_open_is_a_placement_failure():
    ex = FakeEx([OLD_STOP], raise_duplicate=True)
    with pytest.raises(OrderNotConfirmed):
        _kraken(ex).set_stop(LONG, 59000.0, client_id=CID)
    assert ("cancel", "old") not in ex.log


def test_duplicate_take_profit_id_is_success_and_never_cancels_itself():
    tp = {"id": "tp1", "price": 70000.0, "clientOrderId": CID}
    ex = FakeEx([tp], raise_duplicate=True)
    _kraken(ex).set_take_profit(LONG, 70000.0, client_id=CID)
    assert ("cancel", "tp1") not in ex.log


def test_duplicate_market_order_is_neither_resent_nor_fatal():
    ex = FakeEx([], raise_duplicate=True)
    assert _kraken(ex).market(1, 0.01, client_id=CID) == pytest.approx(60000.0)  # precio actual: la posición manda
    assert len(ex.created) == 1


def _flex_balance(**flex):
    return {"info": {"accounts": {"flex": flex}}, "total": {}}


def test_available_margin_reads_flex_and_is_none_when_unknown():
    assert _kraken(FakeEx(balance=_flex_balance(availableMargin="37.5"))).available_margin() == pytest.approx(37.5)
    assert _kraken(FakeEx(balance=_flex_balance(availableMargin=-4.0))).available_margin() == 0.0  # sin margen
    for bal in (
        _flex_balance(),
        _flex_balance(availableMargin=None),
        _flex_balance(availableMargin="abc"),
        _flex_balance(availableMargin=float("nan")),
        None,
        {},
    ):
        assert _kraken(FakeEx(balance=bal)).available_margin() is None


def test_client_order_ids_are_deterministic_unique_and_valid_uuids(tmp_path):
    import uuid

    bot = _bot_with(PaperPerp(114.0), tmp_path)
    c0, c1 = "2026-01-01T00:00:00+00:00", "2026-01-01T04:00:00+00:00"
    a = bot._cid("entry", c0, 1)
    assert a == bot._cid("entry", c0, 1)  # determinista
    assert str(uuid.UUID(a)) == a  # UUID válido
    assert len({a, bot._cid("entry", c0, -1), bot._cid("add", c0, 1), bot._cid("entry", c1, 1)}) == 4
    assert _bot_with(PaperPerp(114.0), tmp_path)._cid("entry", c0, 1) == a  # otro proceso: mismos ids


def test_stop_retry_after_a_timeout_that_did_apply_does_not_duplicate_the_stop(tmp_path, no_sleep):
    """El stop se coloca pero la respuesta no llega (timeout). El reintento lleva el MISMO clientOrderId: Kraken
    contesta 'ya existe' y el bot lo da por colocado, sin segundo stop (que no es reduceOnly-seguro de duplicar)."""
    sim = ExchangeSim(timeout_after_effect=1)
    bot, c, now = _bot_on_sim(sim, tmp_path)
    assert bot.step(c, now=now) is True
    stops = [o for o in sim.orders if o.get("triggerPrice") is not None]
    assert len(stops) == 1 and sim.pos is not None and not bot.st.finished
    assert len([x for x in sim.created if "stopLossPrice" in x[4]]) == 2  # el original y un único reintento
    assert sim.accepted_stops == 1, "el exchange aceptó más de un stop: hubo una ventana con stops duplicados"
    assert (
        len({x[4]["clientOrderId"] for x in sim.created if "stopLossPrice" in x[4]}) == 1
    )  # mismo id en los dos intentos
    assert bot.st.stop_synced and bot.st.stop_failures == 0


def test_stop_cancelled_externally_is_replaced_with_a_new_id_not_a_false_duplicate(tmp_path, no_sleep):
    sim = ExchangeSim()
    bot, c, now = _bot_on_sim(sim, tmp_path)
    bot.step(c, now=now)
    first = next(o for o in sim.orders if o.get("triggerPrice") is not None)
    sim.orders = [o for o in sim.orders if o.get("triggerPrice") is None]  # alguien cancela el stop a mano
    assert bot.step(c, now=now) is True
    stops = [o for o in sim.orders if o.get("triggerPrice") is not None]
    assert len(stops) == 1 and stops[0]["clientOrderId"] != first["clientOrderId"]
    assert sim.pos is not None and not bot.st.finished  # el id repetido NO dispara el fail-safe


# --- validaciones de cordura antes del envío ---
def _plan_bot(tmp_path, plan, x=None):
    x = x or PaperPerp(114.0)
    bot = _bot_with(x, tmp_path)
    bot.strategy = lambda ctx: plan(x) if callable(plan) else plan  # `plan` puede depender del precio actual
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    return bot, x, c, c.index[-1] + pd.Timedelta(hours=4)


@pytest.mark.parametrize("side,offset", [(1, 100.0), (1, 0.0), (-1, -100.0), (-1, 0.0)])
def test_new_entry_with_the_stop_on_the_wrong_side_is_not_sent(tmp_path, side, offset):
    bot, x, c, now = _plan_bot(tmp_path, lambda x: Plan(side, x.last + offset, 5.0))
    assert bot.step(c, now=now) is True
    assert x.pos is None and not x.fills and bot.st.stop is None


@pytest.mark.parametrize("stop", [float("nan"), float("inf"), -5.0, 0.0])
def test_new_entry_with_non_finite_or_non_positive_stop_is_not_sent(tmp_path, stop):
    bot, x, c, now = _plan_bot(tmp_path, Plan(1, stop, 5.0))
    assert bot.step(c, now=now) is True
    assert x.pos is None and not x.fills


def test_crossed_stop_on_an_open_long_closes_at_market_instead_of_sending_an_invalid_stop(tmp_path, no_sleep):
    """Si el canal de salida ya está roto, el stop previsto queda por encima del precio: es una SALIDA."""
    x = FlakyStop(114.0, fail_times=0)
    bot = _bot_with(x, tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    now = c.index[-1] + pd.Timedelta(hours=4)
    assert bot.step(c, now=now) and x.pos is not None
    calls = x.set_stop_calls
    c2 = _next_candle(c)
    x.last = float(c2["close"].iloc[-1])
    bot.strategy = lambda ctx: Plan(1, x.last + 50.0, 0.0)
    assert bot.step(c2, now=now) is True
    assert x.pos is None and not bot.st.finished and bot.st.stop is None
    assert x.set_stop_calls == calls and x.fills[-1][3] is True


def test_take_profit_price_is_none_when_it_cannot_be_represented(tmp_path):
    bot = _bot_with(PaperPerp(114.0), tmp_path)
    bot.st.start_equity = 114.0  # objetivo = 570 $
    assert bot._tp_price(Pos(1, 0.01, 60000.0), 114.0, 60000.0) == pytest.approx(105600.0)
    assert bot._tp_price(Pos(-1, 0.01, 60000.0), 114.0, 60000.0) == pytest.approx(14400.0)
    assert bot._tp_price(Pos(-1, 0.005, 60000.0), 50.0, 60000.0) is None  # saldría negativo: -44000
    assert bot._tp_price(Pos(1, 0.01, 60000.0), 600.0, 60000.0) is None  # ya por encima del objetivo: TP bajo el precio


def test_short_with_an_unrepresentable_take_profit_gets_its_stop_but_no_tp(tmp_path):
    bot, x, c, now = _plan_bot(tmp_path, Plan(-1, 87500.0, 0.5))  # tamaño pequeño: el TP saldría negativo
    assert bot.step(c, now=now) is True
    assert x.pos is not None and x.pos.side == -1
    assert x.stop is not None and x.tp is None


class Margin(PaperPerp):
    def __init__(self, equity, avail):
        super().__init__(equity)
        self.avail = avail

    def available_margin(self):
        return self.avail


def _pyramid_run(tmp_path, avail):
    x = Margin(114.0, avail)
    bot = _bot_with(x, tmp_path)
    c = _trend_candles()
    x.last = float(c["close"].iloc[-1])
    now = c.index[-1] + pd.Timedelta(hours=4)
    bot.step(c, now=now)
    qty0, stop0 = x.pos.qty, x.stop
    c2 = _next_candle(c, step=600.0)  # sube más de 1 ATR: toca piramidar y subir el stop
    x.last = float(c2["close"].iloc[-1])
    bot.step(c2, now=now)
    return x, qty0, stop0


def test_pyramiding_is_limited_by_the_available_margin(tmp_path):
    full, q0, s0 = _pyramid_run(tmp_path / "a", avail=1e9)
    partial, _, _ = _pyramid_run(tmp_path / "b", avail=3.0)
    zero, _, _ = _pyramid_run(tmp_path / "c", avail=0.0)
    unknown, _, _ = _pyramid_run(tmp_path / "d", avail=None)
    assert full.pos.qty > partial.pos.qty > q0  # con margen justo se añade menos
    assert zero.pos.qty == pytest.approx(q0) and unknown.pos.qty == pytest.approx(
        q0
    )  # sin margen o sin dato: no se piramida
    for x in (full, partial, zero, unknown):
        assert x.stop > s0  # pero el stop sí se sube siempre
