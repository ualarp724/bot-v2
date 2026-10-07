"""Acceso al perpetuo BTC/USD de Kraken (PF_XBTUSD) con la misma interfaz para real, demo y papel.

- KrakenPerp(mode="demo"): entorno de pruebas de Kraken (demo-futures.kraken.com), dinero ficticio.
- KrakenPerp(mode="live"): cuenta real. Las claves se leen de las variables de entorno
  KRAKEN_FUTURES_KEY y KRAKEN_FUTURES_SECRET (o de un archivo .env que no se sube a git).
  Crea las claves SIN permiso de retirada.
- PaperPerp: cuenta simulada en memoria para el modo "dry" y los tests; usa velas públicas.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SYMBOL = "BTC/USD:USD"
ROOT = Path(__file__).resolve().parents[2]
_log = logging.getLogger("perps")


class EquityUnavailable(RuntimeError):
    """El exchange no devolvió un capital utilizable. Nunca se sustituye por 0.0: un cero falso haría que
    el bot cerrara la posición real por 'capital por debajo del mínimo'."""


@dataclass
class Pos:
    side: int  # +1 largo, -1 corto
    qty: float  # BTC
    entry: float


class OrderNotConfirmed(RuntimeError):
    """Kraken no devolvió un id verificable (o rechazó la orden): cuenta como fallo de colocación."""


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _is_duplicate(e: Exception) -> bool:
    """Kraken contestó clientOrderIdAlreadyExist (ccxt lo mapea a DuplicateOrderId)."""
    import ccxt

    return isinstance(e, ccxt.DuplicateOrderId)


def _is_stop(order: dict) -> bool:
    return (order.get("triggerPrice") or order.get("stopPrice")) is not None


def _order_id(order) -> str | None:
    """Id verificable de una orden recién creada. ccxt lo mapea desde sendStatus.order_id, pero un rechazo con
    `orderEvents` vacío vuelve SIN id: se prueban las estructuras equivalentes y, si no hay, devuelve None."""
    if not isinstance(order, dict):
        return None
    if order.get("id"):
        return str(order["id"])
    info = order.get("info") or {}
    for k in ("order_id", "orderId"):
        if info.get(k):
            return str(info[k])
    for ev in info.get("orderEvents") or []:
        for key in ("order", "new", "orderPriorExecution"):
            inner = ev.get(key) or {}
            if inner.get("orderId"):
                return str(inner["orderId"])
    return None


def _confirmed_id(order, what: str) -> str:
    oid = _order_id(order)
    status = order.get("status") if isinstance(order, dict) else None
    if oid is None or status in ("rejected", "canceled", "expired"):
        raise OrderNotConfirmed(f"Kraken no confirmó {what}: id={oid!r}, status={status!r}")
    return oid


def _load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def public_client():
    import ccxt

    return ccxt.krakenfutures({"requests_trust_env": True, "enableRateLimit": True})


def candles_4h(client, n: int = 1200) -> pd.DataFrame:
    """Últimas `n` velas de 4 h CERRADAS del perpetuo."""
    since = int((time.time() - (n + 2) * 4 * 3600) * 1000)
    rows = []
    while True:
        batch = client.fetch_ohlcv(SYMBOL, "4h", since=since, limit=500)
        if not batch:
            break
        rows += batch
        if len(batch) < 2 or batch[-1][0] <= since:
            break
        since = batch[-1][0] + 1
        if batch[-1][0] >= (time.time() - 4 * 3600) * 1000:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts")
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(hours=4) <= now].sort_index().astype(float)


class KrakenPerp:
    def __init__(self, mode: str = "demo"):
        import ccxt

        if mode not in ("demo", "live"):
            raise ValueError("mode debe ser 'demo' o 'live'")
        _load_env()
        prefix = "KRAKEN_DEMO_FUTURES" if mode == "demo" else "KRAKEN_FUTURES"
        key, secret = os.environ.get(f"{prefix}_KEY"), os.environ.get(f"{prefix}_SECRET")
        if not key or not secret:
            raise RuntimeError(f"Faltan {prefix}_KEY / {prefix}_SECRET en el entorno o en .env")
        self.ex = ccxt.krakenfutures(
            {"apiKey": key, "secret": secret, "requests_trust_env": True, "enableRateLimit": True}
        )
        if mode == "demo":
            self.ex.set_sandbox_mode(True)
        self.ex.load_markets()
        self.mode = mode

    # --- lectura ---
    def candles(self, n: int = 1200) -> pd.DataFrame:
        return candles_4h(self.ex, n)

    def price(self) -> float:
        return float(self.ex.fetch_ticker(SYMBOL)["last"])

    def equity(self) -> float:
        """Valor de la cuenta de margen (flex) en USD.

        1) flex.portfolioValue / balanceValue / marginEquity (valor real de la cartera).
        2) Si no hay ninguno válido: total['USD'] con un WARNING. En cuentas flex ccxt rellena ese campo con la
           CANTIDAD de colateral en USD, que no incluye el PnL no realizado, así que es una aproximación; solo se
           acepta si es finito y > 0 (un 0 podría ser colateral en otra divisa, no una cuenta vacía).
        Si ninguna fuente da un valor computable lanza EquityUnavailable: nunca se devuelve 0.0 en silencio."""
        bal = self.ex.fetch_balance()
        if not isinstance(bal, dict):
            raise EquityUnavailable(f"fetch_balance no devolvió un balance utilizable: {bal!r}")
        flex = ((bal.get("info") or {}).get("accounts") or {}).get("flex") or {}
        for k in ("portfolioValue", "balanceValue", "marginEquity"):
            if flex.get(k) is None:
                continue
            value = _finite(flex[k])
            if value is not None and value >= 0:
                return value
            _log.warning("equity: flex.%s no es un valor válido (%r); se ignora", k, flex[k])
        usd = _finite((bal.get("total") or {}).get("USD"))
        if usd is not None and usd > 0:
            _log.warning(
                "equity: sin flex.portfolioValue; se usa total['USD']=%.2f (cantidad de colateral USD, "
                "no incluye el PnL no realizado)",
                usd,
            )
            return usd
        raise EquityUnavailable(
            "la respuesta de Kraken no trae un capital computable "
            "(flex.portfolioValue/balanceValue/marginEquity ni total['USD'])"
        )

    def position(self) -> Pos | None:
        for p in self.ex.fetch_positions([SYMBOL]):
            qty = float(p.get("contracts") or 0)
            if qty > 0:
                return Pos(1 if p["side"] == "long" else -1, qty, float(p["entryPrice"]))
        return None

    # --- órdenes ---
    def set_leverage(self, lev: float):
        self.ex.set_leverage(int(lev), SYMBOL)

    def market(self, side: int, qty: float, reduce_only: bool = False, client_id: str | None = None) -> float:
        """Orden a mercado. Con `client_id` es idempotente: si Kraken contesta 'ya existe' la orden ya se envió
        (p. ej. un timeout que sí se aplicó) y NO se reenvía; la posición del exchange es la fuente de verdad."""
        params = {"reduceOnly": True} if reduce_only else {}
        if client_id:
            params["clientOrderId"] = client_id
        try:
            o = self.ex.create_order(SYMBOL, "market", "buy" if side == 1 else "sell", qty, None, params)
        except Exception as e:
            if client_id and _is_duplicate(e):
                _log.warning("market: el clientOrderId %s ya existe, la orden ya se envió; no se repite", client_id)
                return self.price()
            raise
        return float(o.get("average") or o.get("price") or self.price())

    def _has_open_order(self, client_id: str) -> bool:
        return any(o.get("clientOrderId") == client_id for o in self.ex.fetch_open_orders(SYMBOL))

    def _cancel_kind(self, kind: str, keep_client_id: str | None = None):
        for o in self.ex.fetch_open_orders(SYMBOL):
            if keep_client_id and o.get("clientOrderId") == keep_client_id:
                continue  # la orden que acabamos de colocar nosotros: nunca se cancela a sí misma
            if (kind == "stop" and _is_stop(o)) or (kind == "tp" and not _is_stop(o)):
                self.ex.cancel_order(o["id"], SYMBOL)

    def available_margin(self) -> float | None:
        """Margen disponible (flex.availableMargin) en USD; None si no se puede conocer. Negativo => 0.0."""
        try:
            bal = self.ex.fetch_balance()
        except Exception as e:  # noqa: BLE001
            _log.warning("No se pudo leer el margen disponible: %s", e)
            return None
        if not isinstance(bal, dict):
            return None
        flex = ((bal.get("info") or {}).get("accounts") or {}).get("flex") or {}
        value = _finite(flex.get("availableMargin"))
        return None if value is None else max(value, 0.0)

    def stop_price(self) -> float | None:
        """Precio de disparo del stop vivo en el exchange, o None si no hay ninguno."""
        stops = [o for o in self.ex.fetch_open_orders(SYMBOL) if _is_stop(o)]
        if not stops:
            return None
        return float(stops[-1].get("triggerPrice") or stops[-1].get("stopPrice"))

    def set_stop(self, pos: Pos, stop: float, client_id: str | None = None):
        """Crea el stop nuevo y lo confirma ANTES de cancelar los anteriores: si la creación falla, el
        stop vigente sigue protegiendo la posición (antes se cancelaba primero y un fallo la dejaba sin stop).
        Con `client_id` es idempotente: si el intento anterior sí llegó (timeout tras aplicarse), Kraken contesta
        'ya existe'; se da por colocado solo si esa orden está realmente abierta."""
        previous = [
            o["id"]
            for o in self.ex.fetch_open_orders(SYMBOL)
            if _is_stop(o) and not (client_id and o.get("clientOrderId") == client_id)
        ]
        params = {"stopLossPrice": round(stop), "triggerSignal": "mark", "reduceOnly": True}
        if client_id:
            params["clientOrderId"] = client_id
        try:
            new = self.ex.create_order(SYMBOL, "market", "sell" if pos.side == 1 else "buy", pos.qty, None, params)
            _confirmed_id(new, "el stop")
        except Exception as e:
            if not (client_id and _is_duplicate(e)):
                raise
            if not self._has_open_order(client_id):
                raise OrderNotConfirmed(
                    f"clientOrderId {client_id} duplicado pero no hay ninguna orden abierta con él"
                ) from e
            _log.warning(
                "set_stop: el clientOrderId %s ya existe y está abierto: el intento anterior sí llegó", client_id
            )
        for oid in previous:
            try:
                self.ex.cancel_order(oid, SYMBOL)
            except Exception as e:  # noqa: BLE001 — el stop nuevo ya protege; el sobrante se limpia en el próximo set_stop
                _log.warning("No se pudo cancelar el stop anterior %s: %s", oid, e)

    def set_take_profit(self, pos: Pos, price: float, client_id: str | None = None):
        self._cancel_kind("tp", keep_client_id=client_id)
        params = {"reduceOnly": True}
        if client_id:
            params["clientOrderId"] = client_id
        try:
            order = self.ex.create_order(
                SYMBOL, "limit", "sell" if pos.side == 1 else "buy", pos.qty, round(price), params
            )
            _confirmed_id(order, "el take profit")
        except Exception as e:
            if not (client_id and _is_duplicate(e)):
                raise
            if not self._has_open_order(client_id):
                raise OrderNotConfirmed(
                    f"clientOrderId {client_id} duplicado pero no hay ninguna orden abierta con él"
                ) from e
            _log.warning("set_take_profit: el clientOrderId %s ya existe y está abierto", client_id)

    def close_all(self):
        self.ex.cancel_all_orders(SYMBOL)
        pos = self.position()
        if pos:
            self.market(-pos.side, pos.qty, reduce_only=True)


class PaperPerp:
    """Cuenta simulada: órdenes a mercado al precio indicado y stop / take profit evaluados vela a vela."""

    def __init__(self, equity: float, taker: float = 0.0005, client=None):
        self.cash, self.taker, self.client = equity, taker, client
        self.pos: Pos | None = None
        self.stop = self.tp = None
        self.last = None
        self.fills: list[tuple] = []
        self.client_ids: set[str] = set()

    def candles(self, n: int = 1200) -> pd.DataFrame:
        return candles_4h(self.client, n)

    def price(self) -> float:
        return self.last

    def equity(self) -> float:
        return self.cash + (self.pos.side * self.pos.qty * (self.last - self.pos.entry) if self.pos else 0.0)

    def position(self) -> Pos | None:
        return self.pos

    def set_leverage(self, lev: float):
        pass

    def available_margin(self) -> float:
        """Margen libre con el máximo del exchange (10x): equity - nocional/10."""
        used = self.pos.qty * self.last / 10.0 if self.pos else 0.0
        return max(self.equity() - used, 0.0)

    def market(self, side: int, qty: float, reduce_only: bool = False, client_id: str | None = None) -> float:
        if client_id is not None:
            if client_id in self.client_ids:  # como Kraken: mismo clientOrderId = misma orden, no se repite
                return self.last
            self.client_ids.add(client_id)
        px = self.last
        self.cash -= qty * px * self.taker
        if self.pos is None:
            self.pos = Pos(side, qty, px)
        elif side == self.pos.side:
            new = self.pos.qty + qty
            self.pos = Pos(side, new, (self.pos.entry * self.pos.qty + px * qty) / new)
        else:
            closed = min(qty, self.pos.qty)
            self.cash += self.pos.side * closed * (px - self.pos.entry)
            left = self.pos.qty - closed
            self.pos = Pos(self.pos.side, left, self.pos.entry) if left > 1e-12 else None
            if self.pos is None:
                self.stop = self.tp = None
        self.fills.append((side, qty, px, reduce_only))
        return px

    def stop_price(self) -> float | None:
        return self.stop

    def set_stop(self, pos: Pos, stop: float, client_id: str | None = None):
        self.stop = stop

    def set_take_profit(self, pos: Pos, price: float, client_id: str | None = None):
        self.tp = price

    def close_all(self):
        if self.pos:
            self.market(-self.pos.side, self.pos.qty, reduce_only=True)
        self.stop = self.tp = None

    def on_price(self, p: float):
        self.on_candle(p, p, p, p)

    def on_candle(self, o: float, h: float, lo: float, c: float):
        """Avanza el mercado una vela: stop primero (peor caso), luego take profit."""
        if self.pos is not None and self.stop is not None:
            s = self.pos.side
            if (lo <= self.stop) if s == 1 else (h >= self.stop):
                self.last = min(o, self.stop) if s == 1 else max(o, self.stop)
                self.close_all()
            elif self.tp is not None and ((h >= self.tp) if s == 1 else (lo <= self.tp)):
                self.last = self.tp
                self.close_all()
        self.last = c
