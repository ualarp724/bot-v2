"""Bot de alto riesgo sobre el perpetuo de BTC en Kraken: estrategia S4 de research/08.

Ruptura del canal de 20 velas de 4 h a favor de la tendencia de ~50 días, stop de 2 ATR,
30 % del capital en riesgo por operación, apalancamiento máximo 10x y piramidar cada +1 ATR.
Se para solo al llegar al objetivo (5x el capital inicial), si el capital baja del mínimo
o a los 30 días. El stop y el objetivo quedan puestos en el exchange: si el ordenador se
apaga, la posición sigue protegida.

Uso:
  python -m phoenix.perps.bot --mode dry            # simulación con velas reales, sin claves
  python -m phoenix.perps.bot --mode demo --check   # comprueba claves de la cuenta demo de Kraken
  python -m phoenix.perps.bot --mode demo           # opera en la cuenta demo (dinero ficticio)
  python -m phoenix.perps.bot --mode live --i-accept-losing-everything   # dinero real
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from phoenix.perps.exchange import KrakenPerp, PaperPerp, Pos, public_client
from phoenix.perps.sim import Plan
from phoenix.perps.strategies import breakout, load_4h

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
LOT = 0.0001  # BTC, tamaño mínimo y paso del perpetuo de Kraken


@dataclass
class Config:
    target_multiple: float = 5.0
    days: int = 30
    max_leverage: float = 9.0  # Kraken permite 10x; se deja margen para poder piramidar sin rechazos
    min_equity: float = 5.0
    max_start_equity: float = 150.0  # protección: no arrancar si la cuenta tiene más de lo previsto
    stop_retries: int = 3  # intentos de colocar el stop en un mismo ciclo antes de dar la operación por insegura
    stop_retry_delay_s: float = 2.0
    stop_unverified_cycles: int = 3  # ciclos seguidos sin poder colocar NI verificar el stop antes de cerrar
    margin_buffer: float = 0.05  # holgura sobre el margen inicial necesario al piramidar (comisiones, movimiento)
    cid_namespace: str = "phoenix-perps"  # prefijo de los clientOrderId deterministas


STOP_FALLBACK_PCT = 0.04  # stop de emergencia para una posición abierta de la que no se conoce el stop


@dataclass
class State:
    started: str | None = None
    start_equity: float | None = None
    last_candle: str | None = None
    stop: float | None = None  # stop PREVISTO: se persiste antes de operar (puede no estar aún en el exchange)
    stop_synced: bool = True  # False = hay un stop previsto que todavía no está confirmado en el exchange
    stop_failures: int = 0  # ciclos seguidos en los que no se pudo asegurar el stop
    stop_gen: int = 0  # sube cada vez que el stop desaparece del exchange: un id nuevo, no un falso "duplicado"
    memory: dict = field(default_factory=dict)  # precio de la última compra para piramidar
    finished: str | None = None

    @classmethod
    def load(cls, path: Path) -> State:
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))


class Bot:
    def __init__(self, exchange, config: Config, state_path: Path, log: logging.Logger):
        self.x, self.cfg, self.state_path, self.log = exchange, config, state_path, log
        self.st = State.load(state_path)
        self.strategy = breakout(trend_filter=True, pyramid=True, max_leverage=config.max_leverage, memory=self.st.memory)

    # --- utilidades ---
    def _qty(self, equity: float, lev: float, price: float) -> float:
        return math.floor(equity * min(lev, self.cfg.max_leverage) / price / LOT) * LOT

    def _target(self) -> float:
        return self.st.start_equity * self.cfg.target_multiple

    def _tp_price(self, pos: Pos, equity: float, price: float) -> float | None:
        """Precio al que el capital llegaría al objetivo; None si no es representable (negativo, no finito o del
        lado equivocado del precio: p. ej. un corto pequeño cuyo objetivo caería por debajo de cero)."""
        tp = price + pos.side * (self._target() - equity) / pos.qty
        if not math.isfinite(tp) or tp <= 0 or pos.side * (tp - price) <= 0:
            return None
        return tp

    def _cid(self, kind: str, *parts) -> str:
        """clientOrderId DETERMINISTA (UUID v5): la misma orden lógica produce siempre el mismo id, así que un
        reintento tras un timeout no puede duplicarla; otro proceso con el mismo estado calcula el mismo id."""
        key = "|".join([self.cfg.cid_namespace, str(getattr(self.x, "mode", "paper")), kind, *map(str, parts)])
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))

    @staticmethod
    def _stop_ok(side: int, stop: float, ref: float) -> bool:
        """El stop es un número finito y positivo y está del lado de las pérdidas respecto al precio de referencia."""
        return math.isfinite(stop) and stop > 0 and (stop < ref if side == 1 else stop > ref)

    def _finish(self, why: str):
        self.x.close_all()
        self.st.finished = f"{pd.Timestamp.now(tz='UTC').isoformat()} {why}"
        self.st.save(self.state_path)
        try:
            capital = self.x.equity()
        except Exception:  # noqa: BLE001 — el bot ya está cerrado y guardado; no se pierde el log por esto
            capital = float("nan")
        self.log.info("FIN: %s. Capital final %.2f $", why, capital)

    # --- invariante: posición abierta => stop vivo en el exchange ---
    def _stop_alive(self) -> bool | None:
        """True/False según el exchange; None si no se pudo consultar (un fallo de red no prueba que falte el stop)."""
        try:
            return self.x.stop_price() is not None
        except Exception as e:  # noqa: BLE001
            self.log.warning("No se pudo verificar el stop en el exchange: %s", e)
            return None

    def _intend_stop(self, stop: float):
        """Persiste el stop previsto ANTES de tocar el exchange: si el proceso muere entre el fill y set_stop,
        al reiniciar se sabe qué stop falta y se repone."""
        self.st.stop, self.st.stop_synced = stop, False
        self.st.save(self.state_path)

    def _place_stop(self, pos: Pos, stop: float) -> bool:
        cid = self._cid("stop", self.st.stop_gen, pos.side, f"{pos.qty:.4f}", round(stop))  # igual en cada reintento
        for i in range(1, self.cfg.stop_retries + 1):
            try:
                self.x.set_stop(pos, stop, client_id=cid)
                return True
            except Exception as e:  # noqa: BLE001
                self.log.warning("set_stop a %.0f falló (%d/%d): %s", stop, i, self.cfg.stop_retries, e)
                if i < self.cfg.stop_retries and self.cfg.stop_retry_delay_s > 0:
                    time.sleep(self.cfg.stop_retry_delay_s)
        return False

    def _secure_stop(self, pos: Pos, stop: float) -> bool:
        """Coloca el stop con reintentos. True si queda confirmado. Si no se consigue y la posición se queda
        sin protección (el exchange confirma que no hay stop, o no se puede ni verificar durante varios ciclos),
        FAIL-SAFE: cierre a mercado y el bot se detiene."""
        if self._place_stop(pos, stop):
            self.st.stop, self.st.stop_synced, self.st.stop_failures = stop, True, 0
            self.st.save(self.state_path)
            return True
        self.st.stop_failures += 1
        alive = self._stop_alive()
        if alive is True:  # no se pudo mover el stop, pero el anterior sigue vivo: protegida, se reintenta
            self.log.error("No se pudo mover el stop a %.0f; sigue activo el anterior. Se reintenta en el próximo ciclo.", stop)
        elif alive is False or self.st.stop_failures >= self.cfg.stop_unverified_cycles:
            self.log.critical("No se pudo asegurar el stop a %.0f: cierre a mercado y aborto.", stop)
            self._finish("ABORTADO: no se pudo asegurar el stop, posición cerrada a mercado")
            return False
        else:
            self.log.error("No se pudo colocar NI verificar el stop (%d/%d ciclos).",
                           self.st.stop_failures, self.cfg.stop_unverified_cycles)
        self.st.save(self.state_path)
        return False

    def _reconcile_stop(self, pos: Pos | None, equity: float) -> bool:
        """Se ejecuta CADA ciclo, antes de decidir nada, contra el estado REAL del exchange (no contra lo que
        el bot cree haber enviado). False si el stop previsto aún no está sincronizado o el bot abortó."""
        if pos is None:
            if self.st.stop is not None or not self.st.stop_synced:
                self.log.info("Posición cerrada por stop u objetivo. Capital %.2f $", equity)
                self.st.stop, self.st.stop_synced, self.st.stop_failures = None, True, 0
                self.st.save(self.state_path)
            return True
        alive = self._stop_alive()
        if self.st.stop_synced and alive is not False:
            return True  # sincronizado (o no verificable: no se actúa a ciegas)
        if alive is False and self.st.stop_synced:
            self.st.stop_gen += 1  # el stop desapareció: el id anterior ya existió, hace falta uno nuevo
            self.log.error("La posición no tiene stop en el exchange: se repone.")
        stop = self.st.stop if self.st.stop is not None else pos.entry * (1 - pos.side * STOP_FALLBACK_PCT)
        return self._secure_stop(pos, stop)

    # --- un paso del bucle ---
    def step(self, candles: pd.DataFrame, now: pd.Timestamp | None = None) -> bool:
        """Devuelve False cuando el bot ha terminado."""
        now = now or pd.Timestamp.now(tz="UTC")
        if self.st.finished:
            return False
        equity = self.x.equity()
        if self.st.started is None:
            if equity > self.cfg.max_start_equity:
                raise RuntimeError(f"La cuenta tiene {equity:.2f} $, más de {self.cfg.max_start_equity} $: "
                                   "deja solo el dinero que quieres arriesgar o sube max_start_equity.")
            self.st.started, self.st.start_equity = now.isoformat(), equity
            self.x.set_leverage(self.cfg.max_leverage)
            self.log.info("Inicio con %.2f $. Objetivo %.2f $ en %d días.", equity, self._target(), self.cfg.days)
            self.st.save(self.state_path)

        if equity >= self._target():
            self._finish("objetivo alcanzado")
            return False
        if equity <= self.cfg.min_equity:
            self._finish("capital por debajo del mínimo")
            return False
        if now >= pd.Timestamp(self.st.started) + pd.Timedelta(days=self.cfg.days):
            self._finish("fin del plazo")
            return False

        pos = self.x.position()
        if not self._reconcile_stop(pos, equity):
            return not self.st.finished  # stop pendiente de sincronizar (o bot abortado): no se opera esta vela

        last = candles.index[-1]
        if self.st.last_candle == last.isoformat():
            return True  # nada nuevo: el stop ya se ha comprobado arriba
        self.st.last_candle = last.isoformat()

        b4 = load_4h(candles)
        row = b4.iloc[-1].to_dict()
        view = None if pos is None else Pos(pos.side, pos.qty, pos.entry)
        if view is not None:
            view.stop = self.st.stop if self.st.stop is not None else pos.entry * (1 - pos.side * STOP_FALLBACK_PCT)
        plan: Plan | None = self.strategy({"i": len(b4) - 1, "row": row, "pos": view, "equity": equity})
        price = self.x.price() or row["close"]
        if plan is not None:
            self._execute(plan, pos, equity, price)
        self.st.save(self.state_path)
        return not self.st.finished

    def _exit_now(self, why: str):
        """Salida por la estrategia: cierre a mercado (reduce-only) y el bot sigue operando."""
        self.log.warning("SALIDA: %s. Cierre a mercado.", why)
        self.x.close_all()
        self.st.stop, self.st.stop_synced, self.st.stop_failures = None, True, 0

    def _cap_pyramid_by_margin(self, add: float, price: float) -> float:
        """Recorta (o anula) el tamaño a añadir según el margen disponible; sin dato, no se piramida."""
        if add < LOT:
            return add
        try:
            avail = self.x.available_margin()
        except Exception as e:  # noqa: BLE001
            self.log.warning("No se pudo leer el margen disponible: %s", e)
            avail = None
        if avail is None:
            self.log.warning("Margen disponible desconocido: no se piramida.")
            return 0.0
        factor = 1 + self.cfg.margin_buffer
        if add * price / self.cfg.max_leverage * factor <= avail:
            return add
        capped = math.floor(avail * self.cfg.max_leverage / factor / price / LOT) * LOT
        self.log.warning("Margen disponible %.2f $ insuficiente para +%.4f BTC: se añade %.4f BTC.", avail, add, max(capped, 0.0))
        return capped if capped >= LOT else 0.0

    def _execute(self, plan: Plan, pos: Pos | None, equity: float, price: float):
        if pos is not None and plan.side != pos.side:
            self.x.close_all()
            pos = None
            self.st.stop, self.st.stop_synced = None, True  # sin posición no hay stop que proteger
        if plan.side == 0:
            return
        candle = self.st.last_candle
        if pos is None:
            if not self._stop_ok(plan.side, plan.stop, price):
                self.log.warning("Señal %s descartada: stop %s inválido o del lado equivocado del precio %.0f.",
                                 plan.side, plan.stop, price)
                return
            qty = self._qty(equity, plan.leverage, price)
            if qty < LOT:
                self.log.info("Señal %s pero el tamaño es menor que el mínimo", plan.side)
                return
            self._intend_stop(plan.stop)  # a disco ANTES del fill
            fill = self.x.market(plan.side, qty, client_id=self._cid("entry", candle, plan.side))
            pos = self.x.position() or Pos(plan.side, qty, fill)
            self.log.info("ABRE %s %.4f BTC a %.0f, stop %.0f (%.1fx)", "LARGO" if plan.side == 1 else "CORTO",
                          qty, fill, plan.stop, qty * fill / equity)
        else:
            if not (math.isfinite(plan.stop) and plan.stop > 0):
                self.log.warning("Plan con stop inválido (%s): se ignora y se mantiene el stop actual.", plan.stop)
                return
            if not self._stop_ok(pos.side, plan.stop, price):  # el stop previsto ya está del otro lado: es una SALIDA
                self._exit_now(f"el stop previsto {plan.stop:.0f} ya está superado por el precio {price:.0f}")
                return
            self._intend_stop(plan.stop)  # trailing y/o piramidar: ambos mueven el stop
            if plan.leverage > 0:  # piramidar
                add = self._cap_pyramid_by_margin(self._qty(equity, plan.leverage, price) - pos.qty, price)
                if add >= LOT:
                    self.x.market(pos.side, add, client_id=self._cid("add", candle, pos.side, f"{pos.qty:.4f}"))
                    pos = self.x.position() or Pos(pos.side, pos.qty + add, pos.entry)
                    self.log.info("PIRAMIDA +%.4f BTC a %.0f (total %.4f)", add, price, pos.qty)
        if not self._secure_stop(pos, plan.stop):
            return  # sin stop confirmado no se toca el objetivo: o sigue el anterior, o se cerró (fail-safe)
        tp = self._tp_price(pos, self.x.equity(), price)
        if tp is None:
            self.log.warning("Objetivo no representable para este tamaño: solo se vigila por software.")
            self.log.info("Stop en %.0f", plan.stop)
            return
        self.x.set_take_profit(pos, tp, client_id=self._cid("tp", candle, pos.side, f"{pos.qty:.4f}", round(tp)))
        self.log.info("Stop en %.0f, objetivo en %.0f", plan.stop, tp)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["dry", "demo", "live"], default="dry")
    ap.add_argument("--check", action="store_true", help="solo comprobar conexión, saldo y posición")
    ap.add_argument("--i-accept-losing-everything", action="store_true")
    ap.add_argument("--dry-equity", type=float, default=114.0, help="capital simulado en modo dry (USD)")
    a = ap.parse_args()
    if a.mode == "live" and not a.i_accept_losing_everything:
        raise SystemExit("Para operar con dinero real añade --i-accept-losing-everything")

    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG_DIR / f"perps_bot_{a.mode}.log"), logging.StreamHandler()])
    log = logging.getLogger("perps")
    x = PaperPerp(a.dry_equity, client=public_client()) if a.mode == "dry" else KrakenPerp(a.mode)
    if a.check:
        c = x.candles(50)
        log.info("Conexión OK. Última vela cerrada %s, cierre %.0f", c.index[-1], c["close"].iloc[-1])
        if a.mode != "dry":
            log.info("Capital %.2f $, posición %s", x.equity(), x.position())
        return
    bot = Bot(x, Config(), LOG_DIR / f"perps_state_{a.mode}.json", log)
    while True:
        try:
            candles = x.candles()
            if a.mode == "dry":
                x.on_price(float(public_client().fetch_ticker("BTC/USD:USD")["last"]))
            if not bot.step(candles):
                break
        except Exception as e:  # un fallo de red no debe tumbar el bot: se registra y se reintenta
            log.exception("Error en el bucle: %s", e)
        time.sleep(60)


if __name__ == "__main__":
    main()
