"""Estrategias para el bot apalancado, sobre velas de 4 horas. Solo usan velas ya cerradas.

Todas devuelven un `Plan` (o None para no hacer nada) al cierre de cada vela.
El tamaño se fija por riesgo: apalancamiento = riesgo_por_operación / distancia_al_stop,
limitado al máximo del exchange.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenix.perps.sim import Plan


def add_indicators(b: pd.DataFrame, ema_span: int = 300) -> pd.DataFrame:
    """ATR(14), EMA de tendencia y canales de 10/20 velas ANTERIORES. Vale para cualquier marco."""
    b = b.copy()
    prev = b["close"].shift()
    tr = pd.concat([b["high"] - b["low"], (b["high"] - prev).abs(), (b["low"] - prev).abs()], axis=1).max(axis=1)
    b["atr"] = tr.rolling(14).mean()
    b["ema_trend"] = b["close"].ewm(span=ema_span, adjust=False).mean()
    for n in (10, 20, 48):
        b[f"hi{n}"] = b["high"].shift(1).rolling(n).max()
        b[f"lo{n}"] = b["low"].shift(1).rolling(n).min()
    return b


def resample(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        bars.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def load_4h(bars1h: pd.DataFrame) -> pd.DataFrame:
    """Velas de 4 h (00, 04, 08... UTC) con indicadores; la EMA de 300 velas son ~50 días."""
    return add_indicators(resample(bars1h, "4h"), ema_span=300)


def _lev(risk: float, entry: float, stop: float) -> float:
    return risk / max(abs(entry - stop) / entry, 1e-6)


def breakout(
    n_entry=20,
    n_exit=10,
    stop_atr=2.0,
    risk=0.30,
    trend_filter=False,
    pyramid=False,
    max_leverage=10.0,
    memory: dict | None = None,
):
    """Ruptura de canal (Donchian). Salida por el canal contrario o stop de ATR. Opcional: filtro de
    tendencia de ~50 días y piramidar (volver a ponerse al máximo apalancamiento cada +1 ATR).
    `memory` guarda el precio de la última compra para piramidar (el bot lo persiste en disco)."""
    state_last_add = memory if memory is not None else {}

    def strat(st):
        r, pos = st["row"], st["pos"]
        if not np.isfinite(r["atr"]) or not np.isfinite(r[f"hi{n_entry}"]):
            return None
        if pos is None:
            up, dn = r["close"] > r[f"hi{n_entry}"], r["close"] < r[f"lo{n_entry}"]
            if trend_filter:
                up &= r["close"] > r["ema_trend"]
                dn &= r["close"] < r["ema_trend"]
            if up or dn:
                side = 1 if up else -1
                stop = r["close"] - side * stop_atr * r["atr"]
                state_last_add["px"] = r["close"]
                return Plan(side, stop, _lev(risk, r["close"], stop))
            return None
        # En posición: stop dinámico = el más favorable entre el actual y el canal contrario
        chan = r[f"lo{n_exit}"] if pos.side == 1 else r[f"hi{n_exit}"]
        new_stop = max(pos.stop, chan) if pos.side == 1 else min(pos.stop, chan)
        lev = 0.0  # 0 = no cambiar el tamaño
        if pyramid and pos.side * (r["close"] - state_last_add.get("px", pos.entry)) >= r["atr"]:
            state_last_add["px"] = r["close"]
            lev = max_leverage
            trail = r["close"] - pos.side * stop_atr * r["atr"]
            new_stop = max(new_stop, trail) if pos.side == 1 else min(new_stop, trail)
        return Plan(pos.side, new_stop, lev) if (new_stop != pos.stop or lev) else None

    return strat


def bold_trend(stop_pct=0.04, lev=10.0):
    """'Todo o nada': siempre dentro a favor de la tendencia de ~50 días, apalancamiento máximo,
    stop fijo; tras un stop vuelve a entrar en la vela siguiente."""

    def strat(st):
        r, pos = st["row"], st["pos"]
        if pos is not None or not np.isfinite(r["ema_trend"]):
            return None
        side = 1 if r["close"] > r["ema_trend"] else -1
        return Plan(side, r["close"] * (1 - side * stop_pct), lev)

    return strat


def random_entries(seed=0, prob=0.08, stop_atr=2.0, risk=0.30, n_exit=10):
    """Control: entradas al azar con el mismo tamaño y salidas que `breakout`."""
    rng = np.random.default_rng(seed)

    def strat(st):
        r, pos = st["row"], st["pos"]
        if not np.isfinite(r["atr"]) or not np.isfinite(r[f"lo{n_exit}"]):
            return None
        if pos is None:
            if rng.random() < prob:
                side = int(rng.choice([-1, 1]))
                stop = r["close"] - side * stop_atr * r["atr"]
                return Plan(side, stop, _lev(risk, r["close"], stop))
            return None
        chan = r[f"lo{n_exit}"] if pos.side == 1 else r[f"hi{n_exit}"]
        new_stop = max(pos.stop, chan) if pos.side == 1 else min(pos.stop, chan)
        return Plan(pos.side, new_stop, 0.0) if new_stop != pos.stop else None

    return strat
