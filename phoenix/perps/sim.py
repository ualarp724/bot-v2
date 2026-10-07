"""Simulador de un futuro perpetuo de BTC con apalancamiento, en velas de cualquier duración.

Supuestos (Kraken, clientes del EEE, septiembre de 2026):
- Apalancamiento máximo 10x. Margen de mantenimiento = 50 % del inicial (liquidación a ~5 %
  en contra con 10x). La liquidación se come además el 50 % del margen de mantenimiento.
- Comisión de taker 0,05 % del nocional por lado (sin contar la promo de 0 % del primer mes),
  deslizamiento 0,02 % en órdenes a mercado y stops.
- Funding: los largos pagan 0,00125 %/h del nocional (media histórica ~0,01 % cada 8 h); a los
  cortos no se les abona nada (supuesto conservador).
- Señales al cierre de una vela, ejecución a la apertura de la siguiente.
- Dentro de una vela, si se tocan stop y objetivo, se asume el stop (peor caso).
- Al llegar al objetivo (p. ej. 500 €) se cierra todo y se para. Si el capital baja del mínimo
  operable, también se para.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Costs:
    taker: float = 0.0005
    slippage: float = 0.0002
    funding_long_per_hour: float = 0.0000125
    max_leverage: float = 10.0
    mm_fraction_of_im: float = 0.5  # margen de mantenimiento como fracción del inicial


@dataclass
class Plan:
    """Orden que una estrategia quiere ejecutar en la apertura de la vela siguiente."""

    side: int  # +1 largo, -1 corto, 0 cerrar
    stop: float  # precio del stop
    leverage: float  # nocional / capital deseado (se limita al máximo)


@dataclass
class Position:
    side: int
    qty: float  # BTC
    entry: float
    stop: float
    liq: float
    opened: pd.Timestamp


@dataclass
class Result:
    equity_end: float
    hit_target: bool
    ruined: bool
    trades: int
    liquidations: int
    curve: list = field(default_factory=list)


def liquidation_price(entry: float, side: int, leverage: float, costs: Costs) -> float:
    mm = costs.mm_fraction_of_im / leverage  # margen de mantenimiento en % del nocional
    return entry * (1 - side * (1 / leverage - mm))


def _inside(stop: float, liq: float, side: int, buffer: float = 0.003) -> float:
    """El stop nunca puede quedar más allá del precio de liquidación."""
    limit = liq * (1 + side * buffer)
    return max(stop, limit) if side == 1 else min(stop, limit)


def run(
    bars4h: pd.DataFrame,
    strategy,
    start: pd.Timestamp,
    days: int = 30,
    capital: float = 100.0,
    target: float = 500.0,
    min_equity: float = 5.0,
    costs: Costs = Costs(),
    rows: list | None = None,
) -> Result:
    """Simula una ventana de `days` días empezando en `start`. `strategy(state)` devuelve un Plan o None."""
    idx = bars4h.index
    rows = rows if rows is not None else bars4h.to_dict("records")
    hours_per_bar = (idx[1] - idx[0]).total_seconds() / 3600
    i0 = idx.searchsorted(start)
    i1 = idx.searchsorted(start + pd.Timedelta(days=days))
    o, h, lo, c = (bars4h[k].to_numpy() for k in ("open", "high", "low", "close"))
    equity, pos, pending = capital, None, None
    trades = liqs = 0
    curve = []

    def close(price: float, i: int, fee: float = costs.taker):
        nonlocal equity, pos, trades
        equity += pos.side * pos.qty * (price - pos.entry) - pos.qty * price * fee
        trades += 1
        pos = None

    for i in range(i0, i1):
        # 1) Ejecutar lo decidido al cierre anterior, a la apertura de esta vela
        if pending is not None:
            px = o[i]
            if pos is not None and (pending.side == 0 or pending.side != pos.side):
                close(px * (1 - pos.side * costs.slippage), i)
            if pending.side != 0 and equity > min_equity:
                lev = min(pending.leverage, costs.max_leverage)
                mark_now = equity + (pos.side * pos.qty * (px - pos.entry) if pos is not None else 0.0)
                qty_target = mark_now * lev / px
                if pos is None:
                    fill = px * (1 + pending.side * costs.slippage)
                    equity -= qty_target * fill * costs.taker
                    liq = liquidation_price(fill, pending.side, lev, costs)
                    pos = Position(
                        pending.side, qty_target, fill, _inside(pending.stop, liq, pending.side), liq, idx[i]
                    )
                else:  # misma dirección: ajustar tamaño (piramidar) y stop
                    add = qty_target - pos.qty
                    if add > 0:
                        fill = px * (1 + pos.side * costs.slippage)
                        equity -= add * fill * costs.taker
                        pos.entry = (pos.entry * pos.qty + fill * add) / qty_target
                        pos.qty = qty_target
                        pos.liq = liquidation_price(
                            pos.entry, pos.side, min(pos.qty * px / mark_now, costs.max_leverage), costs
                        )
                    pos.stop = _inside(pending.stop, pos.liq, pos.side)
            pending = None

        # 2) Dentro de la vela: liquidación (solo con hueco), stop u objetivo (peor caso primero)
        if pos is not None:
            adverse = lo[i] if pos.side == 1 else h[i]
            favorable = h[i] if pos.side == 1 else lo[i]
            beyond = (lambda px, lvl: px <= lvl) if pos.side == 1 else (lambda px, lvl: px >= lvl)
            if beyond(o[i], pos.liq):  # abre más allá de la liquidación: se pierde el margen
                im = pos.qty * pos.entry / costs.max_leverage
                close(o[i], i)
                equity -= 0.5 * costs.mm_fraction_of_im * im
                liqs += 1
            elif beyond(adverse, pos.stop):
                px = o[i] if beyond(o[i], pos.stop) else pos.stop
                close(px * (1 - pos.side * costs.slippage), i)
            else:
                maker = 0.0002
                tp = (target - equity + pos.side * pos.qty * pos.entry) / (pos.qty * (pos.side - maker))
                if (favorable >= tp) if pos.side == 1 else (favorable <= tp):
                    close(tp, i, fee=maker)  # orden límite
                    curve.append((idx[i], equity))
                    return Result(equity, True, False, trades, liqs, curve)
                if pos.side == 1:  # funding de la duración de la vela
                    equity -= pos.qty * c[i] * costs.funding_long_per_hour * hours_per_bar

        mark = equity + (pos.side * pos.qty * (c[i] - pos.entry) if pos else 0.0)
        curve.append((idx[i], mark))
        if mark <= min_equity:
            if pos is not None:
                close(c[i], i)
            return Result(max(equity, 0.0), False, True, trades, liqs, curve)

        # 3) La estrategia decide al cierre de la vela
        state = {"i": i, "row": rows[i], "pos": pos, "equity": mark}
        pending = strategy(state)

    if pos is not None:
        close(c[i1 - 1], i1 - 1)
    return Result(equity, equity >= target, equity <= min_equity, trades, liqs, curve)
