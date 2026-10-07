"""Motor de backtest para el perpetuo de BTC con reglas intradía y costes de Kraken Futures.

Convenciones (las mismas para 4h, 15m y 5m, para que la comparación sea justa):
- La señal se decide al CIERRE de la vela i y se ejecuta a la APERTURA de la vela i+1. Nunca antes.
- Costes por operación: comisión de taker 0,05 % en entradas, stops y cierres forzados (más un deslizamiento
  adverso de 0,02 %); comisión de maker 0,02 % en objetivos (orden límite, sin deslizamiento, y solo se da por
  ejecutada si el precio la ATRAVIESA: high > objetivo, no basta con tocarla).
- Dentro de una vela, si se tocan stop y objetivo se asume el stop (peor caso). Un hueco más allá del stop se
  ejecuta a la apertura.
- Funding: los largos pagan por hora el 0,00125 % del nocional (supuesto conservador; los cortos no cobran nada).
- Tamaño por riesgo: pérdida en el stop = `risk_frac` del equity, con tope de apalancamiento.
- Reglas intradía: máximo de posiciones simultáneas, máximo de operaciones por día, ventana de entradas, cierre
  obligatorio a una hora (sin riesgo nocturno) y parada del día si el equity cae `daily_loss_limit` desde su valor
  al inicio del día UTC (se comprueba al cierre de cada vela y se cierra a la apertura siguiente, así que la
  pérdida del día puede rebasar el límite en lo que pierda una vela).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT = 0.0001  # BTC: tamaño mínimo y paso del perpetuo de Kraken


@dataclass(frozen=True)
class Costs:
    taker: float = 0.0005
    maker: float = 0.0002
    slippage: float = 0.0002  # mínimo estimado por operación a mercado
    funding_long_per_hour: float = 0.0000125


@dataclass(frozen=True)
class Rules:
    risk_frac: float = 0.005  # pérdida en el stop, como fracción del equity
    max_leverage: float = 5.0
    max_open: int = 1  # posiciones simultáneas (todas en el mismo sentido: la cuenta es neta)
    max_trades_per_day: int | None = 6
    daily_loss_limit: float | None = 0.02
    entry_from_min: int = 0  # minuto UTC del día desde el que se puede entrar
    entry_until_min: int = 24 * 60  # y hasta el que se puede entrar (exclusivo)
    flat_at_min: int | None = None  # cierre obligatorio a la apertura de la vela que empieza a esa hora UTC
    max_hold_bars: int | None = None
    cooldown_bars: int = 0  # velas sin entrar tras una operación perdedora


@dataclass
class Signals:
    """Decisiones tomadas al cierre de cada vela (arrays con la longitud de las velas)."""

    side: np.ndarray  # +1 largo, -1 corto, 0 nada
    stop_dist: np.ndarray  # distancia del stop al precio de entrada, en precio
    exit_long: np.ndarray | None = None  # True = cerrar el largo a la apertura siguiente
    exit_short: np.ndarray | None = None
    tp_r: float | None = None  # objetivo en múltiplos de la distancia del stop


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series  # equity a mercado al cierre de cada vela
    capital: float
    halted_days: int
    skipped: dict[str, int]


class _Pos:
    __slots__ = (
        "equity0",
        "fees",
        "fill",
        "funding",
        "i",
        "ideal",
        "qty",
        "risk_usd",
        "side",
        "slip",
        "stop",
        "time",
        "tp"
    )  # fmt: skip


def simulate(
    bars: pd.DataFrame,
    sig: Signals,
    costs: Costs = Costs(),  # noqa: B008 — dataclass frozen
    rules: Rules = Rules(),  # noqa: B008
    capital: float = 10_000.0,
    i0: int = 0,
    i1: int | None = None,
) -> Result:
    """Simula las velas [i0, i1). Las señales de la vela i0-1 NO se ejecutan (la ventana es autónoma)."""
    n = len(bars)
    i1 = n if i1 is None else i1
    idx = bars.index
    o, h, lo, c = (bars[k].to_numpy(float).tolist() for k in ("open", "high", "low", "close"))
    side_sig = np.nan_to_num(np.asarray(sig.side, float)).astype(int).tolist()
    dist_sig = np.asarray(sig.stop_dist, float).tolist()
    xl = (np.asarray(sig.exit_long, bool) if sig.exit_long is not None else np.zeros(n, bool)).tolist()
    xs = (np.asarray(sig.exit_short, bool) if sig.exit_short is not None else np.zeros(n, bool)).tolist()
    minute = (idx.hour * 60 + idx.minute).to_numpy().tolist()
    day = pd.factorize(idx.normalize())[0].tolist()
    bar_hours = (idx[1] - idx[0]).total_seconds() / 3600 if n > 1 else 0.0

    cash = capital
    positions: list[_Pos] = []
    trades: list[dict] = []
    eq = np.full(i1 - i0, np.nan)
    skipped = {"no_slot": 0, "day_limit": 0, "halted": 0, "window": 0, "cooldown": 0, "size": 0, "dist": 0}
    halted, halt_pending, halted_days = False, False, 0
    day_start_eq, last_mark, trades_today, cool_until = capital, capital, 0, -1
    prev_day = day[i0] if i1 > i0 else 0

    def close(p: _Pos, ideal: float, i: int, reason: str, maker: bool = False):
        nonlocal cash, cool_until
        fill = ideal if maker else ideal * (1 - p.side * costs.slippage)
        price_pnl = p.side * p.qty * (fill - p.fill)
        fee = p.qty * fill * (costs.maker if maker else costs.taker)
        cash += price_pnl - fee
        fees, slip = p.fees + fee, p.slip + p.qty * abs(fill - ideal)
        net = price_pnl - fees - p.funding
        gross = p.side * p.qty * (ideal - p.ideal)
        trades.append(
            {
                "entry_time": p.time, "exit_time": idx[i], "side": p.side, "qty": p.qty, "entry": p.fill, "exit": fill,
                "reason": reason, "bars": i - p.i, "gross": gross, "fees": fees, "slippage": slip,
                "funding": p.funding, "net": net, "risk_usd": p.risk_usd, "r": net / p.risk_usd if p.risk_usd else 0.0,
                "equity0": p.equity0, "notional": p.qty * p.ideal,
            }
        )  # fmt: skip
        if net < 0 and rules.cooldown_bars:
            cool_until = i + rules.cooldown_bars
        positions.remove(p)

    for i in range(i0, i1):
        if day[i] != prev_day:  # nuevo día UTC
            day_start_eq, trades_today, halted, prev_day = last_mark, 0, False, day[i]

        # 1) Cierres obligatorios y por señal, a la apertura de esta vela
        for p in list(positions):
            reason = None
            if rules.flat_at_min is not None and minute[i] >= rules.flat_at_min:
                reason = "sesion"
            elif halt_pending:
                reason = "limite_diario"
            elif rules.max_hold_bars and i - p.i >= rules.max_hold_bars:
                reason = "tiempo"
            elif i > i0 and ((p.side == 1 and xl[i - 1]) or (p.side == -1 and xs[i - 1])):
                reason = "senal"
            if reason:
                close(p, o[i], i, reason)
        halt_pending = False

        # 2) Entrada decidida al cierre anterior, ejecutada a esta apertura
        s = side_sig[i - 1] if i > i0 else 0
        if s != 0:
            dist = dist_sig[i - 1]
            if not (dist > 0 and np.isfinite(dist)):
                skipped["dist"] += 1
            elif halted:
                skipped["halted"] += 1
            elif i <= cool_until:
                skipped["cooldown"] += 1
            elif not (rules.entry_from_min <= minute[i] < rules.entry_until_min):
                skipped["window"] += 1
            elif rules.max_trades_per_day is not None and trades_today >= rules.max_trades_per_day:
                skipped["day_limit"] += 1
            elif len(positions) >= rules.max_open or (positions and positions[0].side != s):
                skipped["no_slot"] += 1
            else:
                eq_now = cash + sum(q.side * q.qty * (o[i] - q.fill) for q in positions)
                used = sum(q.qty * o[i] for q in positions)
                qty = min(rules.risk_frac * eq_now / dist, max(rules.max_leverage * eq_now - used, 0.0) / o[i])
                qty = np.floor(qty / LOT) * LOT
                if qty < LOT or eq_now <= 0:
                    skipped["size"] += 1
                else:
                    p = _Pos()
                    p.side, p.qty, p.ideal, p.i, p.time, p.equity0 = s, float(qty), o[i], i, idx[i], eq_now
                    p.fill = o[i] * (1 + s * costs.slippage)
                    p.fees, p.slip, p.funding = p.qty * p.fill * costs.taker, p.qty * abs(p.fill - o[i]), 0.0
                    p.stop = o[i] - s * dist
                    p.tp = o[i] + s * sig.tp_r * dist if sig.tp_r else None
                    p.risk_usd = p.qty * dist
                    cash -= p.fees
                    positions.append(p)
                    trades_today += 1

        # 3) Dentro de la vela: stop (con huecos) y objetivo; si se tocan los dos, el stop primero
        for p in list(positions):
            if (lo[i] <= p.stop) if p.side == 1 else (h[i] >= p.stop):
                gap = o[i] <= p.stop if p.side == 1 else o[i] >= p.stop
                close(p, o[i] if gap else p.stop, i, "stop")
            elif p.tp is not None and ((h[i] > p.tp) if p.side == 1 else (lo[i] < p.tp)):
                close(p, p.tp, i, "objetivo", maker=True)

        # 4) Funding (largos), equity a mercado y regla de pérdida diaria
        for p in positions:
            if p.side == 1:
                f = p.qty * c[i] * costs.funding_long_per_hour * bar_hours
                cash -= f
                p.funding += f
        mark = cash + sum(p.side * p.qty * (c[i] - p.fill) for p in positions)
        eq[i - i0] = last_mark = mark
        if rules.daily_loss_limit is not None and not halted and mark <= day_start_eq * (1 - rules.daily_loss_limit):
            halted, halted_days = True, halted_days + 1
            halt_pending = bool(positions)

    for p in list(positions):  # fin de la ventana: se cierra lo que quede
        close(p, c[i1 - 1], i1 - 1, "fin")
    if i1 > i0:
        eq[-1] = cash
    cols = [
        "entry_time", "exit_time", "side", "qty", "entry", "exit", "reason", "bars", "gross", "fees", "slippage",
        "funding", "net", "risk_usd", "r", "equity0", "notional",
    ]  # fmt: skip
    return Result(pd.DataFrame(trades, columns=cols), pd.Series(eq, index=idx[i0:i1]), capital, halted_days, skipped)
