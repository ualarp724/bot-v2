"""Motor de backtest único de Phoenix.

Reglas (ver fase 2 del plan):
- Una estrategia entrega, por cada vela t, una orden calculada SOLO con datos
  hasta el cierre de t: lado (+1 compra, -1 venta, 0 nada), distancia de SL y
  de TP en USD/oz y número máximo de velas.
- La entrada se ejecuta en la APERTURA de la vela t+1.
- Precios del CSV = BID. Compra: entra al ask, sale al bid. Venta: entra al bid,
  sale al ask. SL y TP se miden contra el precio al que se cerraría.
- Si SL y TP caen dentro de la misma vela, se asume el SL (peor caso).
- Si la vela abre ya pasado el SL, se sale a la apertura (hueco en contra).
- SL y cierre por tiempo son ejecuciones a mercado (con deslizamiento); el TP
  es una orden límite (sin deslizamiento).
- Una sola posición a la vez; las señales mientras hay posición se ignoran.
- Se cierra al final de la última vela antes de cualquier pausa (diaria, fin de
  semana o festivo), así nunca se queda nada abierto de noche.
- Riesgo fijo en USD por operación (sin interés compuesto): mide la ventaja.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phoenix.costs import commission_usd, position_size, spread_usd
from phoenix.settings import Settings

ORDER_COLUMNS = ["side", "sl_dist", "tp_dist", "max_bars"]


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    risk_usd: float
    capital_usd: float
    skipped_min_lot: int


def run_backtest(
    bars: pd.DataFrame, orders: pd.DataFrame, settings: Settings, risk_usd: float | None = None
) -> BacktestResult:
    """Ejecuta las órdenes sobre las velas y devuelve la lista de operaciones."""
    if not orders.index.equals(bars.index):
        raise ValueError("orders debe tener exactamente el mismo índice que bars")
    missing = set(ORDER_COLUMNS) - set(orders.columns)
    if missing:
        raise ValueError(f"Faltan columnas en orders: {sorted(missing)}")

    ins, costs = settings.instrument, settings.costs
    risk = settings.account.max_risk_usd if risk_usd is None else float(risk_usd)
    slip = costs.slippage_per_market_fill_usd
    size = ins.contract_size_oz

    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    spr = np.array([spread_usd(p, ins, costs) for p in bars["spread_points"].to_numpy(float)])
    side_arr = orders["side"].fillna(0).to_numpy(int)
    sl_arr = orders["sl_dist"].to_numpy(float)
    tp_arr = orders["tp_dist"].to_numpy(float)
    mb_arr = orders["max_bars"].fillna(0).to_numpy(int)

    # Última vela antes de una pausa: la siguiente empieza más tarde de lo normal
    bar_len = pd.Series(idx).diff().median()
    next_gap = np.append((idx[1:] - idx[:-1]) > bar_len, True)

    trades = []
    skipped = 0
    n = len(bars)
    t = 0
    while t < n - 1:
        side = side_arr[t]
        if side == 0 or next_gap[t]:  # sin señal, o la entrada caería tras una pausa
            t += 1
            continue
        sl_d, tp_d = sl_arr[t], tp_arr[t]
        if not (np.isfinite(sl_d) and np.isfinite(tp_d) and sl_d > 0 and tp_d > 0):
            t += 1
            continue
        lots = position_size(sl_d, risk, ins)
        if lots == 0.0:
            skipped += 1
            t += 1
            continue

        e = t + 1  # vela de entrada
        if side == 1:
            entry = o[e] + spr[e] + slip
            sl, tp = entry - sl_d, entry + tp_d
        else:
            entry = o[e] - slip
            sl, tp = entry + sl_d, entry - tp_d
        last = min(e + max(mb_arr[t], 1) - 1, n - 1)

        exit_px, reason, k = np.nan, "", e
        for k in range(e, last + 1):
            if side == 1:  # sale al bid
                if k > e and o[k] <= sl:
                    exit_px, reason = o[k] - slip, "sl_gap"
                    break
                if lo[k] <= sl:
                    exit_px, reason = sl - slip, "sl"
                    break
                if h[k] >= tp:
                    exit_px, reason = tp, "tp"
                    break
            else:  # sale al ask = bid + spread
                if k > e and o[k] + spr[k] >= sl:
                    exit_px, reason = o[k] + spr[k] + slip, "sl_gap"
                    break
                if h[k] + spr[k] >= sl:
                    exit_px, reason = sl + slip, "sl"
                    break
                if lo[k] + spr[k] <= tp:
                    exit_px, reason = tp, "tp"
                    break
            if next_gap[k] or k == last:
                exit_px = (c[k] - slip) if side == 1 else (c[k] + spr[k] + slip)
                reason = "break" if next_gap[k] and k != last else "time"
                break

        gross = (exit_px - entry) * size * lots * side
        pnl = gross - commission_usd(lots, costs)
        trades.append(
            {
                "signal_time": idx[t],
                "entry_time": idx[e],
                "exit_time": idx[k],
                "side": int(side),
                "entry": entry,
                "exit": exit_px,
                "sl": sl,
                "tp": tp,
                "lots": lots,
                "risk_usd": sl_d * size * lots,
                "pnl_usd": pnl,
                "r": pnl / (sl_d * size * lots),
                "reason": reason,
                "bars": k - e + 1,
            }
        )
        t = k + 1  # la siguiente señal se busca a partir de la vela posterior a la salida

    return BacktestResult(pd.DataFrame(trades), risk, settings.account.capital_usd, skipped)
