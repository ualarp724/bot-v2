"""Etiquetas: ¿una operación abierta en la vela siguiente llega al TP antes que al SL?

Usa exactamente las mismas reglas de ejecución que backtest.py (bid/ask, SL
primero, cierre antes de pausas y por tiempo), para que lo que aprende el modelo
sea lo que luego se opera. Las etiquetas SÍ miran el futuro (es su función):
nunca se usan como features y se purgan en el borde entre entrenamiento y validación.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenix.costs import spread_usd
from phoenix.settings import Settings


def tp_first_labels(
    bars: pd.DataFrame, sl_dist: pd.Series, tp_dist: pd.Series, max_bars: int, settings: Settings
) -> pd.DataFrame:
    """Columnas long / short: 1 si ese lado llega al TP antes que al SL, 0 si no, NaN si no se puede saber."""
    ins, costs = settings.instrument, settings.costs
    slip = costs.slippage_per_market_fill_usd
    idx = bars.index
    o, h, lo = (bars[k].to_numpy(float) for k in ("open", "high", "low"))
    spr = np.array([spread_usd(p, ins, costs) for p in bars["spread_points"].to_numpy(float)])
    sl_a, tp_a = sl_dist.to_numpy(float), tp_dist.to_numpy(float)
    bar_len = pd.Series(idx).diff().median()
    next_gap = np.append((idx[1:] - idx[:-1]) > bar_len, True)
    n = len(bars)
    out = np.full((n, 2), np.nan)
    for t in range(n - 1):
        if next_gap[t] or not (np.isfinite(sl_a[t]) and np.isfinite(tp_a[t])):
            continue
        e, last = t + 1, t + max_bars
        if last >= n:
            break
        for j, side in enumerate((1, -1)):
            if side == 1:
                entry = o[e] + spr[e] + slip
                sl, tp = entry - sl_a[t], entry + tp_a[t]
            else:
                entry = o[e] - slip
                sl, tp = entry + sl_a[t], entry - tp_a[t]
            res = 0.0
            for k in range(e, last + 1):
                if side == 1:
                    if (k > e and o[k] <= sl) or lo[k] <= sl:
                        break
                    if h[k] >= tp:
                        res = 1.0
                        break
                else:
                    if (k > e and o[k] + spr[k] >= sl) or h[k] + spr[k] >= sl:
                        break
                    if lo[k] + spr[k] <= tp:
                        res = 1.0
                        break
                if next_gap[k]:
                    break
            out[t, j] = res
    return pd.DataFrame(out, index=idx, columns=["long", "short"])
