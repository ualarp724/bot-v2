"""Velas de marco superior (H1, H4...) sin mirar el futuro.

Una vela H1 que empieza a las 10:00 solo se conoce cuando cierra, a las 11:00.
Con velas M15, eso es al cierre de la vela base de las 10:45. Por eso el valor de
la vela H1 se asigna a partir de esa vela base (y no desde las 10:00, que era la
fuga de `legacy/core/mtf.py`). Una vela superior a medias (la actual en vivo) no
se usa nunca: así backtest y vivo ven exactamente lo mismo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def higher_tf_closed(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """OHLC del marco `rule` alineado a las velas base, solo con velas superiores completas."""
    base = pd.Series(bars.index).diff().median()
    step = pd.Timedelta(rule)
    grouped = bars.resample(rule, label="left", closed="left")
    agg = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    last_base = bars.index.to_series().resample(rule, label="left", closed="left").max().reindex(agg.index)

    # Completa si llegó a su última vela base posible, o si el mercado cerró justo
    # después (pausa diaria, fin de semana, festivo).
    last_idx = pd.DatetimeIndex(last_base)
    pos = bars.index.get_indexer(last_idx)
    nxt = pd.Series(bars.index[1:].append(pd.DatetimeIndex([pd.NaT], tz=bars.index.tz)))
    next_start = pd.DatetimeIndex(nxt.iloc[pos])
    reached_end = last_idx == (agg.index + step - base)
    market_closed = (~next_start.isna()) & (next_start > last_idx + base)
    complete = np.asarray(reached_end) | np.asarray(market_closed)

    agg = agg[complete]
    agg.index = last_idx[complete]
    return agg.reindex(bars.index, method="ffill")
