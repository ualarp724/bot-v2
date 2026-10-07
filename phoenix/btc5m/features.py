"""Features y objetivo para los mercados BTC 5m.

Cómo resuelve Polymarket (comprobado con 864 mercados de septiembre de 2026):
- `price_to_beat` ≈ precio medio de Chainlink en el minuto ANTERIOR al inicio (TWAP de 60 s).
- `final_price` ≈ precio medio del ÚLTIMO minuto de la ventana.
- "Up" si final_price >= price_to_beat.
Con velas de Binance, el objetivo `typ(s+4) >= typ(s-1)` coincide con el resultado real en el
94 % de los mercados (typ = (máximo + mínimo + cierre) / 3 de la vela de 1 minuto).

Momento de decidir: `lag` = número de minutos antes del inicio de la ventana cuya vela es la
última usada. lag=1 → se usa hasta la vela s-1, que cierra justo al inicio (decisión en s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "r1",
    "r5",
    "r15",
    "r60",
    "r240",
    "vol60",
    "vol_ratio",
    "taker5",
    "taker60",
    "vol_rel5",
    "pos60",
    "body5",
    "upwick5",
    "gap_ref",
    "hour_sin",
    "hour_cos",
    "dow",
]


def typical(m: pd.DataFrame) -> pd.Series:
    return (m["high"] + m["low"] + m["close"]) / 3


def features_1m(m: pd.DataFrame) -> pd.DataFrame:
    """Features en la fila t usando solo velas de 1 min hasta t (incluida)."""
    c = m["close"]
    lr = np.log(c).diff()
    vol60 = lr.rolling(60).std()
    f = pd.DataFrame(index=m.index)
    for n in (1, 5, 15, 60, 240):
        f[f"r{n}"] = np.log(c / c.shift(n)) / (vol60 * np.sqrt(n))
    f["vol60"] = vol60
    f["vol_ratio"] = vol60 / lr.rolling(1440).std()
    v = m["volume"]
    f["taker5"] = m["taker_buy_base"].rolling(5).sum() / v.rolling(5).sum()
    f["taker60"] = m["taker_buy_base"].rolling(60).sum() / v.rolling(60).sum()
    f["vol_rel5"] = v.rolling(5).sum() / (v.rolling(1440).mean() * 5)
    lo, hi = m["low"].rolling(60).min(), m["high"].rolling(60).max()
    f["pos60"] = (c - lo) / (hi - lo)
    o5, h5, l5 = m["open"].shift(4), m["high"].rolling(5).max(), m["low"].rolling(5).min()
    rng5 = (h5 - l5).replace(0, np.nan)
    f["body5"] = (c - o5) / rng5
    f["upwick5"] = (h5 - np.maximum(o5, c)) / rng5
    # Distancia entre el precio actual y el precio medio del último minuto (≈ price_to_beat si es la vela s-1)
    f["gap_ref"] = np.log(c / typical(m)) / vol60
    hour = m.index.hour + m.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = m.index.dayofweek
    return f[FEATURES]


def features_at(
    m: pd.DataFrame, starts: pd.DatetimeIndex, lag: int = 1, f1: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Features para ventanas que empiezan en `starts`, usando velas hasta s - lag minutos."""
    f1 = features_1m(m) if f1 is None else f1
    X = f1.reindex(starts - pd.Timedelta(minutes=lag))
    X.index = starts
    return X


def proxy_target(m: pd.DataFrame, starts: pd.DatetimeIndex) -> pd.Series:
    """Objetivo aproximado con Binance: precio medio del minuto s+4 >= el del minuto s-1."""
    typ = typical(m)
    end = typ.reindex(starts + pd.Timedelta(minutes=4)).to_numpy()
    ref = typ.reindex(starts - pd.Timedelta(minutes=1)).to_numpy()
    y = pd.Series((end >= ref).astype(float), index=starts)
    y[np.isnan(end) | np.isnan(ref)] = np.nan
    return y


def window_starts(m: pd.DataFrame) -> pd.DatetimeIndex:
    s = m.index[m.index.minute % 5 == 0]
    return s[(s - pd.Timedelta(minutes=1) >= m.index[0]) & (s + pd.Timedelta(minutes=4) <= m.index[-1])]
