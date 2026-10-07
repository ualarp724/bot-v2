"""Features de Phoenix: pocas, normalizadas por ATR y calculadas solo con el pasado.

Cada fila t usa información hasta el CIERRE de la vela t. La señal que salga de
la fila t se ejecuta en la apertura de t+1 (ver backtest.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenix.timeframes import higher_tf_closed

FEATURES = [
    "ret_1",
    "ret_4",
    "ret_16",
    "ret_64",
    "vol_regime",
    "natr",
    "range_pos_32",
    "dist_ema50",
    "dist_ema200",
    "rsi_14",
    "body",
    "upper_wick",
    "lower_wick",
    "tick_vol_rel",
    "hour_sin",
    "hour_cos",
    "dow",
    "h1_ret",
    "h1_dist_ema50",
    "h4_ret",
    "h4_dist_ema50",
]


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = bars["close"].shift()
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def _htf_block(bars: pd.DataFrame, rule: str, prefix: str) -> pd.DataFrame:
    h = higher_tf_closed(bars, rule)
    # Serie de velas superiores únicas para calcular sus indicadores sin repetir filas
    uniq = h[~h["close"].eq(h["close"].shift()) | h["open"].ne(h["open"].shift())].dropna()
    a = atr(uniq, 14)
    ema = uniq["close"].ewm(span=50, adjust=False).mean()
    block = pd.DataFrame(
        {
            f"{prefix}_ret": (uniq["close"] - uniq["open"]) / a,
            f"{prefix}_dist_ema50": (uniq["close"] - ema) / a,
        }
    )
    return block.reindex(bars.index, method="ffill")


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    c = bars["close"]
    a = atr(bars, 14)
    f = pd.DataFrame(index=bars.index)
    for n in (1, 4, 16, 64):
        f[f"ret_{n}"] = (c - c.shift(n)) / a
    f["vol_regime"] = a / a.rolling(96 * 5).mean()
    f["natr"] = a / c * 100
    lo32, hi32 = bars["low"].rolling(32).min(), bars["high"].rolling(32).max()
    f["range_pos_32"] = (c - lo32) / (hi32 - lo32).replace(0, np.nan)
    f["dist_ema50"] = (c - c.ewm(span=50, adjust=False).mean()) / a
    f["dist_ema200"] = (c - c.ewm(span=200, adjust=False).mean()) / a
    f["rsi_14"] = rsi(c, 14)
    f["body"] = (c - bars["open"]) / a
    f["upper_wick"] = (bars["high"] - bars[["open", "close"]].max(axis=1)) / a
    f["lower_wick"] = (bars[["open", "close"]].min(axis=1) - bars["low"]) / a
    f["tick_vol_rel"] = bars["tick_volume"] / bars["tick_volume"].rolling(96).mean()
    ny = bars.index.tz_convert("America/New_York")
    hour = ny.hour + ny.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = ny.dayofweek
    f = f.join(_htf_block(bars, "1h", "h1")).join(_htf_block(bars, "4h", "h4"))
    f["atr"] = a  # no es feature del modelo: lo usan las estrategias para el SL
    return f[[*FEATURES, "atr"]]
