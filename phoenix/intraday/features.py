"""Features intradía: ruido de alta frecuencia -> retornos cortos, volatilidad relativa, volumen y flujo de órdenes.

Todo se calcula con datos hasta el cierre de la vela t (inclusive) y nada más: la decisión se toma a ese cierre y se
ejecuta a la apertura siguiente. La prueba `test_features_are_causal` lo comprueba recortando el futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev = bars["close"].shift()
    return pd.concat([bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()], axis=1).max(
        axis=1
    )


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(bars).rolling(n).mean()


def intraday_features(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Features de una vela de `minutes` minutos. Las ventanas se miden en velas (una 'hora' = 60/minutes velas)."""
    c, o, h, lo, v = (bars[k] for k in ("close", "open", "high", "low", "volume"))
    per_day = 1440 // minutes
    lc = np.log(c)
    f = pd.DataFrame(index=bars.index)
    # Momentum / retornos cortos
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = lc.diff(n)
    d = c.diff()
    up, dn = (
        d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(),
        (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean(),
    )
    f["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    ema12, ema48 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=48, adjust=False).mean()
    f["ema_gap"] = ema12 / ema48 - 1
    f["dist_ema48"] = c / ema48 - 1
    # Volatilidad relativa (ATR corto frente a largo) y realizada
    tr = true_range(bars)
    f["atr7_pct"] = tr.rolling(7).mean() / c
    f["atr14_pct"] = tr.rolling(14).mean() / c
    f["atr_ratio"] = tr.rolling(7).mean() / tr.rolling(48).mean()
    r1 = lc.diff()
    f["rv12"] = r1.rolling(12).std()
    f["rv_ratio"] = f["rv12"] / r1.rolling(48).std()
    # Forma de la vela (microestructura del precio)
    rng = (h - lo).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - lo) / rng
    f["clv"] = (c - lo) / rng - 0.5
    # Volumen y flujo de órdenes (taker buy = compras agresivas)
    vm = v.rolling(per_day).mean()
    f["vol_z"] = (v - vm) / v.rolling(per_day).std().replace(0, np.nan)
    f["vol_ratio"] = v / v.rolling(48).mean()
    tb = bars["taker_buy_base"] / v.replace(0, np.nan)
    f["taker_ratio"] = tb
    f["taker_ratio_12"] = tb.rolling(12).mean()
    f["flow"] = (2 * bars["taker_buy_base"] - v) / v.rolling(48).mean()
    f["flow_12"] = f["flow"].rolling(12).sum()
    # VWAP anclado a la apertura del día UTC
    day = bars.index.normalize()
    tp = (h + lo + c) / 3
    vwap = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    f["vwap_dev"] = c / vwap - 1
    # Distancia a los extremos de la sesión previa de 24 velas, en ATR
    a14 = tr.rolling(14).mean()
    f["brk_up"] = (c - h.shift(1).rolling(24).max()) / a14
    f["brk_dn"] = (c - lo.shift(1).rolling(24).min()) / a14
    # Hora del día y fin de semana
    minute = bars.index.hour * 60 + bars.index.minute
    f["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    f["weekend"] = (bars.index.dayofweek >= 5).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)
