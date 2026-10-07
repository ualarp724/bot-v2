"""Estrategias: cada una devuelve órdenes (side, sl_dist, tp_dist, max_bars) por vela.

Todas usan el mismo esquema de salida para poder compararse: SL = sl_k * ATR,
TP = tp_m * SL, cierre por tiempo a las `max_bars` velas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _orders(index, side, feats, sl_k, tp_m, max_bars) -> pd.DataFrame:
    sl = sl_k * feats["atr"]
    return pd.DataFrame(
        {"side": np.asarray(side, int), "sl_dist": sl, "tp_dist": tp_m * sl, "max_bars": max_bars}, index=index
    )


def random_orders(bars, feats, *, seed=0, prob=0.03, sl_k=1.0, tp_m=1.5, max_bars=16):
    rng = np.random.default_rng(seed)
    side = np.where(rng.random(len(bars)) < prob, rng.choice([-1, 1], len(bars)), 0)
    return _orders(bars.index, side, feats, sl_k, tp_m, max_bars)


def donchian_orders(bars, feats, *, n=32, sl_k=1.0, tp_m=1.5, max_bars=16):
    """Ruptura del máximo/mínimo de las `n` velas anteriores, a favor de la EMA200."""
    hi = bars["high"].shift(1).rolling(n).max()
    lo = bars["low"].shift(1).rolling(n).min()
    up = (bars["close"] > hi) & (feats["dist_ema200"] > 0)
    dn = (bars["close"] < lo) & (feats["dist_ema200"] < 0)
    side = np.where(up, 1, np.where(dn, -1, 0))
    return _orders(bars.index, side, feats, sl_k, tp_m, max_bars)


def london_breakout_orders(bars, feats, *, sl_k=1.0, tp_m=1.5, max_bars=16):
    """Rango de 07:00-08:00 (hora de Londres); primera ruptura con cierre entre 08:00 y 11:00."""
    ldn = bars.index.tz_convert("Europe/London")
    day = ldn.normalize()
    hour = ldn.hour + ldn.minute / 60
    in_range = (hour >= 7) & (hour < 8)
    rng_hi = bars["high"].where(in_range).groupby(day).transform("max")
    rng_lo = bars["low"].where(in_range).groupby(day).transform("min")
    window = (hour >= 8) & (hour < 11)
    raw = np.where(window & (bars["close"] > rng_hi), 1, np.where(window & (bars["close"] < rng_lo), -1, 0))
    raw = pd.Series(raw, index=bars.index)
    first = raw.ne(0) & (raw.ne(0).astype(int).groupby(day).cumsum() == 1)
    side = np.where(first, raw, 0)
    return _orders(bars.index, side, feats, sl_k, tp_m, max_bars)


def ml_orders(bars, feats, p_long: pd.Series, p_short: pd.Series, *, threshold, sl_k=1.0, tp_m=1.5, max_bars=16):
    pl, ps = p_long.fillna(0).to_numpy(), p_short.fillna(0).to_numpy()
    side = np.where((pl >= threshold) & (pl >= ps), 1, np.where((ps >= threshold) & (ps > pl), -1, 0))
    return _orders(bars.index, side, feats, sl_k, tp_m, max_bars)


def breakeven_prob(tp_m: float, cost_r: float) -> float:
    """Probabilidad de acertar a partir de la cual la operación deja de perder: p*tp_m - (1-p) - coste = 0."""
    return (1.0 + cost_r) / (tp_m + 1.0)


def range_breakout_orders(
    bars,
    feats,
    *,
    tz="Europe/London",
    range_start=7.0,
    range_end=8.0,
    trade_end=11.0,
    trend_filter=False,
    sl_k=1.0,
    tp_m=1.5,
    max_bars=16,
):
    """Ruptura de un rango horario (por defecto Londres 07:00-08:00), una por día.

    `trend_filter`: solo a favor de la tendencia de H4 (dist_ema50 de H4 con el mismo signo).
    """
    loc = bars.index.tz_convert(tz)
    day = loc.normalize()
    hour = loc.hour + loc.minute / 60
    in_range = (hour >= range_start) & (hour < range_end)
    rng_hi = bars["high"].where(in_range).groupby(day).transform("max")
    rng_lo = bars["low"].where(in_range).groupby(day).transform("min")
    window = (hour >= range_end) & (hour < trade_end)
    up = window & (bars["close"] > rng_hi)
    dn = window & (bars["close"] < rng_lo)
    if trend_filter:
        up &= feats["h4_dist_ema50"] > 0
        dn &= feats["h4_dist_ema50"] < 0
    raw = pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=bars.index)
    first = raw.ne(0) & (raw.ne(0).astype(int).groupby(day).cumsum() == 1)
    return _orders(bars.index, np.where(first, raw, 0), feats, sl_k, tp_m, max_bars)
