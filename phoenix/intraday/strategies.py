"""Ruptura de canal (Donchian) intradía: la misma idea que el bot de 4h, parametrizada en velas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from phoenix.intraday.engine import Signals
from phoenix.intraday.features import atr


@dataclass(frozen=True)
class BreakoutParams:
    n_entry: int = 24  # velas del canal de entrada
    n_exit: int = 12  # velas del canal de salida
    stop_atr: float = 2.0
    atr_n: int = 14
    trend_bars: int | None = None  # EMA de tendencia (en velas); None = sin filtro
    vol_filter: bool = False  # operar solo si el ATR% está por encima de su mediana de `vol_window` velas
    vol_window: int = 2000
    hours: tuple[int, int] | None = None  # franja UTC [desde, hasta) en la que se permite entrar
    tp_r: float | None = None


def breakout_signals(bars: pd.DataFrame, p: BreakoutParams) -> Signals:
    h, lo, c = bars["high"], bars["low"], bars["close"]
    a = atr(bars, p.atr_n)
    long = c > h.shift(1).rolling(p.n_entry).max()
    short = c < lo.shift(1).rolling(p.n_entry).min()
    if p.trend_bars:
        ema = c.ewm(span=p.trend_bars, adjust=False).mean()
        long, short = long & (c > ema), short & (c < ema)
    ok = pd.Series(True, index=bars.index)
    if p.vol_filter:
        rel = a / c
        ok &= rel >= rel.rolling(p.vol_window, min_periods=p.vol_window // 2).median()
    if p.hours:
        ok &= (bars.index.hour >= p.hours[0]) & (bars.index.hour < p.hours[1])
    side = np.where(long & ok, 1, np.where(short & ok, -1, 0))
    exit_long = (c < lo.shift(1).rolling(p.n_exit).min()).to_numpy()
    exit_short = (c > h.shift(1).rolling(p.n_exit).max()).to_numpy()
    return Signals(
        side=side, stop_dist=(p.stop_atr * a).to_numpy(), exit_long=exit_long, exit_short=exit_short, tp_r=p.tp_r
    )
