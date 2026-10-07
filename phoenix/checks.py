"""Comprobación anti-fuga: lo calculado en t no puede cambiar si cambia el futuro."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def lookahead_violations(fn: Callable[[pd.DataFrame], pd.DataFrame | pd.Series], bars: pd.DataFrame,
                         cut_points: list[int], seed: int = 0) -> list[pd.Timestamp]:
    """Devuelve los instantes t en los que `fn` cambia al alterar las velas posteriores a t.

    Para cada t de `cut_points` (posiciones), se desordenan al azar los precios de
    las velas posteriores a t y se compara `fn` hasta t con el cálculo original.
    Lista vacía = no se ha detectado fuga.
    """
    rng = np.random.default_rng(seed)
    base = fn(bars)
    bad = []
    for pos in cut_points:
        future = bars.iloc[pos + 1:].copy()
        noise = rng.normal(1.0, 0.02, size=len(future))
        for col in ("open", "high", "low", "close"):
            future[col] = future[col] * noise
        future["high"] = future[["open", "high", "low", "close"]].max(axis=1)
        future["low"] = future[["open", "high", "low", "close"]].min(axis=1)
        altered = pd.concat([bars.iloc[:pos + 1], future])
        a = base.iloc[:pos + 1] if isinstance(base, (pd.DataFrame, pd.Series)) else base
        b = fn(altered).iloc[:pos + 1]
        if not np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True):
            bad.append(bars.index[pos])
    return bad
