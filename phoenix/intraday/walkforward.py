"""Walk-forward estricto: se decide con datos ANTERIORES a cada tramo de test y el test se simula solo con datos
posteriores. Cada tramo OOS usa únicamente lo aprendido o elegido antes de su inicio.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from phoenix.intraday.engine import Costs, Rules, Signals, simulate
from phoenix.intraday.strategies import BreakoutParams, breakout_signals
from phoenix.walkforward import LGBM_PARAMS


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusivo


def make_folds(
    index: pd.DatetimeIndex, first_test: pd.Timestamp, train_months: int = 6, test_months: int = 1
) -> list[Fold]:
    """Tramos de test consecutivos de `test_months` meses desde `first_test`; el de entrenamiento va justo antes."""
    folds, start = [], first_test
    end = index[-1] + (index[-1] - index[-2])
    while start < end:
        stop = min(start + pd.DateOffset(months=test_months), end)
        folds.append(Fold(start - pd.DateOffset(months=train_months), start, stop))
        start = start + pd.DateOffset(months=test_months)
    return folds


def tstat(r: np.ndarray) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else float("-inf")


def tune_breakout(
    bars: pd.DataFrame,
    grid: list[BreakoutParams],
    folds: list[Fold],
    costs: Costs,
    rules: Rules,
    min_trades: int = 30,
) -> tuple[Signals, pd.DataFrame]:
    """En cada tramo elige, SOLO con la ventana de entrenamiento anterior, la configuración con mejor t-estadístico de
    R neto por operación. Si ninguna tiene t > 0 con `min_trades` operaciones, ese tramo se queda fuera (en efectivo)."""
    n, idx = len(bars), bars.index
    sigs = [breakout_signals(bars, p) for p in grid]  # las señales de cada configuración no dependen del tramo
    side, dist = np.zeros(n), np.full(n, np.nan)
    xl, xs = np.zeros(n, bool), np.zeros(n, bool)
    chosen = []
    for k, f in enumerate(folds):
        a, b, e = idx.searchsorted(f.train_start), idx.searchsorted(f.test_start), idx.searchsorted(f.test_end)
        best, best_score, best_n, scores = None, 0.0, 0, []
        for j, sg in enumerate(sigs):
            tr = simulate(bars, sg, costs, rules, i0=a, i1=b).trades
            score = tstat(tr["r"].to_numpy()) if len(tr) >= min_trades else float("-inf")
            scores.append(score)
            if score > best_score:
                best, best_score, best_n = j, score, len(tr)
        chosen.append({"fold": k, "test_start": f.test_start, "params": grid[best] if best is not None else None,
                       "is_t": best_score if best is not None else float("nan"), "is_trades": best_n,
                       "is_scores": scores})  # fmt: skip
        if best is None:
            continue
        s = sigs[best]
        side[b:e], dist[b:e] = np.asarray(s.side)[b:e], np.asarray(s.stop_dist)[b:e]
        xl[b:e], xs[b:e] = np.asarray(s.exit_long)[b:e], np.asarray(s.exit_short)[b:e]
    return Signals(side=side, stop_dist=dist, exit_long=xl, exit_short=xs, tp_r=grid[0].tp_r), pd.DataFrame(chosen)


def restrict(sig: Signals, folds: list[Fold], index: pd.DatetimeIndex) -> Signals:
    """Anula las señales fuera de los tramos de test (para evaluar parámetros fijos sobre el mismo periodo OOS)."""
    keep = np.zeros(len(index), bool)
    for f in folds:
        keep[index.searchsorted(f.test_start) : index.searchsorted(f.test_end)] = True
    return Signals(
        side=np.where(keep, sig.side, 0), stop_dist=sig.stop_dist, exit_long=sig.exit_long, exit_short=sig.exit_short,
        tp_r=sig.tp_r,
    )  # fmt: skip


def make_labels(bars: pd.DataFrame, horizon: int, thr: float) -> pd.Series:
    """Clase por vela t: 2 = sube más de `thr` (entrando a la apertura de t+1 y cerrando `horizon` velas después),
    0 = baja más de `thr`, 1 = nada. NaN si el futuro no existe. Usa velas hasta t+horizon: ver la purga."""
    ret = bars["close"].shift(-horizon) / bars["open"].shift(-1) - 1
    y = pd.Series(np.where(ret > thr, 2, np.where(ret < -thr, 0, 1)), index=bars.index, dtype=float)
    return y.where(ret.notna())


def train_slice(i_train0: int, i_test0: int, horizon: int) -> slice:
    """Filas de entrenamiento [i_train0, i_test0 - horizon - 1]: la etiqueta de la última llega a i_test0 - 1, es
    decir, nunca se mira una vela del tramo de test (purga)."""
    return slice(i_train0, max(i_test0 - horizon, i_train0))


def ml_probabilities(
    bars: pd.DataFrame,
    X: pd.DataFrame,
    folds: list[Fold],
    horizon: int,
    cost_thr: float,
    params: dict | None = None,
    min_rows: int = 2000,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """LightGBM de 3 clases reentrenado en cada tramo SOLO con la ventana anterior (con purga). Devuelve, para cada
    vela de test, la probabilidad de subir y de bajar más que el coste; 0 fuera de los tramos de test."""
    n, idx = len(bars), bars.index
    y = make_labels(bars, horizon, cost_thr)
    p_up, p_dn, info = np.zeros(n), np.zeros(n), []
    prm = {**LGBM_PARAMS, **(params or {})}
    for k, f in enumerate(folds):
        a, b, e = idx.searchsorted(f.train_start), idx.searchsorted(f.test_start), idx.searchsorted(f.test_end)
        sl = train_slice(a, b, horizon)
        Xtr, ytr = X.iloc[sl], y.iloc[sl]
        ok = Xtr.notna().all(axis=1) & ytr.notna()
        if ok.sum() < min_rows or ytr[ok].nunique() < 2:
            info.append({"fold": k, "trained": False, "rows": int(ok.sum())})
            continue
        model = lgb.LGBMClassifier(objective="multiclass", **prm).fit(Xtr[ok], ytr[ok].astype(int))
        Xte = X.iloc[b:e]
        good = Xte.notna().all(axis=1).to_numpy()
        proba = np.zeros((len(Xte), 3))
        proba[np.ix_(good, model.classes_)] = model.predict_proba(Xte[good])
        p_up[b:e], p_dn[b:e] = proba[:, 2], proba[:, 0]
        info.append({"fold": k, "trained": True, "rows": int(ok.sum())})
    return p_up, p_dn, pd.DataFrame(info)


def ml_signals(
    p_up: np.ndarray, p_dn: np.ndarray, atr_series: pd.Series, p_thr: float, stop_atr: float = 2.5
) -> Signals:
    """Entra cuando la probabilidad de subir (o de bajar) supera `p_thr`; la salida es por tiempo
    (`Rules.max_hold_bars = horizonte`) o por stop."""
    side = np.where((p_up >= p_thr) & (p_up > p_dn), 1, np.where((p_dn >= p_thr) & (p_dn > p_up), -1, 0))
    return Signals(side=side, stop_dist=(stop_atr * atr_series).to_numpy())


def walk_forward_ml(
    bars: pd.DataFrame,
    X: pd.DataFrame,
    atr_series: pd.Series,
    folds: list[Fold],
    horizon: int,
    cost_thr: float,
    p_thr: float,
    stop_atr: float = 2.5,
    params: dict | None = None,
    min_rows: int = 2000,
) -> tuple[Signals, pd.DataFrame]:
    p_up, p_dn, info = ml_probabilities(bars, X, folds, horizon, cost_thr, params, min_rows)
    return ml_signals(p_up, p_dn, atr_series, p_thr, stop_atr), info
