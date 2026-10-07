"""Walk-forward: entrenar con N meses, validar los M siguientes y avanzar M meses."""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

LGBM_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "random_state": 42, "verbose": -1, "n_jobs": 4,
}


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusivo
    val_start: pd.Timestamp
    val_end: pd.Timestamp    # exclusivo


def make_windows(start: pd.Timestamp, end: pd.Timestamp, train_months=18, val_months=3) -> list[Window]:
    out = []
    ts = start
    while True:
        tr_end = ts + pd.DateOffset(months=train_months)
        va_end = tr_end + pd.DateOffset(months=val_months)
        if tr_end >= end:
            break
        out.append(Window(ts, tr_end, tr_end, min(va_end, end)))
        ts = ts + pd.DateOffset(months=val_months)
    return out


def fit_side(X: pd.DataFrame, y: pd.Series):
    ok = X.notna().all(axis=1) & y.notna()
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X[ok], y[ok].astype(int))
    return model


def train_rows(index: pd.DatetimeIndex, w: Window, purge_bars: int):
    """Filas de entrenamiento, quitando las últimas `purge_bars` para que sus etiquetas no pisen la validación."""
    mask = (index >= w.train_start) & (index < w.train_end)
    pos = mask.nonzero()[0]
    if purge_bars and len(pos) > purge_bars:
        mask[pos[-purge_bars:]] = False
    return mask
