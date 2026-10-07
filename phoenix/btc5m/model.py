"""Entrenar, guardar y cargar el modelo del bot BTC 5m."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from phoenix.btc5m.features import FEATURES, features_1m, features_at, proxy_target, window_starts
from phoenix.walkforward import LGBM_PARAMS

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"


def train(m: pd.DataFrame, end: pd.Timestamp, months: int = 12, lag: int = 1) -> lgb.LGBMClassifier:
    """Entrena con las ventanas de los `months` meses anteriores a `end` (sin incluir `end`)."""
    starts = window_starts(m)
    X = features_at(m, starts, lag=lag, f1=features_1m(m))
    y = proxy_target(m, starts)
    mask = (starts >= end - pd.DateOffset(months=months)) & (starts < end - pd.Timedelta(minutes=10))
    Xt, yt = X[mask], y[mask]
    ok = Xt.notna().all(axis=1) & yt.notna()
    return lgb.LGBMClassifier(**LGBM_PARAMS).fit(Xt[ok][FEATURES], yt[ok].astype(int))


def save(model: lgb.LGBMClassifier, name: str, meta: dict) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}.txt"
    model.booster_.save_model(str(path))
    (MODEL_DIR / f"{name}.json").write_text(json.dumps({**meta, "features": FEATURES}, indent=2), encoding="utf-8")
    return path


def load(name: str) -> tuple[lgb.Booster, dict]:
    booster = lgb.Booster(model_file=str(MODEL_DIR / f"{name}.txt"))
    meta = json.loads((MODEL_DIR / f"{name}.json").read_text(encoding="utf-8"))
    if meta["features"] != FEATURES:
        raise ValueError("El modelo se entrenó con otras features; hay que reentrenarlo.")
    return booster, meta
