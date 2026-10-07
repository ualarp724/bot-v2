"""Datos para el estudio intradía: velas de 1 minuto -> 5m / 15m / 4h (solo velas completas)."""

from __future__ import annotations

import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "taker_buy_base": "sum"}


def resample(df1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Velas de `minutes` minutos con índice = hora de APERTURA (UTC). Se descartan las que no tienen sus
    `minutes` velas de 1 minuto (huecos de mantenimiento del exchange), para no mezclar velas incompletas."""
    g = df1m.resample(f"{minutes}min", label="left", closed="left")
    out = g.agg(AGG)
    out = out[g["close"].count() == minutes]
    return out.astype(float)


def load_1m(start: str = "2023-01", download: bool = False) -> pd.DataFrame:
    """Velas de 1 minuto de BTCUSDT (Binance, público). El perpetuo de Kraken sigue al índice de BTC."""
    from phoenix.btc5m import binance

    if download:
        binance.download(start=start)
    return binance.load(start)


def synthetic_1m(
    days: int = 400, annual_vol: float = 0.55, seed: int = 0, start: str = "2024-01-01", dof: float | None = 4
) -> pd.DataFrame:
    """Paseo aleatorio SIN ventaja con volatilidad cambiante y colas gordas. Solo valida la tubería y sirve de modelo
    nulo (cualquier resultado positivo sobre estos datos es ruido): NO es BTC.

    El máximo y el mínimo de cada minuto son los de un puente browniano entre su apertura y su cierre, es decir, los
    extremos que de verdad alcanzaría una trayectoria continua. Así las velas de 5m/15m tienen extremos coherentes con
    el camino del precio y un backtest sin ventaja resulta neutral en bruto (con mechas inventadas, un stop paga
    mechas que no existen; sin mechas, ejecutar 'al nivel' del stop regala el salto de la rejilla de 1 minuto).

    `dof` = grados de libertad de la t de Student de los retornos por minuto (colas gordas, como BTC); `None` = gaussiano.
    Con colas gordas hay minutos que SALTAN por encima de un stop: ejecutar 'en el nivel' es entonces optimista y el
    deslizamiento tiene que cubrirlo (véase `test_engine_neutrality`).
    """
    rng = np.random.default_rng(seed)
    n = days * 1440
    per_min = annual_vol / np.sqrt(365 * 1440)
    logvol = np.zeros(days)
    for d in range(1, days):  # volatilidad que cambia de un día a otro y revierte a la media (AR(1) en log)
        logvol[d] = 0.97 * logvol[d - 1] + 0.12 * rng.normal()
    regime = np.repeat(np.exp(logvol), 1440)
    regime = regime / regime.mean()
    sigma = per_min * regime
    z = rng.normal(size=n) if dof is None else rng.standard_t(dof, n) / np.sqrt(dof / (dof - 2))
    ret = z * sigma
    close = 40_000 * np.exp(np.cumsum(ret))
    open_ = np.concatenate([[40_000.0], close[:-1]])
    # Extremos del puente browniano de O a C: P(max > m) = exp(-2 m (m - a) / s²), con a = log(C/O)
    a = np.log(close / open_)
    up = (a + np.sqrt(a**2 - 2 * sigma**2 * np.log(rng.random(n)))) / 2
    dn = (a - np.sqrt(a**2 - 2 * sigma**2 * np.log(rng.random(n)))) / 2
    high, low = open_ * np.exp(up), open_ * np.exp(dn)
    volume = rng.lognormal(0.0, 0.6, n) * 5.0
    taker = volume * np.clip(0.5 + rng.normal(0, 0.08, n), 0.05, 0.95)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "taker_buy_base": taker}, index=idx
    )
