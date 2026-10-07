"""Velas de 1 hora de BTCUSDT (data.binance.vision) para el bot apalancado.

El perpetuo de Kraken sigue al índice de BTC; para señales de 4 h la diferencia con Binance es mínima.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phoenix.btc5m.binance import BASE, _get, read_zip

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "binance_btcusdt_1h"


def download(start: str = "2019-01") -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    n = 0
    for month in pd.period_range(start, end.to_period("M") - 1, freq="M"):
        n += _get(f"{BASE}/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-{month}.zip", DATA_DIR / f"BTCUSDT-1h-{month}.zip")
    for day in pd.date_range(end.to_period("M").start_time, end - pd.Timedelta(days=1), freq="D"):
        n += _get(
            f"{BASE}/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{day:%Y-%m-%d}.zip", DATA_DIR / f"BTCUSDT-1h-{day:%Y-%m-%d}.zip"
        )
    return n


def load_1h() -> pd.DataFrame:
    df = pd.concat([read_zip(f) for f in sorted(DATA_DIR.glob("*.zip"))]).sort_index()
    return df[~df.index.duplicated(keep="last")]
