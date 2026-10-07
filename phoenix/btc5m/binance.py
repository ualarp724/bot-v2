"""Velas de 1 minuto de BTCUSDT desde data.binance.vision (archivos públicos, sin API key).

Se descargan los meses completos y, para el mes en curso, los días sueltos.
Índice = hora de APERTURA de la vela en UTC. Binance pasó de milisegundos a
microsegundos en 2025; se detecta automáticamente.
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "binance_btcusdt_1m"
BASE = "https://data.binance.vision/data/spot"
COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def _get(url: str, dest: Path) -> bool:
    """Descarga con curl (respeta el proxy y los certificados del sistema). True si existe o se bajó."""
    if dest.exists():
        return True
    tmp = dest.with_suffix(".part")
    r = subprocess.run(["curl", "-sfS", "-o", str(tmp), url], capture_output=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    return True


def download(start: str = "2023-01", end: pd.Timestamp | None = None, symbol: str = "BTCUSDT") -> list[Path]:
    """Meses completos desde `start` y días del mes en curso hasta ayer."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    end = (end or pd.Timestamp.now(tz="UTC")).tz_convert(None).normalize()
    got = []
    for month in pd.period_range(start, end.to_period("M") - 1, freq="M"):
        name = f"{symbol}-1m-{month}.zip"
        if _get(f"{BASE}/monthly/klines/{symbol}/1m/{name}", DATA_DIR / name):
            got.append(DATA_DIR / name)
    for day in pd.date_range(end.to_period("M").start_time, end - pd.Timedelta(days=1), freq="D"):
        name = f"{symbol}-1m-{day:%Y-%m-%d}.zip"
        if _get(f"{BASE}/daily/klines/{symbol}/1m/{name}", DATA_DIR / name):
            got.append(DATA_DIR / name)
    return got


def read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, names=COLS)
    if not str(df.iloc[0, 0]).isdigit():  # algunos archivos traen cabecera
        df = df.iloc[1:]
    ts = df["open_time"].astype("int64")
    unit = "us" if ts.iloc[0] > 1e14 else "ms"
    df.index = pd.to_datetime(ts, unit=unit, utc=True)
    return df[["open", "high", "low", "close", "volume", "taker_buy_base"]].astype(float)


def load(start: str | None = None) -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.zip"))
    if start:
        files = [f for f in files if f.stem.split("-1m-")[1] >= start]
    df = pd.concat([read_zip(f) for f in files]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "time_utc"
    return df
