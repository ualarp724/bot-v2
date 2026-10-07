"""Velas de 1 minuto de BTC-USD en Coinbase (en dólares, más parecido al precio de Chainlink que BTCUSDT).

API pública sin clave; 300 velas por llamada. Se guarda por días en data/raw/coinbase_btcusd_1m/.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from phoenix.btc5m.polymarket import get_json

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "coinbase_btcusd_1m"
URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


def _chunk(a: pd.Timestamp) -> list:
    b = a + pd.Timedelta(minutes=299)
    return (
        get_json(
            URL, {"granularity": 60, "start": a.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": b.strftime("%Y-%m-%dT%H:%M:%SZ")}
        )
        or []
    )


def download_day(day: pd.Timestamp) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{day:%Y-%m-%d}.csv.gz"
    if path.exists():
        return -1
    starts = pd.date_range(day, day + pd.Timedelta(minutes=1439), freq="300min")
    with ThreadPoolExecutor(5) as ex:
        rows = [r for part in ex.map(_chunk, starts) for r in part]
    df = (
        pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
        .drop_duplicates("time")
        .sort_values("time")
    )
    df = df[(df.time >= day.timestamp()) & (df.time < (day + pd.Timedelta(days=1)).timestamp())]
    df.to_csv(path, index=False)
    return len(df)


def load(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    parts = []
    for f in sorted(DATA_DIR.glob("*.csv.gz")):
        tag = f.name[:10]
        if (start and tag < start) or (end and tag > end):
            continue
        parts.append(pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True)
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True)
    return df[["open", "high", "low", "close", "volume"]].sort_index()
