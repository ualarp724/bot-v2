"""Datos públicos de Polymarket para los mercados "Bitcoin Up or Down - 5 minutos".

- Mercados (gamma-api): uno cada 5 minutos, slug `btc-updown-5m-<inicio unix>`, serie 10684.
  Incluye el resultado y los precios de Chainlink que usa para resolver:
  `price_to_beat` (precio al inicio) y `final_price`.
- Operaciones (data-api): todas las ejecuciones de takers alrededor del inicio de cada
  ventana. Son los precios a los que se podía comprar de verdad.

Todo se guarda por días en data/raw/polymarket/ y las descargas se reanudan solas.
Solo lectura: aquí no se opera.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "polymarket"
GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SERIES_ID = 10684
HEADERS = {"User-Agent": "Mozilla/5.0 (phoenix-research)", "Accept": "application/json"}
TRADES_BEFORE_S = 120  # segundos antes del inicio
TRADES_AFTER_S = 60  # segundos después del inicio


def get_json(url: str, params: dict | None = None, retries: int = 5):
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            raise


def parse_event(e: dict) -> dict | None:
    if not e.get("markets"):
        return None
    m = e["markets"][0]
    outcomes = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
    tokens = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
    prices = (
        json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else (m.get("outcomePrices") or [])
    )
    up_i = outcomes.index("Up")
    meta = e.get("eventMetadata") or {}
    resolved = m.get("umaResolutionStatus") == "resolved" and len(prices) == 2
    return {
        "slug": e["slug"],
        "start_ts": int(e["slug"].rsplit("-", 1)[1]),
        "condition_id": m["conditionId"],
        "up_token": tokens[up_i],
        "down_token": tokens[1 - up_i],
        "outcome_up": (float(prices[up_i]) == 1.0) if resolved else None,
        "price_to_beat": meta.get("priceToBeat"),
        "final_price": meta.get("finalPrice"),
        "volume": m.get("volumeNum") or m.get("volume"),
        "fee_rate": (m.get("feeSchedule") or {}).get("rate"),
    }


def markets_for_day(day: pd.Timestamp) -> pd.DataFrame:
    """Todos los mercados cuya ventana termina ese día (UTC)."""
    rows = []
    for h in range(0, 24, 6):  # tramos de 6 h = 72 mercados (< 100 por llamada)
        a = day + pd.Timedelta(hours=h)
        b = a + pd.Timedelta(hours=6) - pd.Timedelta(seconds=1)
        evs = get_json(
            f"{GAMMA}/events",
            {
                "series_id": SERIES_ID,
                "limit": 100,
                "end_date_min": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": b.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        rows += [r for r in map(parse_event, evs) if r]
    df = pd.DataFrame(rows)
    return df.drop_duplicates("slug").sort_values("start_ts").reset_index(drop=True) if len(df) else df


def trades_around_start(condition_id: str, start_ts: int) -> list[dict]:
    d = get_json(
        f"{DATA_API}/trades",
        {"market": condition_id, "start": start_ts - TRADES_BEFORE_S, "end": start_ts + TRADES_AFTER_S, "limit": 10000},
    )
    return [
        {
            "ts": t["timestamp"],
            "side": t["side"],
            "outcome": t["outcome"],
            "price": float(t["price"]),
            "size": float(t["size"]),
        }
        for t in d
    ]


def download_day(day: pd.Timestamp, workers: int = 8) -> tuple[int, int]:
    """Mercados + operaciones de un día. Devuelve (mercados, operaciones). Se salta si ya está."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{day:%Y-%m-%d}"
    m_path, t_path = DATA_DIR / f"markets_{tag}.csv", DATA_DIR / f"trades_{tag}.csv.gz"
    if m_path.exists() and t_path.exists():
        return -1, -1
    markets = markets_for_day(day)
    if markets.empty:
        markets.to_csv(m_path, index=False)
        pd.DataFrame(columns=["slug", "ts", "side", "outcome", "price", "size"]).to_csv(t_path, index=False)
        return 0, 0

    def one(row):
        try:
            return [{"slug": row.slug, **t} for t in trades_around_start(row.condition_id, row.start_ts)]
        except Exception:  # noqa: BLE001 — se devuelve None y, si falla alguno, el día entero se reintenta
            return None

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, markets.itertuples()))
    if any(r is None for r in results):  # algo falló: no se guarda el día para reintentarlo entero
        raise RuntimeError(f"{tag}: {sum(r is None for r in results)} mercados sin operaciones")
    trades = pd.DataFrame([t for r in results for t in r])
    markets.to_csv(m_path, index=False)
    trades.to_csv(t_path, index=False)
    return len(markets), len(trades)


def load_buys_near_start(
    start: str | None = None, end: str | None = None, rel_from: int = -60, rel_to: int = 15
) -> pd.DataFrame:
    """Compras de takers entre inicio + rel_from y inicio + rel_to (s). Carga día a día para no llenar la memoria."""
    parts = []
    for f in sorted(DATA_DIR.glob("trades_*.csv.gz")):
        tag = f.name.split("_")[1][:10]
        if (start and tag < start) or (end and tag > end):
            continue
        df = pd.read_csv(f, dtype={"slug": "category", "side": "category", "outcome": "category"})
        if df.empty:
            continue
        start_ts = df["slug"].astype(str).str.rsplit("-", n=1).str[1].astype("int64")
        rel = df["ts"] - start_ts
        parts.append(df[(df["side"] == "BUY") & (rel >= rel_from) & (rel <= rel_to)])
    if not parts:
        return pd.DataFrame(columns=["slug", "ts", "side", "outcome", "price", "size"])
    out = pd.concat(parts, ignore_index=True)
    for c in ("slug", "side", "outcome"):
        out[c] = out[c].astype(str)
    return out


def load(kind: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """kind = 'markets' o 'trades'. Fechas YYYY-MM-DD inclusivas."""
    pattern = "markets_*.csv" if kind == "markets" else "trades_*.csv.gz"
    parts = []
    for f in sorted(DATA_DIR.glob(pattern)):
        tag = f.name.split("_")[1][:10]
        if (start and tag < start) or (end and tag > end):
            continue
        df = pd.read_csv(f, dtype={"up_token": str, "down_token": str})
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
