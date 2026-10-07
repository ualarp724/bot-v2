"""Bot en SIMULACIÓN para los mercados BTC 5m de Polymarket. No envía órdenes.

Cada 5 minutos, justo al empezar una ventana:
1. Descarga las últimas velas de 1 min de Binance (API pública de datos de mercado).
2. Calcula las mismas features que el backtest y la probabilidad de "Up" con el modelo guardado.
3. Lee el libro de órdenes de Polymarket de "Up" y "Down" y toma el mejor precio de venta (ask).
4. Aplica la misma regla que el backtest: apuesta si p - ask - comisión > margen y hay tamaño.
5. Apunta la apuesta simulada y, cuando la ventana se resuelve, su resultado.

Todo queda en logs/paper_btc5m.csv para comparar la simulación en vivo con el backtest.
Uso: python -m phoenix.btc5m.paper --model btc5m_lgbm [--minutes 60]
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from phoenix.btc5m.backtest import fee_per_share
from phoenix.btc5m.features import FEATURES, features_1m
from phoenix.btc5m.model import load as load_model
from phoenix.btc5m.polymarket import CLOB, GAMMA, get_json, parse_event

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "logs" / "paper_btc5m.csv"
BINANCE = "https://data-api.binance.vision/api/v3/klines"
FIELDS = ["start_utc", "slug", "p_up", "ask_up", "ask_up_size", "ask_down", "ask_down_size", "side", "price",
          "shares", "exp_edge", "decided_at_s", "outcome_up", "won", "pnl"]


def recent_klines(n: int = 1700) -> pd.DataFrame:
    """Últimas `n` velas de 1 min ya CERRADAS."""
    rows, end = [], None
    while len(rows) < n:
        params = {"symbol": "BTCUSDT", "interval": "1m", "limit": min(1000, n)}
        if end:
            params["endTime"] = end
        batch = get_json(BINANCE, params)
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
                                     "trades", "taker_buy_base", "tbq", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df[~df.index.duplicated()].sort_index()
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]  # fuera la vela en curso
    return df[["open", "high", "low", "close", "volume", "taker_buy_base"]].astype(float)


def best_ask(token_id: str) -> tuple[float, float]:
    book = get_json(f"{CLOB}/book", {"token_id": token_id})
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
    if not asks:
        return np.nan, 0.0
    price = min(p for p, _ in asks)
    return price, sum(s for p, s in asks if p == price)


def market(start_ts: int) -> dict | None:
    evs = get_json(f"{GAMMA}/events", {"slug": f"btc-updown-5m-{start_ts}"})
    return parse_event(evs[0]) if evs else None


def decide(p_up: float, ask_up: float, ask_dn: float, margin: float, fee_rate: float = 0.07):
    edge_up = p_up - ask_up - fee_per_share(ask_up, fee_rate) if np.isfinite(ask_up) else -np.inf
    edge_dn = (1 - p_up) - ask_dn - fee_per_share(ask_dn, fee_rate) if np.isfinite(ask_dn) else -np.inf
    if max(edge_up, edge_dn) <= margin:
        return None, None, max(edge_up, edge_dn)
    return ("Up", ask_up, edge_up) if edge_up >= edge_dn else ("Down", ask_dn, edge_dn)


def _append(row: dict):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def settle():
    """Rellena resultado y P&L de las apuestas ya resueltas."""
    if not LOG.exists():
        return
    df = pd.read_csv(LOG)
    todo = df["outcome_up"].isna() & (pd.to_datetime(df["start_utc"]) + pd.Timedelta(minutes=7) < pd.Timestamp.now(tz="UTC"))
    for i in df.index[todo]:
        info = market(int(pd.Timestamp(df.at[i, "start_utc"]).timestamp()))
        if not info or info["outcome_up"] is None:
            continue
        df.at[i, "outcome_up"] = info["outcome_up"]
        if isinstance(df.at[i, "side"], str):
            won = (df.at[i, "side"] == "Up") == bool(info["outcome_up"])
            shares, price = df.at[i, "shares"], df.at[i, "price"]
            df.at[i, "won"] = won
            df.at[i, "pnl"] = (shares if won else 0.0) - shares * price - shares * fee_per_share(price)
    df.to_csv(LOG, index=False)


def run(model_name: str, minutes: float, stake: float, margin: float):
    booster, meta = load_model(model_name)
    deadline = time.time() + minutes * 60
    print(f"Bot en simulación con {model_name} (entrenado hasta {meta.get('trained_until')}), apuesta {stake} $, margen {margin}")
    while time.time() < deadline:
        now = time.time()
        start_ts = int(now // 300 + 1) * 300
        info = market(start_ts)  # se prepara antes de que empiece
        history = recent_klines()  # historial hasta la última vela cerrada
        time.sleep(max(0.0, start_ts + 1.0 - time.time()))  # 1 s después del inicio: la vela s-1 ya está cerrada
        t_dec = time.time()
        k = pd.concat([history, recent_klines(n=5)])
        k = k[~k.index.duplicated(keep="last")].sort_index()
        f = features_1m(k)
        x = f.loc[[pd.Timestamp(start_ts - 60, unit="s", tz="UTC")], FEATURES] if pd.Timestamp(start_ts - 60, unit="s", tz="UTC") in f.index else None
        row = dict.fromkeys(FIELDS, "")
        row.update(start_utc=datetime.fromtimestamp(start_ts, tz=UTC).isoformat(), slug=f"btc-updown-5m-{start_ts}")
        if info is None or x is None or x.isna().any(axis=1).iloc[0]:
            print(f"{row['start_utc']}: sin mercado o sin datos")
            _append(row)
            continue
        p_up = float(booster.predict(x.to_numpy())[0])
        (a_up, s_up), (a_dn, s_dn) = best_ask(info["up_token"]), best_ask(info["down_token"])
        side, price, edge = decide(p_up, a_up, a_dn, margin)
        row.update(p_up=round(p_up, 4), ask_up=a_up, ask_up_size=s_up, ask_down=a_dn, ask_down_size=s_dn,
                   exp_edge=round(edge, 4), decided_at_s=round(time.time() - start_ts, 2))
        if side:
            size_ok = (s_up if side == "Up" else s_dn) * price >= stake
            if size_ok:
                row.update(side=side, price=price, shares=round(stake / price, 4))
        print(f"{row['start_utc']}: p_up={p_up:.3f} ask_up={a_up} ask_down={a_dn} -> {row['side'] or 'no apuesta'} "
              f"(decidido en {time.time() - t_dec:.1f} s)")
        _append(row)
        settle()
    settle()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="btc5m_lgbm")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--stake", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args()
    run(a.model, a.minutes, a.stake, a.margin)
