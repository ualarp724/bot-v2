"""Apuestas simuladas en los mercados BTC 5m a precios REALES de Polymarket.

Para cada mercado:
1. El modelo da p = probabilidad de "Up" con datos hasta el momento de decidir.
2. Precio de compra de cada lado = VWAP de las compras reales de takers en ese lado dentro de
   la ventana de ejecución [inicio + fill_from_s, inicio + fill_to_s]. Si nadie compró ese lado
   en esa ventana, no hay apuesta (no se inventan precios).
3. Comisión de taker por acción: fee_rate * precio * (1 - precio) (0,07 en cripto).
4. Se apuesta al lado con más ventaja esperada si  p_lado - precio - comisión > margen.
5. Cada apuesta es de `stake` USD: acciones = stake / precio. Si gana, cada acción paga 1 USD.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fee_per_share(price: float | np.ndarray, rate: float = 0.07):
    return rate * price * (1 - price)


def fill_prices(trades: pd.DataFrame, markets: pd.DataFrame, fill_from_s: int, fill_to_s: int) -> pd.DataFrame:
    """VWAP de compras de takers por mercado y lado dentro de la ventana de ejecución."""
    t = trades.merge(markets[["slug", "start_ts"]], on="slug")
    rel = t["ts"] - t["start_ts"]
    t = t[(t["side"] == "BUY") & (rel >= fill_from_s) & (rel <= fill_to_s)]
    if t.empty:
        return pd.DataFrame(index=markets["slug"], columns=["px_up", "px_down"], dtype=float)
    t = t.assign(notional=t["price"] * t["size"])
    g = t.groupby(["slug", "outcome"])[["notional", "size"]].sum()
    vwap = (g["notional"] / g["size"]).unstack("outcome")
    out = pd.DataFrame(index=markets["slug"])
    out["px_up"] = vwap.get("Up")
    out["px_down"] = vwap.get("Down")
    return out


FILL_WINDOWS = [(-60, -30), (-55, -5), (-30, -1), (0, 2), (1, 5), (1, 10), (5, 15)]


def fill_summary(trades: pd.DataFrame, markets: pd.DataFrame, windows=FILL_WINDOWS) -> pd.DataFrame:
    """Tabla compacta por mercado con el VWAP de compra de cada lado en varias ventanas de ejecución.

    Sustituye a las operaciones crudas (cientos de MB) para repetir los backtests en cualquier equipo.
    """
    out = markets.set_index("slug")[["start_ts", "outcome_up", "price_to_beat", "final_price", "volume"]].copy()
    for a, b in windows:
        px = fill_prices(trades, markets, a, b)
        out[f"px_up_{a}_{b}"] = px["px_up"]
        out[f"px_down_{a}_{b}"] = px["px_down"]
    return out.reset_index()


def prices_from_summary(summary_df: pd.DataFrame, window: tuple[int, int]) -> pd.DataFrame:
    a, b = window
    return summary_df.set_index("slug")[[f"px_up_{a}_{b}", f"px_down_{a}_{b}"]].set_axis(["px_up", "px_down"], axis=1)


def simulate(
    markets: pd.DataFrame,
    p_up: pd.Series,
    prices: pd.DataFrame,
    *,
    margin: float = 0.02,
    stake: float = 5.0,
    fee_rate: float = 0.07,
) -> pd.DataFrame:
    """Una fila por apuesta. `markets` necesita slug, start_ts, outcome_up; `p_up` y `prices` indexados por slug."""
    df = markets.set_index("slug")[["start_ts", "outcome_up"]].join(p_up.rename("p_up")).join(prices)
    df = df.dropna(subset=["p_up", "outcome_up"])
    edge_up = df["p_up"] - df["px_up"] - fee_per_share(df["px_up"], fee_rate)
    edge_dn = (1 - df["p_up"]) - df["px_down"] - fee_per_share(df["px_down"], fee_rate)
    edge_up, edge_dn = edge_up.fillna(-np.inf), edge_dn.fillna(-np.inf)
    side_up = edge_up >= edge_dn
    edge = np.where(side_up, edge_up, edge_dn)
    bet = df[edge > margin].copy()
    su = side_up[edge > margin]
    bet["side"] = np.where(su, "Up", "Down")
    bet["price"] = np.where(su, bet["px_up"], bet["px_down"])
    bet["p_side"] = np.where(su, bet["p_up"], 1 - bet["p_up"])
    bet["exp_edge"] = edge[edge > margin]
    bet["won"] = np.where(su, bet["outcome_up"].astype(bool), ~bet["outcome_up"].astype(bool))
    shares = stake / bet["price"]
    fee = shares * fee_per_share(bet["price"], fee_rate)
    bet["pnl"] = np.where(bet["won"], shares * 1.0, 0.0) - stake - fee
    bet["time"] = pd.to_datetime(bet["start_ts"], unit="s", utc=True)
    return bet.reset_index()


def summary(bets: pd.DataFrame, stake: float = 5.0) -> dict:
    if bets.empty:
        return {"bets": 0}
    days = max((bets["time"].max() - bets["time"].min()).days, 1)
    daily = bets.groupby(bets["time"].dt.date)["pnl"].sum()
    return {
        "bets": len(bets),
        "bets_per_day": len(bets) / days,
        "win_rate_pct": 100 * bets["won"].mean(),
        "avg_price": bets["price"].mean(),
        "avg_expected_edge": bets["exp_edge"].mean(),
        "pnl_usd": bets["pnl"].sum(),
        "roi_per_bet_pct": 100 * bets["pnl"].sum() / (stake * len(bets)),
        "days_positive_pct": 100 * (daily > 0).mean(),
        "worst_day_usd": daily.min(),
        "max_dd_usd": float((daily.cumsum().cummax() - daily.cumsum()).max()),
    }
