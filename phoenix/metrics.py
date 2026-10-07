"""Métricas de un backtest. Todo se calcula sobre operaciones cerradas."""
from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(pnl: np.ndarray, capital: float) -> tuple[float, float]:
    """Drawdown máximo de la curva capital + P&L acumulado: (USD, % del pico)."""
    equity = capital + np.concatenate([[0.0], np.cumsum(pnl)])
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    i = int(np.argmax(dd))
    return float(dd[i]), float(100.0 * dd[i] / peak[i]) if peak[i] > 0 else 0.0


def monte_carlo_dd(pnl: np.ndarray, capital: float, n: int = 2000, q: float = 0.95, seed: int = 0) -> float:
    """Drawdown (% ) en el percentil q reordenando las operaciones al azar."""
    if len(pnl) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    dds = [max_drawdown(rng.permutation(pnl), capital)[1] for _ in range(n)]
    return float(np.quantile(dds, q))


def summarize(trades: pd.DataFrame, capital: float) -> dict:
    if trades.empty:
        return {"trades": 0}
    pnl = trades["pnl_usd"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    dd_usd, dd_pct = max_drawdown(pnl, capital)
    daily = trades.groupby(trades["exit_time"].dt.tz_convert("UTC").dt.date)["pnl_usd"].sum()
    days = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(days.date, fill_value=0.0) / capital
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    months = max((trades["exit_time"].max() - trades["entry_time"].min()).days / 30.44, 1e-9)
    return {
        "trades": len(trades),
        "win_rate_pct": 100.0 * len(wins) / len(pnl),
        "avg_r": float(trades["r"].mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "pnl_usd": float(pnl.sum()),
        "return_pct": 100.0 * pnl.sum() / capital,
        "max_dd_usd": dd_usd,
        "max_dd_pct": dd_pct,
        "mc_dd95_pct": monte_carlo_dd(pnl, capital),
        "sharpe_daily_annualized": sharpe,
        "trades_per_month": len(trades) / months,
    }
