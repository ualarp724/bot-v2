"""Métricas del informe: operaciones, rentabilidad, riesgo ajustado, drawdowns y peso de los costes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenix.intraday.engine import Result

DAYS_PER_YEAR = 365  # el perpetuo cotiza 24/7


def daily_returns(eq: pd.Series, capital: float) -> pd.Series:
    d = eq.resample("1D").last().dropna()
    prev = pd.concat([pd.Series([capital]), d.iloc[:-1].reset_index(drop=True)], ignore_index=True)
    return pd.Series(d.to_numpy() / prev.to_numpy() - 1, index=d.index)


def sharpe(r: pd.Series) -> float:
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(DAYS_PER_YEAR)) if len(r) > 2 and s > 0 else float("nan")


def sortino(r: pd.Series) -> float:
    down = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    return float(r.mean() / down * np.sqrt(DAYS_PER_YEAR)) if len(r) > 2 and down > 0 else float("nan")


def drawdown_episodes(daily_eq: pd.Series, min_depth: float = 0.005) -> pd.DataFrame:
    """Episodios de drawdown (de un máximo hasta recuperarlo o hasta el final) con profundidad >= min_depth."""
    peak = daily_eq.cummax()
    under = (daily_eq < peak).to_numpy()
    rows, i, n = [], 0, len(daily_eq)
    while i < n:
        if not under[i]:
            i += 1
            continue
        j = i
        while j < n and under[j]:
            j += 1
        depth = float((daily_eq.iloc[i:j] / peak.iloc[i:j] - 1).min())
        end = daily_eq.index[j] if j < n else daily_eq.index[-1]
        if -depth >= min_depth:
            rows.append((daily_eq.index[i - 1], end, (end - daily_eq.index[i - 1]).days, depth, j < n))
        i = j
    return pd.DataFrame(rows, columns=["peak", "end", "days", "depth", "recovered"])


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(n_boot)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(res: Result) -> dict:
    t, eq, cap = res.trades, res.equity, res.capital
    days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400, 1e-9) if len(eq) else float("nan")
    out: dict = {"trades": len(t), "days": days, "halted_days": res.halted_days}
    d = daily_returns(eq, cap) if len(eq) else pd.Series(dtype=float)
    daily_eq = eq.resample("1D").last().dropna() if len(eq) else eq
    dd = (eq / eq.cummax() - 1) if len(eq) else eq
    eps = drawdown_episodes(daily_eq) if len(daily_eq) else pd.DataFrame(columns=["days"])
    out.update(
        total_return=float(eq.iloc[-1] / cap - 1) if len(eq) else float("nan"),
        sharpe=sharpe(d), sortino=sortino(d),
        t_daily=float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))) if len(d) > 2 and d.std(ddof=1) > 0 else float("nan"),
        mdd_pct=float(dd.min()) if len(dd) else float("nan"),
        mdd_usd=float((eq.cummax() - eq).max()) if len(eq) else float("nan"),
        dd_episodes=len(eps), dd_avg_days=float(eps["days"].mean()) if len(eps) else 0.0,
        dd_max_days=float(eps["days"].max()) if len(eps) else 0.0,
    )  # fmt: skip
    if t.empty:
        out.update(trades_per_day=0.0, trades_per_week=0.0, win_rate=float("nan"), rr=float("nan"),
                   profit_factor=float("nan"), exp_usd=float("nan"), exp_pct=float("nan"), exp_r=float("nan"),
                   exp_r_lo=float("nan"), exp_r_hi=float("nan"), t_trades=float("nan"), gross=0.0, net=0.0, fees=0.0,
                   slippage=0.0, funding=0.0, costs=0.0, cost_over_gross=float("nan"), gross_bps=float("nan"),
                   cost_bps=float("nan"))  # fmt: skip
        return out
    net, gross = t["net"].to_numpy(), t["gross"].to_numpy()
    wins, losses = net[net > 0], net[net <= 0]
    r = t["r"].to_numpy()
    costs = float((t["fees"] + t["slippage"] + t["funding"]).sum())
    lo, hi = bootstrap_mean_ci(r)
    out.update(
        trades_per_day=len(t) / days, trades_per_week=7 * len(t) / days,
        win_rate=float((net > 0).mean()),
        rr=float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        profit_factor=float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else float("inf"),
        exp_usd=float(net.mean()), exp_pct=float((t["net"] / t["equity0"]).mean()), exp_r=float(r.mean()),
        exp_r_lo=lo, exp_r_hi=hi,
        t_trades=float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan"),
        gross=float(gross.sum()), net=float(net.sum()), fees=float(t["fees"].sum()), slippage=float(t["slippage"].sum()),
        funding=float(t["funding"].sum()), costs=costs,
        cost_over_gross=costs / gross.sum() if gross.sum() > 0 else float("nan"),
        gross_bps=float((t["gross"] / t["notional"]).mean() * 1e4),
        cost_bps=float(((t["fees"] + t["slippage"] + t["funding"]) / t["notional"]).mean() * 1e4),
    )  # fmt: skip
    return out
