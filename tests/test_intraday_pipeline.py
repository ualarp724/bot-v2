import itertools

import numpy as np
import pandas as pd
import pytest

from phoenix.checks import lookahead_violations
from phoenix.intraday.data import resample, synthetic_1m
from phoenix.intraday.engine import Costs, Result, Rules, Signals, simulate
from phoenix.intraday.features import atr, intraday_features
from phoenix.intraday.metrics import drawdown_episodes, sharpe, sortino, summarize
from phoenix.intraday.strategies import BreakoutParams, breakout_signals
from phoenix.intraday.walkforward import (
    make_folds,
    make_labels,
    restrict,
    train_slice,
    tune_breakout,
    walk_forward_ml,
)


@pytest.fixture(scope="module")
def bars15():
    return resample(synthetic_1m(days=170, seed=3, start="2024-01-01"), 15)


# ---------------------------------------------------------------- datos
def test_resample_keeps_only_complete_bars_and_aggregates_ohlcv():
    idx = pd.date_range("2026-01-05 10:00", periods=45, freq="1min", tz="UTC")
    one = pd.DataFrame(
        {"open": np.arange(45) + 100.0, "high": np.arange(45) + 101.0, "low": np.arange(45) + 99.0,
         "close": np.arange(45) + 100.5, "volume": 1.0, "taker_buy_base": 0.4},
        index=idx,
    )  # fmt: skip
    one = one.drop(one.index[20])  # falta el minuto 10:20: la vela de 10:15 queda incompleta
    out = resample(one, 15)
    assert list(out.index) == [pd.Timestamp("2026-01-05 10:00", tz="UTC"), pd.Timestamp("2026-01-05 10:30", tz="UTC")]
    b = out.iloc[0]
    assert (b.open, b.high, b.low, b.close, b.volume, b.taker_buy_base) == (100.0, 115.0, 99.0, 114.5, 15.0, 6.0)


def test_synthetic_data_is_reproducible_and_has_no_drift():
    a, b = synthetic_1m(days=20, seed=1), synthetic_1m(days=20, seed=1)
    assert a.equals(b) and (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    r = np.log(a["close"]).diff().dropna()
    assert abs(r.mean()) < 5 * r.std() / np.sqrt(len(r))  # sin deriva apreciable


# ---------------------------------------------------------------- ausencia de fugas
def _causal(fn, bars, cuts):
    """fn(bars[:k]) debe coincidir con las k primeras filas de fn(bars). Una fuga de UNA vela solo se nota en la última
    fila de cada recorte, por eso hacen falta muchos puntos de corte consecutivos."""
    full = np.asarray(fn(bars), float)
    for k in cuts:
        part = np.asarray(fn(bars.iloc[:k]), float)
        if not np.allclose(full[:k], part, equal_nan=True):
            raise AssertionError(f"fuga detectada con corte en {k}")


def test_features_are_causal(bars15):
    _causal(lambda b: intraday_features(b, 15), bars15, cuts=range(1500, 1560))
    assert lookahead_violations(lambda b: intraday_features(b, 15), bars15, cut_points=[500, 1000, 2000]) == []


def _signal_matrix(b, p):
    s = breakout_signals(b, p)
    return np.column_stack([s.side, s.stop_dist, s.exit_long, s.exit_short])


def test_the_causality_check_detects_deliberate_leaks(bars15):
    """El detector tiene que saltar con una fuga de una sola vela, tanto en features como en señales."""
    leaky_feat = lambda b: intraday_features(b, 15).assign(oops=b["close"].shift(-1) / b["close"] - 1)  # noqa: E731
    with pytest.raises(AssertionError, match="fuga"):
        _causal(leaky_feat, bars15, cuts=range(1500, 1560))

    def leaky_signals(b):
        m = _signal_matrix(b, BreakoutParams())
        m[:, 0] = np.where(b["close"].shift(-1).to_numpy() > b["close"].to_numpy() * 1.001, 1, m[:, 0])
        return m

    with pytest.raises(AssertionError, match="fuga"):
        _causal(leaky_signals, bars15, cuts=range(1000, 1300))


@pytest.mark.parametrize(
    "p",
    [
        BreakoutParams(),
        BreakoutParams(n_entry=10, trend_bars=96, tp_r=2.0),
        BreakoutParams(vol_filter=True, vol_window=300, hours=(8, 20)),
    ],
)
def test_breakout_signals_are_causal(bars15, p):
    _causal(lambda b: _signal_matrix(b, p), bars15, cuts=range(1000, 1300))


def test_breakout_logic_and_filters():
    n = 40
    idx = pd.date_range("2026-01-05 00:00", periods=n, freq="15min", tz="UTC")
    c = np.full(n, 100.0)
    c[30] = 105.0  # rompe el máximo de las 5 velas previas
    bars = pd.DataFrame(
        {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1.0, "taker_buy_base": 0.5}, index=idx
    )
    p = BreakoutParams(n_entry=5, n_exit=3, stop_atr=2.0, atr_n=5)
    s = breakout_signals(bars, p)
    assert s.side[30] == 1 and s.side[29] == 0 and s.stop_dist[30] == pytest.approx(2.0 * atr(bars, 5).iloc[30])
    assert (
        breakout_signals(bars, BreakoutParams(n_entry=5, atr_n=5, hours=(9, 10))).side[30] == 0
    )  # 07:30 UTC: fuera de franja
    assert breakout_signals(bars, BreakoutParams(n_entry=5, atr_n=5, hours=(7, 8))).side[30] == 1


# ---------------------------------------------------------------- walk-forward
def test_folds_are_contiguous_and_never_overlap_the_training_window(bars15):
    folds = make_folds(bars15.index, pd.Timestamp("2024-02-15", tz="UTC"), train_months=1, test_months=1)
    assert len(folds) >= 2
    for a, b in itertools.pairwise(folds):
        assert a.test_end == b.test_start
    for f in folds:
        assert f.train_start < f.test_start < f.test_end <= bars15.index[-1] + pd.Timedelta("15min")


def test_labels_and_purge_never_touch_the_test_window():
    idx = pd.date_range("2026-01-05", periods=8, freq="15min", tz="UTC")
    o = np.array([100, 100, 100, 100, 100, 100, 100, 100.0])
    c = np.array([100, 100, 103, 100, 100, 100, 100, 100.0])
    bars = pd.DataFrame({"open": o, "high": c + 1, "low": c - 1, "close": c}, index=idx)
    y = make_labels(bars, horizon=2, thr=0.01)
    assert y.iloc[0] == 2  # entra a la apertura 1 (100) y 2 velas después cierra en 103 (+3 %)
    assert y.iloc[1] == 1 and y.iloc[-1] != y.iloc[-1] and y.iloc[-2] != y.iloc[-2]  # sin futuro -> NaN
    sl = train_slice(10, 100, horizon=12)
    assert sl.stop - 1 + 12 <= 100 - 1  # la etiqueta de la última fila de entrenamiento no pisa el test


def _noisy_after(bars, cut, seed=0):
    noisy = bars.copy()
    rng = np.random.default_rng(seed)
    for col in ("open", "high", "low", "close"):
        noisy.iloc[cut:, noisy.columns.get_loc(col)] *= rng.normal(1.0, 0.03, len(noisy) - cut)
    return noisy


_GRID = [BreakoutParams(n_entry=n, stop_atr=k, atr_n=14) for n in (12, 36) for k in (1.5, 2.5)]
_RULES = Rules(daily_loss_limit=0.02, flat_at_min=23 * 60 + 45)


def _scores_equal(a, b, upto):
    return all(
        np.allclose(x, y, equal_nan=True)
        for x, y in zip(a["is_scores"].iloc[:upto], b["is_scores"].iloc[:upto], strict=True)
    )


def test_tuned_walk_forward_is_not_affected_by_the_future_and_is_flat_outside_the_test_windows(bars15):
    folds = make_folds(bars15.index, pd.Timestamp("2024-02-01", tz="UTC"), train_months=1, test_months=1)
    sig, chosen = tune_breakout(bars15, _GRID, folds, Costs(), _RULES, min_trades=5)
    # No vacuo: hay varios tramos, puntuaciones numéricas y alguna señal. (Sobre ruido con costes lo normal es que el
    # walk-forward se abstenga en casi todos los tramos: ninguna configuración tiene t > 0 en entrenamiento.)
    assert len(folds) >= 4 and np.isfinite(np.concatenate(chosen["is_scores"].tolist())).sum() >= 8
    assert chosen["params"].notna().sum() >= 1 and (np.asarray(sig.side) != 0).sum() > 0
    keep = np.zeros(len(bars15), bool)
    for f in folds:
        keep[bars15.index.searchsorted(f.test_start) : bars15.index.searchsorted(f.test_end)] = True
    assert (np.asarray(sig.side)[~keep] == 0).all() and len(chosen) == len(folds)
    # Se altera todo lo posterior al inicio del test del último tramo: las puntuaciones de entrenamiento y las señales
    # anteriores NO pueden cambiar. Se comparan las PUNTUACIONES (el ganador podría coincidir por casualidad).
    cut = bars15.index.searchsorted(folds[-1].test_start)
    sig2, chosen2 = tune_breakout(_noisy_after(bars15, cut), _GRID, folds, Costs(), _RULES, min_trades=5)
    assert _scores_equal(chosen, chosen2, len(folds))
    assert np.array_equal(np.asarray(sig.side)[:cut], np.asarray(sig2.side)[:cut])


def test_the_walk_forward_integrity_check_detects_a_selection_that_peeks_into_the_test_window(bars15, monkeypatch):
    import phoenix.intraday.walkforward as wf

    folds = make_folds(bars15.index, pd.Timestamp("2024-02-01", tz="UTC"), train_months=1, test_months=1)
    real = wf.simulate

    def peeking(bars, sg, costs, rules, i0=0, i1=None):  # el "entrenamiento" se alarga 10 días dentro del test
        return real(bars, sg, costs, rules, i0=i0, i1=min(i1 + 960, len(bars)))

    cut = bars15.index.searchsorted(folds[-1].test_start)
    monkeypatch.setattr(wf, "simulate", peeking)
    _, a = wf.tune_breakout(bars15, _GRID, folds, Costs(), _RULES, min_trades=5)
    _, b = wf.tune_breakout(_noisy_after(bars15, cut), _GRID, folds, Costs(), _RULES, min_trades=5)
    assert not _scores_equal(a, b, len(folds))  # con la fuga, alterar el futuro cambia lo "aprendido": se detecta


def test_restrict_zeroes_signals_outside_the_test_windows(bars15):
    folds = make_folds(bars15.index, pd.Timestamp("2024-02-15", tz="UTC"), 1, 1)
    full = breakout_signals(bars15, BreakoutParams(n_entry=12))
    r = restrict(full, folds, bars15.index)
    first = bars15.index.searchsorted(folds[0].test_start)
    assert (r.side[:first] == 0).all() and (r.side[first:] == full.side[first:]).all()


def test_ml_training_labels_are_identical_when_the_future_is_cut_off_at_each_test_start(bars15):
    """Con la purga, las etiquetas de entrenamiento de cada tramo no dependen de NINGUNA vela del tramo de test:
    recortar los datos en su inicio no cambia ninguna. Sin purga, las últimas etiquetas quedarían sin futuro (NaN)."""
    horizon = 8
    folds = make_folds(bars15.index, pd.Timestamp("2024-02-15", tz="UTC"), train_months=1, test_months=1)
    y_full = make_labels(bars15, horizon, 0.0014)
    for f in folds:
        a, b = bars15.index.searchsorted(f.train_start), bars15.index.searchsorted(f.test_start)
        sl = train_slice(a, b, horizon)
        y_cut = make_labels(bars15.iloc[:b], horizon, 0.0014)
        assert y_full.iloc[sl].notna().all() and np.array_equal(y_full.iloc[sl], y_cut.iloc[sl])
        no_purge = slice(a, b)  # sin purga: las últimas `horizon` etiquetas miran dentro del test
        assert not np.array_equal(y_full.iloc[no_purge], y_cut.iloc[no_purge], equal_nan=True)


def test_ml_walk_forward_trains_per_fold_and_only_signals_in_test_windows(bars15):
    X = intraday_features(bars15, 15)
    folds = make_folds(bars15.index, pd.Timestamp("2024-02-15", tz="UTC"), train_months=1, test_months=1)
    kw = {"horizon": 8, "cost_thr": 0.0014, "p_thr": 0.4, "params": {"n_estimators": 25, "n_jobs": 1}, "min_rows": 500}
    sig, info = walk_forward_ml(bars15, X, atr(bars15, 14), folds, **kw)
    assert info["trained"].all() and (np.asarray(sig.side) != 0).sum() > 0
    first = bars15.index.searchsorted(folds[0].test_start)
    assert (np.asarray(sig.side)[:first] == 0).all()
    # Alterar lo posterior al test del último tramo no cambia las señales de los tramos anteriores.
    cut = bars15.index.searchsorted(folds[-1].test_start)
    noisy = _noisy_after(bars15, cut, seed=1)
    sig2, _ = walk_forward_ml(noisy, intraday_features(noisy, 15), atr(noisy, 14), folds, **kw)
    assert np.array_equal(np.asarray(sig.side)[:cut], np.asarray(sig2.side)[:cut])


# ---------------------------------------------------------------- métricas
def _result(trades: pd.DataFrame, daily_eq: list[float], capital=10_000.0):
    idx = pd.date_range("2026-01-05", periods=len(daily_eq), freq="1D", tz="UTC")
    return Result(trades, pd.Series(daily_eq, index=idx), capital, 0, {})


def _trades(net, gross=None, costs=0.0):
    net = np.asarray(net, float)
    gross = net + costs if gross is None else np.asarray(gross, float)
    return pd.DataFrame(
        {"net": net, "gross": gross, "fees": costs, "slippage": 0.0, "funding": 0.0, "r": net / 100.0,
         "equity0": 10_000.0, "notional": 10_000.0},
        index=range(len(net)),
    )  # fmt: skip


def test_trade_statistics_match_hand_calculation():
    # Ganadoras 100 y 50, perdedoras -50 y -30: win rate 50 %, PF = 150/80, R:R = 75/40.
    r = summarize(_result(_trades([100, 50, -50, -30], costs=10.0), [10_000, 10_050, 10_100, 10_070]))
    assert r["trades"] == 4 and r["win_rate"] == 0.5
    assert r["profit_factor"] == pytest.approx(150 / 80) and r["rr"] == pytest.approx(75 / 40)
    assert r["exp_usd"] == pytest.approx(17.5) and r["exp_pct"] == pytest.approx(17.5 / 10_000)
    assert r["gross"] == pytest.approx(70 + 40) and r["costs"] == pytest.approx(40.0)
    assert r["cost_over_gross"] == pytest.approx(40.0 / 110.0)
    assert r["trades_per_day"] == pytest.approx(4 / 3) and r["trades_per_week"] == pytest.approx(28 / 3)
    gross_neg = summarize(_result(_trades([-10, -20], gross=[-5, -10], costs=0.0), [10_000, 9_990, 9_970]))
    assert np.isnan(gross_neg["cost_over_gross"])  # sin beneficio bruto no hay "reparto" que medir


def test_sharpe_sortino_and_drawdown_episodes():
    r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(365))
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert sortino(r) == pytest.approx(r.mean() / down * np.sqrt(365))
    eq = pd.Series(
        [100, 110, 99, 105, 120, 90, 100.0], index=pd.date_range("2026-01-01", periods=7, freq="1D", tz="UTC")
    )
    eps = drawdown_episodes(eq)
    assert eps["days"].tolist() == [3, 2] and eps["recovered"].tolist() == [True, False]
    assert eps["depth"].tolist() == pytest.approx([-0.1, -0.25])
    s = summarize(_result(_trades([10.0]), list(eq * 100), capital=10_000.0))
    assert (
        s["mdd_pct"] == pytest.approx(-0.25)
        and s["dd_avg_days"] == pytest.approx(2.5)
        and s["mdd_usd"] == pytest.approx(3_000.0)
    )


def test_empty_trades_do_not_break_the_summary():
    s = summarize(_result(_trades([]), [10_000, 10_000, 10_000]))
    assert s["trades"] == 0 and np.isnan(s["win_rate"]) and s["costs"] == 0.0


def test_end_to_end_on_synthetic_data_net_equals_gross_minus_costs(bars15):
    sig = breakout_signals(bars15, BreakoutParams(n_entry=20, trend_bars=96))
    rules = Rules(daily_loss_limit=0.02, flat_at_min=23 * 60 + 45, entry_until_min=22 * 60, max_trades_per_day=4)
    res = simulate(bars15, sig, Costs(), rules)
    s = summarize(res)
    assert s["trades"] > 20 and s["net"] == pytest.approx(s["gross"] - s["costs"], abs=1e-6)
    assert (
        res.trades["exit_time"].dt.date.eq(res.trades["entry_time"].dt.date).all()
    )  # ninguna operación cruza la medianoche
    assert isinstance(sig, Signals)
