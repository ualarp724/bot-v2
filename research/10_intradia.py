"""Estudio intradía (5m / 15m) del perpetuo de BTC frente a la configuración de 4h, con walk-forward y costes de Kraken.

Uso (desde la raíz del proyecto):
    python research/10_intradia.py --download            # baja velas de 1 min de Binance (público) y genera el informe
    python research/10_intradia.py                        # reutiliza lo ya descargado en data/raw/
    python research/10_intradia.py --synthetic --out /tmp/prueba.md   # PRUEBA DE LA TUBERÍA con un paseo aleatorio (no es BTC)

Protocolo (todo fuera de muestra):
  - Los tramos de test son meses consecutivos desde --first-test; cada uno solo usa lo decidido/aprendido ANTES.
  - Parámetros fijos: no se ajusta nada. Walk-forward: el parámetro se elige en la ventana anterior (o el modelo se
    reentrena con purga) y se evalúa en el mes siguiente. Se simula UNA vez, de forma continua, todo el periodo OOS.
  - Las mismas reglas de costes y de riesgo para 4h, 15m y 5m (riesgo 0,5 % por operación, 5x máximo).
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phoenix.intraday.data import load_1m, resample, synthetic_1m  # noqa: E402
from phoenix.intraday.engine import Costs, Rules, simulate  # noqa: E402
from phoenix.intraday.features import atr, intraday_features  # noqa: E402
from phoenix.intraday.metrics import summarize  # noqa: E402
from phoenix.intraday.strategies import BreakoutParams, breakout_signals  # noqa: E402
from phoenix.intraday.walkforward import make_folds, ml_probabilities, ml_signals, tune_breakout  # noqa: E402

HERE = Path(__file__).resolve().parent
COSTS = Costs()  # taker 0,05 %, maker 0,02 %, deslizamiento 0,02 %, funding de largos
RISK, LEV = 0.005, 5.0
INTRADAY = Rules(
    risk_frac=RISK, max_leverage=LEV, max_open=1, max_trades_per_day=6, daily_loss_limit=0.02,
    entry_until_min=22 * 60, flat_at_min=23 * 60 + 45,
)  # fmt: skip
OVERNIGHT = Rules(risk_frac=RISK, max_leverage=LEV, max_open=1, max_trades_per_day=None, daily_loss_limit=None)
ROUND_TRIP = 2 * (COSTS.taker + COSTS.slippage)  # coste mínimo de una operación completa a mercado: 0,14 %


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="descarga las velas de 1 minuto que falten")
    ap.add_argument("--synthetic", action="store_true", help="paseo aleatorio sin ventaja: SOLO valida la tubería")
    ap.add_argument("--synthetic-days", type=int, default=520)
    ap.add_argument("--start", default="2023-01", help="primer mes de datos (AAAA-MM)")
    ap.add_argument("--first-test", default="2023-07-01", help="inicio del primer tramo fuera de muestra")
    ap.add_argument("--train-months", type=int, default=6)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def load(a) -> pd.DataFrame:
    if a.synthetic:
        return synthetic_1m(days=a.synthetic_days, start=a.start + "-01")
    try:
        df = load_1m(a.start, download=a.download)
    except (ValueError, FileNotFoundError, OSError) as e:
        sys.exit(
            f"No hay velas de 1 minuto en data/raw/ ({e}).\n"
            "Ejecuta con --download desde una red que alcance data.binance.vision. En el entorno en la nube esa\n"
            "máquina está bloqueada por la política de red (403): añádela a 'Allowed domains' o ejecuta esto en local."
        )
    if df.empty:
        sys.exit("Datos vacíos: revisa la descarga (--download).")
    return df


def params_for(minutes: int, **kw) -> BreakoutParams:
    """Ruptura de ~8 h de canal, salida a ~4 h, stop 2 ATR y filtro de tendencia de ~1 día, medidos en velas."""
    bph = 60 / minutes
    base = dict(n_entry=int(8 * bph), n_exit=int(4 * bph), stop_atr=2.0, trend_bars=int(24 * bph))
    return BreakoutParams(**{**base, **kw})


def grid_for(minutes: int) -> list[BreakoutParams]:
    bph = 60 / minutes
    return [
        BreakoutParams(n_entry=int(h * bph), n_exit=int(h / 2 * bph), stop_atr=k, trend_bars=t)
        for h in (4, 8, 16)
        for k in (1.5, 2.5)
        for t in (None, int(24 * bph))
    ]


def verdict(s: dict) -> str:
    if s["trades"] < 30:
        return "Muestra insuficiente"
    if s["exp_r_lo"] > 0 and s["net"] > 0:
        return "Rentable (IC95 de R > 0)"
    if s["exp_r_hi"] < 0:
        return "Pierde (IC95 de R < 0)"
    return "No concluyente (IC95 incluye 0)"


def f(x, kind="n", d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d" if not (isinstance(x, float) and np.isinf(x)) else "∞"
    return {"p": f"{100 * x:.{d}f} %", "n": f"{x:.{d}f}", "u": f"{x:,.0f} $", "i": f"{x:,.0f}"}[kind]


def table(rows: list[list[str]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main():
    a = parse()
    t0 = time.time()
    df1 = load(a)
    first_test = pd.Timestamp(a.first_test, tz="UTC")
    bars = {m: resample(df1, m) for m in (5, 15, 240)}
    dropped = {m: int(len(df1) // m - len(b)) for m, b in bars.items()}
    folds = make_folds(bars[15].index, first_test, a.train_months, 1)
    print(f"{len(df1):,} velas de 1 min → 5m {len(bars[5]):,} | 15m {len(bars[15]):,} | 4h {len(bars[240]):,}; {len(folds)} tramos OOS")

    results: dict[str, dict] = {}
    wf_info: dict[str, pd.DataFrame] = {}
    keep: dict[str, object] = {}  # resultados completos por si hay que consultarlos

    def run(name, b, sig, rules, costs=COSTS):
        res = simulate(b, sig, costs, rules, i0=b.index.searchsorted(first_test))
        s = summarize(res)
        s["verdict"] = verdict(s)
        results[name] = s
        keep[name] = res
        print(f"  {name:48s} ops {s['trades']:5d} | neto {s['net']:+10.0f} $ | PF {s['profit_factor']:.2f} | Sharpe {s['sharpe']:+.2f}")
        return res

    # ---- 4h: configuración anterior (ruptura 20/10, stop 2 ATR, filtro EMA300), con el MISMO motor y costes ----
    print("4h:")
    old = BreakoutParams(n_entry=20, n_exit=10, stop_atr=2.0, trend_bars=300)
    run("4h · Ruptura 20/10 (config. anterior)", bars[240], breakout_signals(bars[240], old), OVERNIGHT)

    atrs, feats = {}, {}
    for m in (15, 5):
        b = bars[m]
        print(f"{m}m:")
        atrs[m] = atr(b, 14)
        run(f"{m}m · Ruptura (parámetros fijos)", b, breakout_signals(b, params_for(m)), INTRADAY)
        sig, chosen = tune_breakout(b, grid_for(m), folds, COSTS, INTRADAY, min_trades=30)
        wf_info[f"{m}m"] = chosen
        run(f"{m}m · Ruptura walk-forward", b, sig, INTRADAY)
        # LightGBM: probabilidades de cada tramo con el modelo reentrenado con la ventana anterior (con purga)
        horizon = int(3 * 60 / m)  # mantener la posición ~3 h como máximo
        feats[m] = intraday_features(b, m)
        p_up, p_dn, info = ml_probabilities(b, feats[m], folds, horizon, ROUND_TRIP, params={"n_estimators": 200})
        wf_info[f"{m}m LightGBM"] = info
        ml_rules = Rules(**{**INTRADAY.__dict__, "max_hold_bars": horizon})
        for thr in (0.40, 0.45):
            run(f"{m}m · LightGBM (p ≥ {thr:.2f})", b, ml_signals(p_up, p_dn, atrs[m], thr), ml_rules)

    # ---- Filtros (ablación sobre la ruptura de parámetros fijos) ----
    print("Filtros:")
    for m in (15, 5):
        b = bars[m]
        for label, kw in (
            ("+ volatilidad", dict(vol_filter=True, vol_window=int(14 * 24 * 60 / m))),
            ("+ sesión 13-21 UTC", dict(hours=(13, 21))),
            ("+ ambos", dict(vol_filter=True, vol_window=int(14 * 24 * 60 / m), hours=(13, 21))),
        ):
            run(f"{m}m · Ruptura {label}", b, breakout_signals(b, params_for(m, **kw)), INTRADAY)

    # ---- Sensibilidad a costes sobre la ruptura de parámetros fijos ----
    print("Sensibilidad a costes:")
    sens = []
    for m in (15, 5):
        b, sg = bars[m], breakout_signals(bars[m], params_for(m))
        for label, c in (
            ("deslizamiento 0 %", Costs(slippage=0.0)),
            ("deslizamiento 0,02 % (base)", COSTS),
            ("deslizamiento 0,05 %", Costs(slippage=0.0005)),
            ("sin comisión ni deslizamiento", Costs(taker=0.0, maker=0.0, slippage=0.0)),
        ):
            res = simulate(b, sg, c, INTRADAY, i0=b.index.searchsorted(first_test))
            s = summarize(res)
            sens.append([f"{m}m", label, f(s["trades"], "i"), f(s["net"], "u"), f(s["profit_factor"]), f(s["sharpe"])])

    # ---------------------------------------------------------------- informe
    names = list(results)
    main_rows = [n for n in names if "+" not in n.split("·")[1]]
    mkt = "SINTÉTICO (paseo aleatorio sin ventaja)" if a.synthetic else "BTCUSDT spot de Binance (proxy del perpetuo de Kraken)"
    oos_start, oos_end = bars[15].index[bars[15].index.searchsorted(first_test)], bars[15].index[-1]
    md = []
    if a.synthetic:
        md += [
            "> **⚠ PRUEBA DE LA TUBERÍA CON DATOS SINTÉTICOS.** Los datos son un paseo aleatorio sin ventaja: cualquier resultado",
            "> positivo aquí es ruido y NO dice nada de BTC. Sirve para comprobar que el motor, el walk-forward y las métricas",
            "> funcionan y como MODELO NULO: lo que una estrategia sin ventaja pierde solo por costes.\n",
        ]
    md += [
        "# Intradía (5m/15m) frente a 4h: backtest fuera de muestra con costes de Kraken Futures\n",
        f"Generado por `research/10_intradia.py` el {pd.Timestamp.now(tz='UTC'):%d/%m/%Y}. Datos: **{mkt}**, "
        f"{len(df1):,} velas de 1 minuto ({df1.index[0]:%d/%m/%Y} – {df1.index[-1]:%d/%m/%Y}). "
        f"**Periodo fuera de muestra: {oos_start:%d/%m/%Y} – {oos_end:%d/%m/%Y}** ({len(folds)} tramos mensuales; "
        f"entrenamiento/selección con los {a.train_months} meses anteriores). "
        f"Velas incompletas descartadas: 5m {dropped[5]}, 15m {dropped[15]}, 4h {dropped[240]}.\n",
        "## Supuestos\n",
        f"- **Costes:** taker {COSTS.taker:.2%} (entradas, stops y cierres forzados), maker {COSTS.maker:.2%} (objetivos, solo si el "
        f"precio los atraviesa), deslizamiento {COSTS.slippage:.2%} en cada operación a mercado y funding de largos "
        f"{COSTS.funding_long_per_hour:.5%}/h. Coste mínimo de una operación completa a mercado: **{ROUND_TRIP:.2%} del nocional**.",
        f"- **Riesgo (igual en todos los marcos):** {RISK:.1%} del equity por operación (pérdida en el stop), apalancamiento máximo {LEV:.0f}x, "
        "capital inicial 10.000 $. La configuración anterior de 4h se vuelve a ejecutar con ESTE motor y ESTE riesgo para que "
        "la comparación sea justa; **no** es la 'lotería' de 100→500 € de `research/08` (30 % de riesgo por operación), que no "
        "admite Sharpe/PF comparables.",
        "- **Reglas intradía (5m/15m):** 1 posición, máximo 6 operaciones/día, sin entradas desde las 22:00 UTC, **cierre obligatorio "
        "a las 23:45 UTC** (nunca se cruza la medianoche) y parada del día si el equity cae un 2 % desde el inicio del día UTC (se "
        "evalúa al cierre de cada vela; la pérdida del día puede rebasar el límite en lo que pierda una vela). 4h: sin cierre ni límite diario.",
        "- **Ejecución:** la señal se decide al cierre de la vela y se ejecuta a la apertura de la siguiente; si stop y objetivo caen en la "
        "misma vela se asume el stop; un hueco más allá del stop se ejecuta a la apertura.",
        f"- **Multiplicidad:** se han evaluado {len(names)} configuraciones y {len(grid_for(15))} combinaciones por tramo en el walk-forward. "
        f"Con un 5 % de significación cabe esperar ~{0.05 * len(names):.1f} 'positivos' por azar: léase cada fila con ese descuento.\n",
    ]

    def stat(n):
        return results[n]

    md += ["## 1. Comparativa (todo fuera de muestra, costes descontados)\n"]
    head = ["Configuración", "Ops", "Ops/día", "Ops/semana", "Win rate", "R:R medio", "Profit Factor", "Expectativa/op", "Expectativa (R) [IC95]"]
    rows = []
    for n in main_rows:
        s = stat(n)
        rows.append([n, f(s["trades"], "i"), f(s["trades_per_day"]), f(s["trades_per_week"], d=1), f(s["win_rate"], "p", 1),
                     f(s["rr"]), f(s["profit_factor"]),
                     f"{f(s['exp_usd'])} $ ({f(s['exp_pct'], 'p', 3)})" if s["trades"] else "n/d",
                     f"{f(s['exp_r'], d=3)} [{f(s['exp_r_lo'], d=3)}; {f(s['exp_r_hi'], d=3)}]"])  # fmt: skip
    md += [table(rows, head), ""]
    head = ["Configuración", "Retorno total", "Sharpe", "Sortino", "t (diario)", "MDD", "MDD ($)", "DD medio (días)", "DD máx. (días)", "Días parados", "Veredicto"]
    rows = []
    for n in main_rows:
        s = stat(n)
        rows.append([n, f(s["total_return"], "p", 1), f(s["sharpe"]), f(s["sortino"]), f(s["t_daily"]), f(s["mdd_pct"], "p", 1),
                     f(s["mdd_usd"], "u"), f(s["dd_avg_days"], d=1), f(s["dd_max_days"], d=0), f(s["halted_days"], "i"), s["verdict"]])  # fmt: skip
    md += [table(rows, head), "",
           "*Sharpe y Sortino anualizados con retornos diarios del equity (365 días, el perpetuo cotiza 24/7), comisiones y "
           "deslizamiento ya descontados; Sortino con desviación a la baja respecto a 0. MDD sobre el equity a mercado de cada vela. "
           "Duración del drawdown: de un máximo a su recuperación (o al final del periodo) para episodios de profundidad ≥ 0,5 %.*\n"]

    md += ["## 2. Anatomía de los costes: ¿se lleva el broker el retorno?\n"]
    head = ["Configuración", "P&L bruto", "Comisiones", "Deslizamiento", "Funding", "Costes totales", "P&L neto", "Costes / bruto",
            "Bruto/op (pb)", "Coste/op (pb)"]
    rows = []
    for n in main_rows:
        s = stat(n)
        co = f(s["cost_over_gross"], "p", 0) if np.isfinite(s["cost_over_gross"]) else ("sin bruto positivo" if s["gross"] <= 0 else "n/d")
        rows.append([n, f(s["gross"], "u"), f(s["fees"], "u"), f(s["slippage"], "u"), f(s["funding"], "u"), f(s["costs"], "u"),
                     f(s["net"], "u"), co, f(s["gross_bps"], d=1), f(s["cost_bps"], d=1)])  # fmt: skip
    md += [table(rows, head), "",
           "*'Bruto/op' y 'Coste/op' en puntos básicos del nocional (1 pb = 0,01 %). Una estrategia solo gana neto si su ventaja bruta "
           "por operación supera el coste por operación.*\n"]

    md += ["## 3. Filtros adicionales (ablación sobre la ruptura de parámetros fijos)\n",
           "Cada fila añade UN filtro a la configuración fija del mismo marco; no se ha elegido ninguno mirando el resultado OOS.\n"]
    head = ["Configuración", "Ops", "Win rate", "Profit Factor", "Exp. (R) [IC95]", "Sharpe", "MDD", "P&L neto", "Veredicto"]
    rows = []
    for n in names:
        if "·" in n and (("+" in n.split("·")[1]) or "parámetros fijos" in n):
            s = stat(n)
            rows.append([n, f(s["trades"], "i"), f(s["win_rate"], "p", 1), f(s["profit_factor"]),
                         f"{f(s['exp_r'], d=3)} [{f(s['exp_r_lo'], d=3)}; {f(s['exp_r_hi'], d=3)}]", f(s["sharpe"]),
                         f(s["mdd_pct"], "p", 1), f(s["net"], "u"), s["verdict"]])  # fmt: skip
    md += [table(rows, head), ""]

    md += ["## 4. Sensibilidad a los costes (ruptura de parámetros fijos)\n",
           table(sens, ["Marco", "Supuesto", "Ops", "P&L neto", "Profit Factor", "Sharpe"]),
           "\n*Si la fila 'sin comisión ni deslizamiento' tampoco gana, no hay ventaja bruta que los costes se estén comiendo.*\n"]

    md += ["## 5. Walk-forward: qué se eligió en cada tramo\n"]
    for k in ("15m", "5m"):
        ch = wf_info[k]
        used = ch["params"].notna()
        cnt = Counter(f"n_entry={p.n_entry}, stop={p.stop_atr} ATR, tendencia={'sí' if p.trend_bars else 'no'}" for p in ch["params"][used])
        md += [f"- **{k}:** el walk-forward operó en {int(used.sum())} de {len(ch)} tramos (en el resto ninguna configuración tuvo "
               f"t > 0 con ≥ 30 operaciones en el entrenamiento y se quedó en efectivo). "
               + ("Elegidas: " + "; ".join(f"{c}× [{p}]" for p, c in cnt.most_common()) + "." if cnt else "")]
    for k in ("15m LightGBM", "5m LightGBM"):
        md += [f"- **{k}:** modelo entrenado en {int(wf_info[k]['trained'].sum())} de {len(wf_info[k])} tramos."]
    md += ["", "## 6. Diagnóstico\n", "*(Se rellena a mano al leer los resultados: ver el mensaje de la sesión.)*\n",
           "## Límites de este estudio\n",
           "- Datos de Binance spot (BTCUSDT) como proxy del perpetuo de Kraken: se ignoran la base y las diferencias de liquidez.",
           "- Deslizamiento constante (0,02 %): en movimientos bruscos o con poca liquidez real será mayor; no hay libro de órdenes ni cola.",
           "- Los objetivos limit se dan por ejecutados solo si el precio los atraviesa, pero sin modelar la prioridad en la cola.",
           "- El resultado depende del régimen del periodo OOS; un solo periodo no garantiza nada sobre el siguiente.\n",
           f"Tiempo de cálculo: {time.time() - t0:.0f} s."]
    out = Path(a.out) if a.out else HERE / ("10_intradia_SINTETICO.md" if a.synthetic else "10_intradia.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nInforme escrito en {out}")


if __name__ == "__main__":
    main()
