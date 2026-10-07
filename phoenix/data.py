"""Carga, validación y limpieza de velas exportadas de MT5.

Convenciones:
- El índice es la hora de APERTURA de la vela, en UTC.
- Columnas: open, high, low, close, tick_volume, spread_points.
- La hora del servidor de Vantage es la hora de Nueva York + 7 h (GMT+2 en
  invierno, GMT+3 en verano), así que se pasa a UTC a través de America/New_York.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from phoenix.settings import Settings

NY_TZ = "America/New_York"
COLUMNS = ["open", "high", "low", "close", "tick_volume", "spread_points"]


def read_mt5_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV exportado por MT5 (separado por tabuladores). Índice = hora del servidor, sin zona."""
    df = pd.read_csv(path, sep="\t")
    if len(df.columns) < 2:
        df = pd.read_csv(path, sep=",")
    df.columns = [c.strip("<>").upper() for c in df.columns]
    rename = {
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "TICKVOL": "tick_volume",
        "SPREAD": "spread_points",
    }
    missing = {"DATE", "TIME", *rename} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {sorted(missing)}")
    index = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="%Y.%m.%d %H:%M:%S")
    out = df.rename(columns=rename)[COLUMNS].astype(float)
    out.index = pd.DatetimeIndex(index, name="time_server")
    return out


def server_to_utc(index: pd.DatetimeIndex, offset_from_new_york_hours: int = 7) -> pd.DatetimeIndex:
    """Convierte la hora del servidor (Nueva York + offset) a UTC.

    Las horas ambiguas o inexistentes por el cambio de horario caen en fin de
    semana (mercado cerrado); si apareciera alguna con datos, se lanza un error.
    """
    ny_naive = index - pd.Timedelta(hours=offset_from_new_york_hours)
    ny = ny_naive.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="NaT")
    if ny.isna().any():
        bad = index[ny.isna()]
        raise ValueError(f"Horas ambiguas por cambio de horario con datos: {list(bad[:5])}")
    return ny.tz_convert("UTC").rename("time_utc")


@dataclass
class ValidationReport:
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicates: int
    non_monotonic: int
    bad_ohlc: int
    non_positive: int
    unexpected_gaps: pd.Series = field(repr=False)

    @property
    def ok(self) -> bool:
        return self.duplicates == 0 and self.non_monotonic == 0 and self.bad_ohlc == 0 and self.non_positive == 0


def _is_expected_gap(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> bool:
    """True si el hueco se explica por la pausa diaria (17:00-18:00 NY) o el fin de semana."""
    start_ny, end_ny = start_utc.tz_convert(NY_TZ), end_utc.tz_convert(NY_TZ)
    if (end_ny - start_ny) > pd.Timedelta(days=4):
        return False
    # Fin de semana: empieza viernes y acaba domingo/lunes
    if start_ny.dayofweek == 4 and end_ny.dayofweek in (6, 0):
        return True
    # Pausa diaria: el hueco cubre las 17:00-18:00 de Nueva York del mismo día
    brk_start = start_ny.normalize() + pd.Timedelta(hours=17)
    brk_end = brk_start + pd.Timedelta(hours=1)
    return start_ny <= brk_start and end_ny >= brk_end and (end_ny - start_ny) <= pd.Timedelta(hours=2)


def validate_bars(df: pd.DataFrame, bar_minutes: int) -> ValidationReport:
    idx = df.index
    step = pd.Timedelta(minutes=bar_minutes)
    diffs = idx.to_series().diff()
    gap_mask = diffs > step
    gaps = diffs[gap_mask]
    unexpected = {}
    for end, gap in gaps.items():
        last_bar_open = end - gap
        if not _is_expected_gap(last_bar_open + step, end):
            unexpected[end] = gap
    return ValidationReport(
        rows=len(df),
        first=idx.min() if len(df) else None,
        last=idx.max() if len(df) else None,
        duplicates=int(idx.duplicated().sum()),
        non_monotonic=int((diffs < pd.Timedelta(0)).sum()),
        bad_ohlc=int(
            (
                (df["high"] < df[["open", "close"]].max(axis=1))
                | (df["low"] > df[["open", "close"]].min(axis=1))
                | (df["high"] < df["low"])
            ).sum()
        ),
        non_positive=int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        unexpected_gaps=pd.Series(unexpected, dtype="timedelta64[ns]"),
    )


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena, quita duplicados (se queda con el primero) y filas con OHLC imposible."""
    out = df.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    ok = (
        (out["high"] >= out[["open", "close"]].max(axis=1))
        & (out["low"] <= out[["open", "close"]].min(axis=1))
        & (out[["open", "high", "low", "close"]] > 0).all(axis=1)
    )
    return out[ok]


def load_bars(path: str | Path, settings: Settings) -> pd.DataFrame:
    """Lee, pasa a UTC y limpia. Resultado listo para calcular features."""
    df = read_mt5_csv(path)
    df.index = server_to_utc(df.index, settings.server_offset_from_new_york_hours)
    return clean_bars(df)


def dev_data(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Periodo de desarrollo (fase 3). Nunca incluye el test."""
    s = settings.splits
    return df.loc[s.dev_start : s.dev_end]


def holdout_data(df: pd.DataFrame, settings: Settings, *, i_know_this_is_the_final_test: bool = False) -> pd.DataFrame:
    """Periodo de test intocable. Solo se abre en la fase 4, una vez."""
    if not i_know_this_is_the_final_test:
        raise PermissionError("El test es intocable hasta la fase 4 (ver el plan).")
    s = settings.splits
    return df.loc[s.test_start : s.test_end]
