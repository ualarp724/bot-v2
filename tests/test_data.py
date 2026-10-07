import pandas as pd
import pytest

from phoenix.data import clean_bars, dev_data, holdout_data, read_mt5_csv, server_to_utc, validate_bars
from phoenix.settings import load_settings

HEADER = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"


def _write(tmp_path, rows):
    p = tmp_path / "bars.csv"
    p.write_text(HEADER + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return p


def test_read_mt5_csv(tmp_path):
    p = _write(tmp_path, [("2025.01.06", "10:00:00", 2600, 2601, 2599, 2600.5, 100, 0, 7)])
    df = read_mt5_csv(p)
    assert list(df.columns) == ["open", "high", "low", "close", "tick_volume", "spread_points"]
    assert df.index[0] == pd.Timestamp("2025-01-06 10:00")
    assert df["spread_points"].iloc[0] == 7


def test_server_to_utc_winter_and_summer():
    # Invierno: servidor GMT+2 -> 10:00 servidor = 08:00 UTC
    # Verano (horario de EE. UU.): servidor GMT+3 -> 10:00 servidor = 07:00 UTC
    idx = pd.DatetimeIndex(["2025-01-15 10:00", "2025-07-15 10:00"])
    utc = server_to_utc(idx, 7)
    assert utc[0] == pd.Timestamp("2025-01-15 08:00", tz="UTC")
    assert utc[1] == pd.Timestamp("2025-07-15 07:00", tz="UTC")


def test_server_to_utc_us_dst_week():
    # 2025-03-10: EE. UU. ya ha cambiado de hora (9 mar) pero Europa no (30 mar).
    # El servidor sigue a EE. UU.: ya es GMT+3.
    utc = server_to_utc(pd.DatetimeIndex(["2025-03-10 10:00"]), 7)
    assert utc[0] == pd.Timestamp("2025-03-10 07:00", tz="UTC")


def _bars(times, **over):
    idx = pd.DatetimeIndex(pd.to_datetime(times), tz="UTC")
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5,
                       "tick_volume": 1.0, "spread_points": 7.0}, index=idx)
    for k, v in over.items():
        df[k] = v
    return df


def test_validate_detects_problems():
    df = _bars(["2025-01-06 10:00", "2025-01-06 10:15", "2025-01-06 10:15", "2025-01-06 13:00"])
    df.iloc[1, df.columns.get_loc("high")] = 8.0  # high < close
    rep = validate_bars(df, 15)
    assert rep.duplicates == 1
    assert rep.bad_ohlc == 1
    assert len(rep.unexpected_gaps) == 1  # hueco de 10:15 a 13:00 en día laborable
    assert not rep.ok


def test_validate_daily_break_and_weekend_are_expected():
    # Verano (NY = UTC-4). Pausa diaria: última vela 20:45 UTC (termina 21:00 = 17:00 NY),
    # siguiente 22:00 UTC (18:00 NY). Fin de semana: viernes 20:45 UTC -> domingo 22:00 UTC.
    times = [
        "2025-07-17 20:30",
        "2025-07-17 20:45",
        *pd.date_range("2025-07-17 22:00", "2025-07-18 20:45", freq="15min").strftime("%Y-%m-%d %H:%M"),
        "2025-07-20 22:00",
    ]
    df = _bars(times)
    rep = validate_bars(df, 15)
    assert len(rep.unexpected_gaps) == 0


def test_clean_bars_removes_duplicates_and_bad_rows():
    df = _bars(["2025-01-06 10:15", "2025-01-06 10:00", "2025-01-06 10:00", "2025-01-06 10:30"])
    df.iloc[3, df.columns.get_loc("low")] = 12.0  # low > open
    out = clean_bars(df)
    assert len(out) == 2
    assert out.index.is_monotonic_increasing


def test_test_split_is_locked():
    s = load_settings()
    df = _bars(["2025-06-02 10:00", "2025-09-01 10:00"])
    assert len(dev_data(df, s)) == 1
    with pytest.raises(PermissionError):
        holdout_data(df, s)
    assert len(holdout_data(df, s, i_know_this_is_the_final_test=True)) == 1
