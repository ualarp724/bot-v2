"""Carga la configuración del activo desde config/<símbolo>.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data" / "raw"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    point: float
    contract_size_oz: float
    lot_step: float
    min_lot: float
    max_lot: float


@dataclass(frozen=True)
class Costs:
    commission_per_lot_per_side_usd: float
    spread_floor_usd: float
    slippage_per_market_fill_usd: float


@dataclass(frozen=True)
class Account:
    currency: str
    capital_eur: float
    eurusd_at_decision: float
    max_risk_per_trade_pct: float
    flat_before_daily_break: bool

    @property
    def capital_usd(self) -> float:
        return self.capital_eur * self.eurusd_at_decision

    @property
    def max_risk_usd(self) -> float:
        return self.capital_usd * self.max_risk_per_trade_pct / 100.0


@dataclass(frozen=True)
class Splits:
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class Settings:
    instrument: Instrument
    costs: Costs
    account: Account
    splits: Splits
    server_offset_from_new_york_hours: int


def _strip_comments(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_settings(symbol: str = "xauusd") -> Settings:
    raw = json.loads((CONFIG_DIR / f"{symbol.lower()}.json").read_text(encoding="utf-8"))
    splits = {k: pd.Timestamp(v, tz="UTC") for k, v in _strip_comments(raw["splits"]).items()}
    return Settings(
        instrument=Instrument(**_strip_comments(raw["instrument"])),
        costs=Costs(**_strip_comments(raw["costs"])),
        account=Account(**_strip_comments(raw["account"])),
        splits=Splits(**splits),
        server_offset_from_new_york_hours=int(raw["server_time"]["offset_from_new_york_hours"]),
    )
