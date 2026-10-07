import pytest

from phoenix.costs import commission_usd, max_sl_distance_usd, position_size, slippage_usd, spread_usd
from phoenix.settings import load_settings

S = load_settings()
INS, COSTS = S.instrument, S.costs


def test_spread_uses_floor_when_bar_spread_is_too_low():
    assert spread_usd(1, INS, COSTS) == pytest.approx(0.12)   # 0,01 < suelo
    assert spread_usd(20, INS, COSTS) == pytest.approx(0.20)  # 0,20 > suelo


def test_commission_round_trip():
    assert commission_usd(0.01, COSTS) == pytest.approx(0.06)
    assert commission_usd(1.0, COSTS) == pytest.approx(6.0)


def test_slippage_only_market_fills():
    assert slippage_usd(0.01, 2, INS, COSTS) == pytest.approx(0.10)
    assert slippage_usd(0.01, 0, INS, COSTS) == 0


def test_position_size_rounds_down_and_respects_limits():
    # riesgo 5,69 $, SL 5 $/oz -> 0,01138 lotes -> 0,01
    assert position_size(5.0, 5.69, INS) == pytest.approx(0.01)
    # SL 6 $/oz: el lote mínimo arriesgaría 6 $ > 5,69 $ -> no se opera
    assert position_size(6.0, 5.69, INS) == 0.0
    # SL diminuto -> se limita al lote máximo
    assert position_size(0.1, 100.0, INS) == pytest.approx(INS.max_lot)
    assert position_size(0, 5, INS) == 0.0


def test_account_risk_budget():
    acc = S.account
    assert acc.capital_usd == pytest.approx(227.8, abs=0.1)
    assert acc.max_risk_usd == pytest.approx(5.69, abs=0.01)
    assert max_sl_distance_usd(acc.max_risk_usd, INS) == pytest.approx(5.69, abs=0.01)
