"""Costes de operar XAUUSD en la cuenta Raw ECN y tamaño de posición.

Precios de MT5 = BID. Una compra entra al ASK (bid + spread) y sale al BID;
una venta entra al BID y sale al ASK. El spread se paga una vez por operación.
Todo en USD por onza salvo que se diga lo contrario (1 lote = 100 oz).
"""

from __future__ import annotations

import math

from phoenix.settings import Costs, Instrument


def spread_usd(spread_points: float, instrument: Instrument, costs: Costs) -> float:
    """Spread en USD/oz: el de la vela o el suelo configurado, el mayor de los dos."""
    return max(float(spread_points) * instrument.point, costs.spread_floor_usd)


def commission_usd(lots: float, costs: Costs) -> float:
    """Comisión de ida y vuelta (dos lados)."""
    return 2.0 * costs.commission_per_lot_per_side_usd * lots


def slippage_usd(lots: float, market_fills: int, instrument: Instrument, costs: Costs) -> float:
    """Deslizamiento: se paga en cada ejecución a mercado (entrada, SL, cierre por tiempo). El TP es límite: 0."""
    return costs.slippage_per_market_fill_usd * market_fills * instrument.contract_size_oz * lots


def position_size(sl_distance_usd: float, risk_usd: float, instrument: Instrument) -> float:
    """Lotes para no perder más de `risk_usd` si salta el SL.

    Se redondea hacia abajo al paso de lote. Si ni el lote mínimo cabe en el
    riesgo, devuelve 0.0 (la operación no se puede hacer con esta cuenta).
    """
    if sl_distance_usd <= 0 or risk_usd <= 0:
        return 0.0
    raw = risk_usd / (sl_distance_usd * instrument.contract_size_oz)
    steps = math.floor(raw / instrument.lot_step + 1e-9)
    lots = round(steps * instrument.lot_step, 8)
    if lots < instrument.min_lot:
        return 0.0
    return min(lots, instrument.max_lot)


def max_sl_distance_usd(risk_usd: float, instrument: Instrument) -> float:
    """SL máximo (USD/oz) que permite el riesgo con el lote mínimo."""
    return risk_usd / (instrument.min_lot * instrument.contract_size_oz)
