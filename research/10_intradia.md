# Intradía (5m/15m) frente a 4h: backtest fuera de muestra con costes de Kraken Futures

Generado por `research/10_intradia.py` el 07/10/2026. Datos: **BTCUSDT spot de Binance (proxy del perpetuo de Kraken)**, 1,979,920 velas de 1 minuto (01/01/2023 – 06/10/2026). **Periodo fuera de muestra: 01/07/2023 – 06/10/2026** (40 tramos mensuales; entrenamiento/selección con los 6 meses anteriores). Velas incompletas descartadas: 5m 0, 15m 0, 4h 0.

## Supuestos

- **Costes:** taker 0.05% (entradas, stops y cierres forzados), maker 0.02% (objetivos, solo si el precio los atraviesa), deslizamiento 0.02% en cada operación a mercado y funding de largos 0.00125%/h. Coste mínimo de una operación completa a mercado: **0.14% del nocional**.
- **Riesgo (igual en todos los marcos):** 0.5% del equity por operación (pérdida en el stop), apalancamiento máximo 5x, capital inicial 10.000 $. La configuración anterior de 4h se vuelve a ejecutar con ESTE motor y ESTE riesgo para que la comparación sea justa; **no** es la 'lotería' de 100→500 € de `research/08` (30 % de riesgo por operación), que no admite Sharpe/PF comparables.
- **Reglas intradía (5m/15m):** 1 posición, máximo 6 operaciones/día, sin entradas desde las 22:00 UTC, **cierre obligatorio a las 23:45 UTC** (nunca se cruza la medianoche) y parada del día si el equity cae un 2 % desde el inicio del día UTC (se evalúa al cierre de cada vela; la pérdida del día puede rebasar el límite en lo que pierda una vela). 4h: sin cierre ni límite diario.
- **Ejecución:** la señal se decide al cierre de la vela y se ejecuta a la apertura de la siguiente; si stop y objetivo caen en la misma vela se asume el stop; un hueco más allá del stop se ejecuta a la apertura.
- **Multiplicidad:** se han evaluado 15 configuraciones y 12 combinaciones por tramo en el walk-forward. Con un 5 % de significación cabe esperar ~0.8 'positivos' por azar: léase cada fila con ese descuento.

## 1. Comparativa (todo fuera de muestra, costes descontados)

| Configuración | Ops | Ops/día | Ops/semana | Win rate | R:R medio | Profit Factor | Expectativa/op | Expectativa (R) [IC95] |
|---|---|---|---|---|---|---|---|---|
| 4h · Ruptura 20/10 (config. anterior) | 146 | 0.12 | 0.9 | 29.5 % | 3.62 | 1.51 | 18.02 $ (0.171 %) | 0.341 [-0.095; 0.861] |
| 15m · Ruptura (parámetros fijos) | 2,621 | 2.20 | 15.4 | 24.2 % | 1.75 | 0.56 | -3.75 $ (-0.151 %) | -0.312 [-0.393; -0.229] |
| 15m · Ruptura walk-forward | 135 | 0.11 | 0.8 | 31.1 % | 1.76 | 0.79 | -7.46 $ (-0.074 %) | -0.148 [-0.460; 0.219] |
| 15m · LightGBM (p ≥ 0.40) | 6,655 | 5.57 | 39.0 | 38.8 % | 0.93 | 0.59 | -1.50 $ (-0.106 %) | -0.227 [-0.251; -0.203] |
| 15m · LightGBM (p ≥ 0.45) | 6,343 | 5.31 | 37.2 | 39.8 % | 0.88 | 0.58 | -1.57 $ (-0.099 %) | -0.206 [-0.233; -0.180] |
| 5m · Ruptura (parámetros fijos) | 2,993 | 2.51 | 17.5 | 16.2 % | 2.58 | 0.50 | -3.34 $ (-0.255 %) | -0.631 [-0.748; -0.506] |
| 5m · Ruptura walk-forward | 62 | 0.05 | 0.4 | 21.0 % | 2.58 | 0.69 | -15.79 $ (-0.156 %) | -0.313 [-0.928; 0.539] |
| 5m · LightGBM (p ≥ 0.40) | 3,902 | 3.27 | 22.9 | 31.3 % | 0.94 | 0.43 | -2.56 $ (-0.222 %) | -0.510 [-0.563; -0.454] |
| 5m · LightGBM (p ≥ 0.45) | 4,128 | 3.46 | 24.2 | 31.4 % | 1.05 | 0.48 | -2.42 $ (-0.206 %) | -0.472 [-0.525; -0.417] |

| Configuración | Retorno total | Sharpe | Sortino | t (diario) | MDD | MDD ($) | DD medio (días) | DD máx. (días) | Días parados | Veredicto |
|---|---|---|---|---|---|---|---|---|---|---|
| 4h · Ruptura 20/10 (config. anterior) | 26.3 % | 0.90 | 1.68 | 1.63 | -6.4 % | 797 $ | 116.2 | 271 | 0 | No concluyente (IC95 incluye 0) |
| 15m · Ruptura (parámetros fijos) | -98.4 % | -3.75 | -5.62 | -6.79 | -98.5 % | 10,268 $ | 596.5 | 1188 | 135 | Pierde (IC95 de R < 0) |
| 15m · Ruptura walk-forward | -10.1 % | -0.49 | -0.87 | -0.89 | -17.1 % | 1,778 $ | 228.0 | 905 | 1 | No concluyente (IC95 incluye 0) |
| 15m · LightGBM (p ≥ 0.40) | -99.9 % | -9.39 | -9.84 | -16.98 | -99.9 % | 10,021 $ | 1193.0 | 1193 | 156 | Pierde (IC95 de R < 0) |
| 15m · LightGBM (p ≥ 0.45) | -99.8 % | -8.61 | -9.21 | -15.57 | -99.8 % | 10,038 $ | 1193.0 | 1193 | 142 | Pierde (IC95 de R < 0) |
| 5m · Ruptura (parámetros fijos) | -100.0 % | -4.84 | -7.79 | -8.75 | -100.0 % | 10,137 $ | 1193.0 | 1193 | 460 | Pierde (IC95 de R < 0) |
| 5m · Ruptura walk-forward | -9.8 % | -0.49 | -0.85 | -0.88 | -14.1 % | 1,474 $ | 296.3 | 874 | 4 | No concluyente (IC95 incluye 0) |
| 5m · LightGBM (p ≥ 0.40) | -100.0 % | -9.60 | -10.09 | -17.37 | -100.0 % | 10,045 $ | 1193.0 | 1193 | 382 | Pierde (IC95 de R < 0) |
| 5m · LightGBM (p ≥ 0.45) | -100.0 % | -9.39 | -9.96 | -16.99 | -100.0 % | 9,998 $ | 1193.0 | 1193 | 362 | Pierde (IC95 de R < 0) |

*Sharpe y Sortino anualizados con retornos diarios del equity (365 días, el perpetuo cotiza 24/7), comisiones y deslizamiento ya descontados; Sortino con desviación a la baja respecto a 0. MDD sobre el equity a mercado de cada vela. Duración del drawdown: de un máximo a su recuperación (o al final del periodo) para episodios de profundidad ≥ 0,5 %.*

## 2. Anatomía de los costes: ¿se lleva el broker el retorno?

| Configuración | P&L bruto | Comisiones | Deslizamiento | Funding | Costes totales | P&L neto | Costes / bruto | Bruto/op (pb) | Coste/op (pb) |
|---|---|---|---|---|---|---|---|---|---|
| 4h · Ruptura 20/10 (config. anterior) | 3,444 $ | 394 $ | 158 $ | 262 $ | 814 $ | 2,631 $ | 24 % | 62.1 | 20.6 |
| 15m · Ruptura (parámetros fijos) | -1,117 $ | 6,115 $ | 2,446 $ | 159 $ | 8,720 $ | -9,837 $ | sin bruto positivo | 1.1 | 14.3 |
| 15m · Ruptura walk-forward | 373 $ | 962 $ | 385 $ | 33 $ | 1,380 $ | -1,007 $ | 370 % | 2.6 | 14.4 |
| 15m · LightGBM (p ≥ 0.40) | 2,842 $ | 9,065 $ | 3,626 $ | 143 $ | 12,834 $ | -9,992 $ | 452 % | 2.1 | 14.2 |
| 15m · LightGBM (p ≥ 0.45) | 1,492 $ | 8,110 $ | 3,244 $ | 120 $ | 11,474 $ | -9,982 $ | 769 % | 1.9 | 14.2 |
| 5m · Ruptura (parámetros fijos) | 572 $ | 7,477 $ | 2,991 $ | 101 $ | 10,569 $ | -9,997 $ | 1846 % | 0.4 | 14.2 |
| 5m · Ruptura walk-forward | 36 $ | 715 $ | 286 $ | 14 $ | 1,015 $ | -979 $ | 2812 % | 7.9 | 14.2 |
| 5m · LightGBM (p ≥ 0.40) | 1,315 $ | 8,026 $ | 3,210 $ | 77 $ | 11,313 $ | -9,998 $ | 861 % | 2.1 | 14.1 |
| 5m · LightGBM (p ≥ 0.45) | 1,015 $ | 7,814 $ | 3,125 $ | 74 $ | 11,013 $ | -9,998 $ | 1085 % | 2.0 | 14.1 |

*'Bruto/op' y 'Coste/op' en puntos básicos del nocional (1 pb = 0,01 %). Una estrategia solo gana neto si su ventaja bruta por operación supera el coste por operación.*

## 3. Filtros adicionales (ablación sobre la ruptura de parámetros fijos)

Cada fila añade UN filtro a la configuración fija del mismo marco; no se ha elegido ninguno mirando el resultado OOS.

| Configuración | Ops | Win rate | Profit Factor | Exp. (R) [IC95] | Sharpe | MDD | P&L neto | Veredicto |
|---|---|---|---|---|---|---|---|---|
| 15m · Ruptura (parámetros fijos) | 2,621 | 24.2 % | 0.56 | -0.312 [-0.393; -0.229] | -3.75 | -98.5 % | -9,837 $ | Pierde (IC95 de R < 0) |
| 5m · Ruptura (parámetros fijos) | 2,993 | 16.2 % | 0.50 | -0.631 [-0.748; -0.506] | -4.84 | -100.0 % | -9,997 $ | Pierde (IC95 de R < 0) |
| 15m · Ruptura + volatilidad | 1,584 | 29.6 % | 0.82 | -0.120 [-0.210; -0.019] | -1.33 | -69.1 % | -6,387 $ | Pierde (IC95 de R < 0) |
| 15m · Ruptura + sesión 13-21 UTC | 1,360 | 28.7 % | 0.69 | -0.262 [-0.363; -0.153] | -2.67 | -83.8 % | -8,369 $ | Pierde (IC95 de R < 0) |
| 15m · Ruptura + ambos | 995 | 32.4 % | 0.82 | -0.118 [-0.224; 0.005] | -1.11 | -49.8 % | -4,669 $ | No concluyente (IC95 incluye 0) |
| 5m · Ruptura + volatilidad | 2,331 | 19.9 % | 0.64 | -0.347 [-0.453; -0.234] | -3.25 | -98.6 % | -9,854 $ | Pierde (IC95 de R < 0) |
| 5m · Ruptura + sesión 13-21 UTC | 1,749 | 20.0 % | 0.54 | -0.538 [-0.646; -0.417] | -4.55 | -98.8 % | -9,877 $ | Pierde (IC95 de R < 0) |
| 5m · Ruptura + ambos | 1,403 | 22.3 % | 0.63 | -0.341 [-0.461; -0.215] | -2.98 | -91.7 % | -9,162 $ | Pierde (IC95 de R < 0) |

## 4. Sensibilidad a los costes (ruptura de parámetros fijos)

| Marco | Supuesto | Ops | P&L neto | Profit Factor | Sharpe |
|---|---|---|---|---|---|
| 15m | deslizamiento 0 % | 2,647 | -9,487 $ | 0.66 | -2.69 |
| 15m | deslizamiento 0,02 % (base) | 2,621 | -9,837 $ | 0.56 | -3.75 |
| 15m | deslizamiento 0,05 % | 2,567 | -9,965 $ | 0.45 | -5.22 |
| 15m | sin comisión ni deslizamiento | 2,686 | 2,811 $ | 1.03 | 0.39 |
| 5m | deslizamiento 0 % | 3,295 | -9,984 $ | 0.61 | -3.57 |
| 5m | deslizamiento 0,02 % (base) | 2,993 | -9,997 $ | 0.50 | -4.84 |
| 5m | deslizamiento 0,05 % | 2,145 | -9,999 $ | 0.39 | -6.29 |
| 5m | sin comisión ni deslizamiento | 3,652 | 2,513 $ | 1.02 | 0.38 |

*Si la fila 'sin comisión ni deslizamiento' tampoco gana, no hay ventaja bruta que los costes se estén comiendo.*

## 5. Walk-forward: qué se eligió en cada tramo

- **15m:** el walk-forward operó en 3 de 40 tramos (en el resto ninguna configuración tuvo t > 0 con ≥ 30 operaciones en el entrenamiento y se quedó en efectivo). Elegidas: 3× [n_entry=64, stop=2.5 ATR, tendencia=no].
- **5m:** el walk-forward operó en 1 de 40 tramos (en el resto ninguna configuración tuvo t > 0 con ≥ 30 operaciones en el entrenamiento y se quedó en efectivo). Elegidas: 1× [n_entry=192, stop=2.5 ATR, tendencia=sí].
- **15m LightGBM:** modelo entrenado en 40 de 40 tramos.
- **5m LightGBM:** modelo entrenado en 40 de 40 tramos.

## 6. Diagnóstico

*(Se rellena a mano al leer los resultados: ver el mensaje de la sesión.)*

## Límites de este estudio

- Datos de Binance spot (BTCUSDT) como proxy del perpetuo de Kraken: se ignoran la base y las diferencias de liquidez.
- Deslizamiento constante (0,02 %): en movimientos bruscos o con poca liquidez real será mayor; no hay libro de órdenes ni cola.
- Los objetivos limit se dan por ejecutados solo si el precio los atraviesa, pero sin modelar la prioridad en la cola.
- El resultado depende del régimen del periodo OOS; un solo periodo no garantiza nada sobre el siguiente.

Tiempo de cálculo: 125 s.