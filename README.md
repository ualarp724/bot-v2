# Phoenix — bot de trading para XAUUSD

Proyecto en reconstrucción siguiendo el plan "Plan Phoenix: bot de oro".
Objetivo: comprobar con pruebas honestas si hay una estrategia en oro (M15)
que gane dinero después de costes, y solo entonces operarla.

## Estructura

| Carpeta | Qué contiene |
| --- | --- |
| `phoenix/` | Código de verdad: datos, costes, features, etiquetas, backtest, modelo |
| `research/` | Experimentos e informes de investigación (fase 3) |
| `live/` | Ejecución en vivo: runner de Python y EA puente de MT5 (fase 5) |
| `tests/` | Tests con pytest |
| `config/` | Configuración (activo, costes, riesgo) |
| `data/raw/` | Datos exportados de MT5. No van a git |
| `legacy/` | Código, modelos, docs y resultados antiguos. Solo referencia |

`legacy/` no se mantiene: sus resultados tienen fugas de datos del futuro
(ver la auditoría del proyecto). Las piezas útiles se migran a `phoenix/`
una a una, con tests. Sus rutas a los CSV ya no funcionan porque los datos
están ahora en `data/raw/`.

## Entorno

Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # ejecución + desarrollo; solo para ejecutar: requirements.txt
pytest --cov=phoenix                  # recoge solo tests/ y falla si la cobertura baja de 65 %
ruff check . && ruff format --check .
```

## Calidad y CI

`.github/workflows/ci.yml` ejecuta en cada push y pull request (y cada lunes, para pillar avisos nuevos):

| Job | Qué comprueba |
| --- | --- |
| `lint` | `ruff check` y `ruff format --check` (configuración en `pyproject.toml`) |
| `audit` | `pip-audit` sobre `requirements-dev.txt`, dependencias transitivas incluidas |
| `secrets` | `gitleaks` sobre todo el historial; regla propia para claves `KRAKEN_*_KEY/SECRET` |
| `test` | `pytest --cov=phoenix` en Python 3.11 y 3.13; `fail_under = 65` en `pyproject.toml` |

- El workflow no usa secretos: tiene `contents: read`, acciones fijadas por SHA y Dependabot las mantiene al día.
- `legacy/` y `research/` quedan fuera de ruff y de pytest. `legacy/` no se mantiene y `research/` son experimentos.
- Excepción documentada de `pip-audit`: `CVE-2026-104874` (multidict 6.7.1). `ccxt` lo fija con `==` y la fuga
  de memoria solo afecta a servidores; quitar la excepción de `ci.yml` cuando `ccxt` permita `multidict>=6.9.1`.
- Si gitleaks marca un falso positivo, se añade su huella exacta a `.gitleaksignore` (con un comentario que lo explique).

## Datos en `data/raw/`

| Archivo | Activo | Marco | Uso |
| --- | --- | --- | --- |
| `vantage_gold.csv` | XAUUSD | M15 | Principal |
| `m5.csv` | XAUUSD | M5 | Apoyo |
| `h1.csv` | XAUUSD | H1 | Apoyo (desde 2018) |
| `vantage_live_gold.csv` | XAUUSD | — | Instantánea antigua del live |
| `vantage_btc1.csv`, `vantage_btc2.csv` | BTCUSD | M5 / M15 | Otros activos, más adelante |
| `vantage_nas100.csv` | NAS100 | M15 | Otros activos, más adelante |
| `vantage_eurusd.csv` | EURUSD | M15 | Otros activos, más adelante |

Todas las horas se pasan a UTC (el servidor de Vantage va con Nueva York + 7 h).

Periodos (ver `config/xauusd.json`):

- **Desarrollo:** 15/11/2021 – 31/07/2025. Aquí se investiga (fase 3).
- **Test intocable:** 01/08/2025 – 06/02/2026. Solo se abre en la fase 4 con
  `holdout_data(..., i_know_this_is_the_final_test=True)`.
- **Test 2:** feb–sep 2026, cuando se exporte de MT5.

Investigación (desde `research/`, cada script escribe su informe `.md`):

- `01_datos.py`: calidad de datos y costes
- `02_walkforward.py`: primera tanda (azar, reglas simples, LightGBM)
- `03_ruptura_y_movimiento.py`: segunda tanda (rupturas + modelo de movimiento)
- `experiments.csv`: registro de todas las configuraciones probadas

## Estudio aparte: Polymarket "Bitcoin Up or Down - 5 minutos"

Código en `phoenix/btc5m/` (solo lectura de datos públicos y simulación; no envía órdenes):

| Archivo | Qué hace |
| --- | --- |
| `binance.py` | Velas de 1 min de BTCUSDT (data.binance.vision) |
| `coinbase.py` | Velas de 1 min de BTC-USD (Coinbase) |
| `polymarket.py` | Mercados, resultado y operaciones alrededor del inicio |
| `features.py` | Features en el momento de decidir y objetivo aproximado |
| `backtest.py` | Apuestas a precios reales con comisión 0,07·p·(1−p) |
| `model.py` | Entrenar / guardar / cargar el modelo (`models/`, fuera de git) |
| `paper.py` | Bot en simulación en vivo (`python -m phoenix.btc5m.paper`) |

Scripts: `research/05_descargar_btc5m.py` (descarga), `06_backtest_btc5m.py` (backtest),
`07_retraso_chainlink.py` (retraso del precio de referencia). El resumen de precios por mercado
está en `data/raw/polymarket_btc5m_resumen.csv.gz`, así que el backtest se puede repetir sin
volver a descargar las operaciones. Desde España la API de Polymarket está bloqueada por la DGOJ.

## Estudio intradía (5m/15m) frente a 4h

`phoenix/intraday/` + `research/10_intradia.py`: backtest fuera de muestra (walk-forward mensual) con los costes de Kraken
Futures (taker 0,05 %, maker 0,02 %, deslizamiento ≥ 0,02 %), una posición como máximo, cierre obligatorio a las 23:45 UTC
y parada del día con -2 %. Compara la ruptura de 4h con la ruptura intradía (parámetros fijos y walk-forward) y un LightGBM,
con las mismas reglas de riesgo (0,5 % por operación). Necesita velas de 1 minuto de Binance (públicas):

```bash
python research/10_intradia.py --download                          # descarga + informe en research/10_intradia.md
python research/10_intradia.py --synthetic --out /tmp/prueba.md    # prueba de la tubería con un paseo aleatorio (NO es BTC)
```

El motor se valida con 40+ tests (cuentas a mano, ausencia de fugas del futuro con controles positivos y neutralidad: sin
ventaja y sin costes el bruto es ≈ 0).

## Bot de alto riesgo: perpetuo de BTC en Kraken (`phoenix/perps/`)

Objetivo que pidió Arturo: intentar pasar de 100 € a 500 € en un mes, aceptando perderlos.
Estrategia S4 de `research/08_bot_arriesgado.md`: ruptura de 20 velas de 4 h a favor de la
tendencia de ~50 días, stop de 2 ATR, 30 % de riesgo por operación, 9x como máximo y piramidar.
En simulación llega a 500 € en ~11 % de los meses (16,7 % en el test de 2026); el resultado
mediano es acabar con unos 70–80 €.

```bash
cp .env.example .env            # y pega tus claves (sin permiso de retirada)
python -m phoenix.perps.bot --mode dry           # simulación con velas reales
python -m phoenix.perps.bot --mode demo --check  # comprobar la cuenta demo
python -m phoenix.perps.bot --mode demo          # operar con dinero ficticio
python -m phoenix.perps.bot --mode live --i-accept-losing-everything
```

El stop y el objetivo quedan en Kraken, así que si el Mac se apaga la posición sigue protegida;
el bot solo tiene que estar encendido para abrir, piramidar y mover el stop cada 4 horas.
