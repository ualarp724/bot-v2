#!/usr/bin/env bash
# Skills candidatas para Phoenix (bot de trading Python: pandas/LightGBM/ccxt/Kraken).
# EJECUTAR EN TU TERMINAL, desde la raíz del repo, con acceso a skills.sh y GitHub.
# (La sesión en la nube bloquea skills.sh con 403, por eso no se pudo ejecutar allí.)
#
# Fase 1 imprime resultados REALES de `npx skills find` (con installs y fuente) para que juzgues.
# Fase 2 instala solo los candidatos que existan de verdad en su repo; los nombres salen de memoria
# y no están verificados contra el índice, así que cada uno se comprueba con `--list` antes de instalar.
set -uo pipefail

strip() { sed 's/\x1b\[[0-9;]*m//g'; }

QUERIES=(
  "python testing" "pytest" "python lint" "backtesting" "trading" "quantitative"
  "ccxt" "secrets management" "github actions" "python performance" "pandas" "security"
)

# repo@skill — prioridad: testing/validación > seguridad > CI/CD > flujo Git > rendimiento
CANDIDATES=(
  "wshobson/agents@backtesting-frameworks"          # look-ahead, walk-forward, sesgos (tu problema de fugas)
  "wshobson/agents@risk-metrics-calculation"        # Sharpe/drawdown/VaR para validar el backtest
  "wshobson/agents@python-testing-patterns"         # pytest, fixtures, mocks de exchange
  "obra/superpowers@test-driven-development"        # test antes del fix en la ruta live
  "obra/superpowers@systematic-debugging"           # depurar fallos de ejecución de órdenes
  "obra/superpowers@verification-before-completion" # no dar nada por hecho sin evidencia
  "wshobson/agents@secrets-management"              # claves Kraken, rotación, .env
  "wshobson/agents@github-actions-templates"        # CI: ruff + pytest + cobertura
  "trailofbits/skills@insecure-defaults"            # configuración insegura por defecto
  "wshobson/agents@python-performance-optimization" # profiling de features/backtest
  "obra/superpowers@finishing-a-development-branch" # cierre de ramas/PRs
)

echo "=== FASE 1: descubrimiento (revisa installs y fuente) ==="
for q in "${QUERIES[@]}"; do
  echo; echo "### npx skills find \"$q\""
  npx --yes skills find "$q" 2>&1 | strip | grep -v '^npm notice' | head -15
done

echo
read -r -p "¿Instalar los candidatos que existan en su repo? [y/N] " ans
[[ "$ans" =~ ^[yY]$ ]] || { echo "Cancelado."; exit 0; }

echo "=== FASE 2: instalación verificada ==="
for c in "${CANDIDATES[@]}"; do
  c="${c%% *}"; repo="${c%@*}"; skill="${c#*@}"
  if npx --yes skills add "$repo" --list 2>&1 | strip | grep -qw -- "$skill"; then
    echo ">> instalando $c"
    npx --yes skills add "$c" -y
  else
    echo "-- OMITIDA (no aparece en $repo): $skill"
  fi
done

echo
echo "Hecho. Las skills quedan en .agents/skills y .claude/skills (están versionadas en git)."
echo "Revisa cada SKILL.md antes de hacer commit: es contenido de terceros en un repo con claves de exchange."
