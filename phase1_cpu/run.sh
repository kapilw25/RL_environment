#!/usr/bin/env bash
# Gated Phase 1 runbook. Chains setup, checks, smoke, full run, export.
# Pauses at G2 and G3 for a human eyeball unless --yes is passed.
#   ./run.sh                 # fast budget, interactive gates
#   ./run.sh --yes           # no pauses
#   ./run.sh --budget=full   # 5M-frame budget (overnight)
set -euo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
AUTO=0
BUDGET=fast
for a in "$@"; do
  case "$a" in
    --yes) AUTO=1 ;;
    --budget=*) BUDGET="${a#*=}" ;;
  esac
done
pause() { if [ "$AUTO" -eq 0 ]; then read -r -p ">> $1 (Enter to continue, Ctrl-C to stop) " _; fi; }

echo "== A: env sanity =="
$PY -c "import stable_baselines3, gymnasium, minatar; print('stack ok')"

echo "== B / G1: check environments =="
$PY - <<'PYEOF'
from envs import make_env, minatar_available
print("minatar_available:", minatar_available())
for eid in ["GridWorld-v0", "MinAtar/Breakout-v1"]:
    env = make_env(eid, 0)()
    print(eid, "obs", env.observation_space.shape, "act", env.action_space)
    env.close()
PYEOF
pause "G1: environments look right?"

echo "== C / G2: smoke tests =="
$PY train.py --env-id GridWorld-v0 --timesteps 100000 --seed 0 --outdir results/_smoke_grid
$PY evaluate.py --model results/_smoke_grid/final_model.zip --env-id GridWorld-v0 --episodes 10
$PY train.py --env-id MinAtar/Breakout-v1 --timesteps 50000 --seed 0 --outdir results/_smoke_brk
$PY evaluate.py --model results/_smoke_brk/final_model.zip --env-id MinAtar/Breakout-v1 --episodes 10
pause "G2: reward rising on both? go / no-go for the full run"

echo "== E1: full benchmark ($BUDGET) =="
$PY benchmark.py --budget "$BUDGET" --seeds 0 1 2

echo "== F: export metrics for the viz =="
$PY ../report/export_metrics.py --out ../docs/metrics_phase1.json

echo "== G3: open docs/replay.html and confirm real curves =="
pause "G3: viz verified?"

echo "== done. see results/summary.md =="
