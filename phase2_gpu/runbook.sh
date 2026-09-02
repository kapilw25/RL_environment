#!/usr/bin/env bash
# End-to-end Phase 2 runbook for the CUDA GPU box (RTX 6000 PRO / Blackwell).
# One command, unattended:
#   venv -> torch(cu128)+deps -> verify GPU -> sanity gate -> full GRPO train -> pre/post eval -> export.
#
# Usage:
#   ./runbook.sh
#
# Toggles (env vars, all optional):
#   TORCH_INDEX_URL=<url>   torch wheel index (default cu128; use .../nightly/cu128 if the arch fails)
#   USE_VLLM=1              use vLLM fast rollouts (default 0 = HF generation, most robust first run)
#   LIMIT=100              quick eval slice per task instead of the full test sets
#   GATE_STEPS=50 GATE_N=200   size of the short GPU sanity-gate train
#   SKIP_INSTALL=1 SKIP_GATE=1 SKIP_TRAIN=1 SKIP_EVAL=1   re-run only part of the pipeline
#   FORCE_REINSTALL=1      reinstall torch even if CUDA already works
#
# Aborts on hard failures (no CUDA, train crash); it does not judge whether the score moved.
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
ROOT="$(cd .. && pwd)"
LOGDIR="$ROOT/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/phase2_runbook.log"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1
echo "=== Phase 2 runbook @ $(date) | log: $LOG ==="

VENV="$HERE/.venv"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# ---- 1/6 venv ----
if [ ! -x "$VENV/bin/python" ]; then
  echo "== [1/6] create venv =="
  python3 -m venv "$VENV"
fi
set +u; source "$VENV/bin/activate"; set -u
pip install -q --upgrade pip

# ---- 2/6 install torch (cu128) + deps ----
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  echo "== [2/6] install: torch from $TORCH_INDEX_URL, then the stack =="
  if [ "${FORCE_REINSTALL:-0}" = "1" ] || ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    pip install --index-url "$TORCH_INDEX_URL" torch
  fi
  # required stack = requirements.txt minus vllm (vllm is installed best-effort next)
  CORE="$(grep -vE '^[[:space:]]*#|^[[:space:]]*$|vllm' requirements.txt | tr '\n' ' ')"
  pip install $CORE
  if pip install "vllm>=0.7"; then echo "[install] vllm ok"; else echo "!! vllm install failed; HF generation will be used"; fi
else
  echo "== [2/6] SKIP_INSTALL=1 =="
fi

# ---- verify the GPU actually runs a kernel (catches Blackwell/sm_120 torch mismatch) ----
echo "== verify CUDA =="
python - <<'PYEOF'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("CUDA not available. Install a Blackwell-capable torch and retry, e.g. "
             "TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128 ./runbook.sh")
try:
    float(torch.randn(4096, device="cuda").sum())   # real kernel: fails loudly on an arch mismatch
except Exception as e:
    sys.exit(f"torch is installed but a GPU kernel failed ({e}). This is usually a Blackwell/sm_120 "
             f"mismatch: retry with TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128")
print("[cuda ok]", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PYEOF

# vLLM: default off for a robust first run; USE_VLLM=1 enables it (needs colocate/served vllm).
export USE_VLLM="${USE_VLLM:-0}"
if [ "$USE_VLLM" = "1" ] && ! python -c "import vllm" 2>/dev/null; then
  echo "!! USE_VLLM=1 but vllm is not importable; falling back to HF generation"; export USE_VLLM=0
fi
echo "[config] USE_VLLM=$USE_VLLM (1 = vLLM fast rollouts)"

# small helper: print gsm8k score (%) from an lm-eval output dir
gate_score() {
  python - "$1" <<'PYEOF'
import glob, json, sys
d = sys.argv[1]
fs = sorted(glob.glob(d + "/**/results*.json", recursive=True) + glob.glob(d + "/results*.json"))
if not fs:
    print("n/a"); raise SystemExit
r = json.load(open(fs[-1])).get("results", {}).get("gsm8k", {})
for k in ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"):
    if isinstance(r.get(k), (int, float)):
        print(f"{r[k]*100:.1f}"); break
else:
    print("n/a")
PYEOF
}

# ---- 3/6 GPU sanity gate (unattended: run, log the early signal, continue) ----
if [ "${SKIP_GATE:-0}" != "1" ]; then
  echo "== [3/6] sanity gate: base GSM8K(100) -> ${GATE_STEPS:-50} GRPO steps -> re-check =="
  lm_eval --model hf --model_args "pretrained=${MODEL},dtype=bfloat16" \
    --tasks gsm8k --limit 100 --output_path results/gate_pre/
  python train_grpo.py --steps "${GATE_STEPS:-50}" --n "${GATE_N:-200}" --outdir results/grpo_gate
  lm_eval --model hf --model_args "pretrained=${MODEL},peft=results/grpo_gate,dtype=bfloat16" \
    --tasks gsm8k --limit 100 --output_path results/gate_post/
  echo "-- gate GSM8K: pre=$(gate_score results/gate_pre) post=$(gate_score results/gate_post) (informational; continuing)"
else
  echo "== [3/6] SKIP_GATE=1 =="
fi

# ---- 4/6 full GRPO train ----
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  echo "== [4/6] full GRPO train (config.yaml: 500 steps, 2000 problems) -> results/grpo =="
  python train_grpo.py --outdir results/grpo
else
  echo "== [4/6] SKIP_TRAIN=1 =="
fi

# ---- 5/6 pre/post eval on the 5 benchmarks (full unless LIMIT set) ----
if [ "${SKIP_EVAL:-0}" != "1" ]; then
  echo "== [5/6] pre/post eval on 5 benchmarks${LIMIT:+ (LIMIT=$LIMIT)} =="
  ./eval_harness.sh
else
  echo "== [5/6] SKIP_EVAL=1 =="
fi

# ---- 6/6 export metrics + summary table ----
echo "== [6/6] export Phase-2 metrics + summary table =="
python ../report/export_metrics_phase2.py --out ../docs/metrics_phase2.json --table results/summary_table.md

echo "=== DONE @ $(date) ==="
echo "  adapter : results/grpo"
echo "  metrics : docs/metrics_phase2.json   (commit to light up replay.html's Phase 2 tab)"
echo "  table   : results/summary_table.md   (paste into report_phase2.md)"
