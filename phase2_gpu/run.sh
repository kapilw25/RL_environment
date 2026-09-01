#!/usr/bin/env bash
# Gated Phase 2 runbook: sanity gate -> full GRPO train -> pre/post eval.
# Run on the GPU box with the venv active.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
pause() { read -r -p ">> $1 (Enter to continue, Ctrl-C to stop) " _; }

echo "== 🚦 sanity gate: base GSM8K (100) -> 50 GRPO steps -> re-check =="
lm_eval --model hf --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16" \
  --tasks gsm8k --limit 100 --output_path results/gate_pre/
$PY train_grpo.py --smoke --outdir results/grpo_smoke
lm_eval --model hf --model_args "pretrained=Qwen/Qwen2.5-0.5B-Instruct,peft=results/grpo_smoke,dtype=bfloat16" \
  --tasks gsm8k --limit 100 --output_path results/gate_post/
pause "did GSM8K move up (gate_pre vs gate_post)? go / no-go"

echo "== full GRPO train =="
$PY train_grpo.py --outdir results/grpo

echo "== pre/post eval on the 5 benchmarks =="
./eval_harness.sh
echo "done. fill report_phase2.md from results/eval_pre vs results/eval_post"
