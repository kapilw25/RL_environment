#!/usr/bin/env bash
# M1-runnable end-to-end smoke: tiny GRPO train (CPU) -> tiny lm-eval (CPU) -> export.
# Proves the Phase 2 pipeline works on this laptop. The FULL run belongs on the CUDA
# box (see run.sh): vLLM is CUDA-only and lm-eval generation segfaults on MPS, so this
# smoke uses CPU. Scores are ~0 (0.5B base on gsm8k with 2 examples); real numbers need the GPU.
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python

echo "== 1/3 GRPO smoke (3 steps, multi-skill mix, CPU) =="
$PY train_grpo.py --smoke --outdir results/grpo_smoke

echo "== 2/3 lm-eval pre/post (gsm8k, --limit 2, CPU) =="
./.venv/bin/lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float32 \
  --tasks gsm8k --limit 2 --device cpu --apply_chat_template --output_path results/eval_pre/
./.venv/bin/lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct,peft=results/grpo_smoke,dtype=float32 \
  --tasks gsm8k --limit 2 --device cpu --apply_chat_template --output_path results/eval_post/

echo "== 3/3 export Phase 2 metrics for docs/replay.html =="
$PY ../report/export_metrics_phase2.py --out ../docs/metrics_phase2.json
echo "smoke done."
