#!/usr/bin/env bash
# Evaluate base vs GRPO-tuned model on the 5 public NLP benchmarks (pre / post).
# Post loads the LoRA adapter on top of the base via lm-eval's peft= arg.
#   ./eval_harness.sh            # full eval
#   LIMIT=100 ./eval_harness.sh  # quick 100-example slice per task
set -euo pipefail
cd "$(dirname "$0")"

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER="results/grpo"
TASKS="gsm8k,mmlu,arc_challenge,hellaswag,truthfulqa"
EXTRA=""; [ -n "${LIMIT:-}" ] && EXTRA="--limit ${LIMIT}"

echo "== PRE: base model =="
lm_eval --model hf --model_args "pretrained=${MODEL},dtype=bfloat16" \
  --tasks "$TASKS" --batch_size auto --output_path results/eval_pre/ $EXTRA

echo "== POST: base + GRPO LoRA adapter =="
lm_eval --model hf --model_args "pretrained=${MODEL},peft=${ADAPTER},dtype=bfloat16" \
  --tasks "$TASKS" --batch_size auto --output_path results/eval_post/ $EXTRA

echo "done: compare results/eval_pre vs results/eval_post"
