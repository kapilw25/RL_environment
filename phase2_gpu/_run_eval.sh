#!/usr/bin/env bash
# Detached eval launcher: HF token (avoid rate-limit hangs) + batch=auto (per-task max that fits, no OOM).
source /venv/main/bin/activate
export HF_TOKEN=$(grep -E '^HF_TOKEN=' /workspace/RL_environment/.env | sed 's/^HF_TOKEN=//; s/#.*//; s/[[:space:]]//g')
export BATCH=auto
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/RL_environment/phase2_gpu
exec ./eval_harness.sh
