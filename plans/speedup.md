Why 50% is out of reach for this exact workload
- Well-tuned dense transformers hit 40–55% MFU on H100s with bf16 + FA2/3; SFT/DPO hit 35–50% only if the batch is large enough to keep arithmetic intensity high. (Introl, zeroentropy)
- In GRPO/RL, the rollout (generation) phase is >70% of each step's wall-clock, and autoregressive decode is memory-bandwidth-bound (inherently low MFU). That's the sawtooth you saw. (verl-based RL reports)
- A 0.5B model has low arithmetic intensity — small GEMMs don't saturate tensor cores — so its MFU ceiling is far below a 7B+ on the same GPU. And a consumer 4080S (no NVLink, less bandwidth) sits below H100 numbers.

The levers, ranked for our case (Lambda MFU whitepaper reports ~25–35% median MFU uplift from stacking these):

1. vLLM for rollout — removes the >70% generation tax from the critical path (this is exactly why verl/RL frameworks integrate vLLM + FSDP). Biggest single win; takes effective util from single digits to the 10–25% range.
2. Bigger batch / more parallel sequences (num_generations, per_device_train_batch_size, generation batch) — raises arithmetic intensity so the update-phase GEMMs stay full. This is the "batch large enough" condition behind the 35–50% SFT numbers.
3. FlashAttention-2/3 + bf16 — the 40–55% figures all assume it; confirm attn_implementation="flash_attention_2".
4. Use a bigger model (1.5B/3B/7B) — counterintuitively raises MFU: larger matmuls saturate tensor cores. A 0.5B will never reach 50% regardless of tuning.
5. Sequence packing (no pad waste), torch.compile/fused kernels, overlap reward computation to avoid CPU stalls.

## 🧮 Hardware to run ALL of these techniques in a SINGLE run

Why the numbers jump: VRAM is dominated by (a) the model weights, (b) a SECOND copy of the weights for the colocated vLLM engine, and (c) the KV cache + activation headroom that "large batch + long completions" need. So plan for roughly `2x model weights + KV cache + batch activations`, plus FA2 (which lowers attention memory) and torch.compile (a small cache).

Single-GPU colocate (LoRA train + vLLM rollout + FA2 + bf16 + packing + torch.compile), per model tier:

| Model | GPU VRAM | GPUs | CPU RAM | Disk | Realistic MFU | Example card |
|---|---|---|---|---|---|---|
| 0.5B (current) | 16 GB | 1 | 32 GB | 32 GB | ~10 to 25% | RTX 4080S / 4090 |
| 1.5B | 24 GB | 1 | 32 GB | 50 GB | ~15 to 30% | RTX 4090 / A5000 (24 GB) |
| 3B | 40 to 48 GB | 1 | 48 GB | 60 GB | ~25 to 40% | A6000 / L40S / A100-40 |
| 7B (to reach the 40 to 55% regime) | 80 GB | 1 | 64 to 96 GB | 80 to 100 GB | ~40 to 55% | A100-80 / H100-80 |

Cleaner and faster alternative (removes colocate memory contention, what verl / OpenRLHF do): 2 GPUs, one training (FSDP + LoRA), one running `trl vllm-serve` for rollout, so neither GPU holds two jobs. Example for 7B: 2x 48 GB. This raises throughput but doubles GPU cost.

Per-technique incremental cost (what each lever adds on top of a plain LoRA run):
- vLLM colocate: +1x model weights (its own copy) + KV cache; KV scales with `num_generations x max_completion_length x prompts`. Tune `gpu_memory_utilization` (leave room for training).
- Large batch / more generations: +activation memory (grows with batch x seq_len). This is the main VRAM lever after weights, and the one the 16 GB card runs out of first.
- FA2 / FA3: needs `flash-attn` (matched wheel, or compile with nvcc: +~5 GB CUDA dev toolkit on disk, +build time/CPU). It LOWERS attention memory, so it partly pays for itself.
- Bigger model: dominant disk + VRAM driver. HF-cache weights on disk: 0.5B ~1 GB, 1.5B ~3 GB, 3B ~6 GB, 7B ~14 to 16 GB (double in VRAM for train + vLLM copies).
- torch.compile: +Triton compile cache (~1 to 2 GB disk) + a one-time CPU compile stall at start.
- Sequence packing + `train_n` up to 10k: negligible disk; more CPU dataloader workers help.

Disk breakdown for the full stack (why 50 to 100 GB): venv with cu-torch + vLLM + flash-attn + transformers/trl/peft/accelerate/datasets/lm-eval + tensorboard is ~15 to 18 GB, plus the model cache (above), plus ~3 GB of eval datasets, plus ~2 to 5 GB of adapters/checkpoints/tb logs, plus the optional ~5 GB CUDA dev toolkit for the flash-attn build.

Bottom line: on our 4080S (16 GB, 32 GB disk) the honest ceiling with everything on is a 1.5B at modest batch, ~10 to 25% MFU. To actually approach 40 to 55% MFU you need a 7B on an 80 GB A100/H100 (single-GPU colocate) or 2x 48 GB (dedicated vLLM server) - a different, pricier machine than a $0.18/hr 4080S.
