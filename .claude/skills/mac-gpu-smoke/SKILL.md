---
name: mac-gpu-smoke
description: Smoke-test a CUDA/GPU ML pipeline (LLM / GRPO / RL training + eval) on an Apple Silicon Mac before the real run on a GPU box. Covers what the M1 GPU (PyTorch MPS + unified memory) can and cannot do, and how to run a tiny real end-to-end smoke (CPU/MPS, vLLM off, bf16 off) that proves the code works. Use when building GPU-targeted training/eval code on a Mac and you want to validate the whole pipeline locally, or when the user asks whether "Mac unified memory works like a GPU".
---

# 🍎 Mac GPU smoke: prove a GPU pipeline on Apple Silicon first

Goal: build a CUDA-targeted pipeline (e.g. GRPO / LoRA fine-tune + lm-eval) on a Mac, run a **tiny real end-to-end smoke** so the code is proven, and leave the full training for the GPU box. Never claim it "works" from a compile alone: actually run a few steps.

## 🧠 The reality of "unified memory ≈ GPU"

Partly true. The M1/M2/M3 GPU (PyTorch **MPS** backend) shares system RAM (unified memory), so it genuinely runs small models. But it is **not** a substitute for a CUDA GPU (A100 / RTX etc.):

- ✅ **Works on MPS:** loading a small model (`.to("mps")`) and `model.generate(...)` (inference). Proven: a 0.5B model generates in a few seconds.
- ✅ **Works on CPU:** a tiny training smoke and `lm-eval` (slow but reliable).
- ❌ **vLLM:** CUDA-only. Do not install it on the Mac; set `use_vllm=False` for the smoke.
- ❌ **bf16 on MPS:** flaky / partial op coverage. Use fp32 (or fp16 for inference only).
- ❌ **lm-eval generation on MPS:** segfaults (exit 139). Run eval on `--device cpu`.
- ❌ **Speed / scale:** far slower than CUDA; fine for a smoke, hopeless for the real run.

## ✅ The plan

1. Build all the GPU code on the Mac (env, train, eval, config).
2. Run a **tiny real smoke** here (CPU/MPS, vLLM off, bf16 off) to prove the pipeline end to end: data to train to save to eval to export.
3. Keep the full run (real budget, vLLM, bf16) for the GPU box.

## 📦 Install the Mac-testable subset (no vLLM)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers trl peft accelerate datasets lm-eval pyyaml
python -c "import torch; print('mps', torch.backends.mps.is_available())"   # expect True
```

The standard macOS arm64 torch wheel includes MPS. Omit `vllm` (CUDA-only).

## 🔬 The smoke recipe

- **Model:** something small (0.5B). **Steps:** 3 to 5. **Group/batch:** `num_generations=4`, `per_device_train_batch_size=4`. **Length:** `max_completion_length=64`. **Data:** a handful of examples.
- **Training smoke:** force **CPU** for reliability (MPS training hits op-coverage gaps): `use_cpu=True`, `bf16=False`, `use_vllm=False`. Confirm it saves an adapter / checkpoint.
- **Prove MPS works separately** (the "unified memory as GPU" claim): load the model `.to("mps")` and `model.generate` a few tokens. This is the honest MPS demo; the training itself stays on CPU.
- **Eval smoke:** `lm-eval` on **CPU** with a tiny slice: `--device cpu --limit 2 --apply_chat_template` and `dtype=float32`. MPS segfaults here.
- **Point:** validate the pipeline runs, not the numbers. Scores will be near-baseline (tiny model + few steps).

## 🧯 Gotchas (learned the hard way)

- **Version drift:** library configs change. Before trusting kwargs, inspect the dataclass: `import dataclasses; [f.name for f in dataclasses.fields(GRPOConfig)]`. (Example: trl 1.x dropped `max_prompt_length`.)
- **lm-eval on MPS = segfault (139):** always `--device cpu` on a Mac.
- **`apply_chat_template(..., return_tensors="pt")`** may return a dict (BatchEncoding), not a tensor: use `return_dict=True` and `model.generate(**inputs, ...)`.
- **bf16:** off for CPU and MPS smokes; fp32 for training, fp16 only for MPS inference.
- **Don't `pip install -r requirements.txt`** if it pins `vllm` (or CUDA torch): it will fail on the Mac. Install the subset by hand, keep the full `requirements.txt` for the GPU box.

## ✅ Verify checklist

- [ ] `torch.backends.mps.is_available()` is True
- [ ] the env / dataset builds and the reward/verifier returns expected values
- [ ] a 3-step training smoke saves an adapter/checkpoint
- [ ] the model loads on `mps` and generates (proves unified-memory GPU)
- [ ] `lm-eval` runs on CPU and writes results JSON
- [ ] the exporter turns those JSONs into the expected schema

Then say plainly: pipeline validated on the Mac; the full run needs the CUDA GPU box.

Related: [[unattended-run]] (drive the eventual long GPU run), [[readme-sync]] (keep the Mac-smoke vs GPU commands documented).
