# Phase 2: RL-Fine-Tuning a Small LLM (GRPO) on 5 Public NLP Benchmarks

Runs on a CUDA GPU (RTX 6000 PRO). Base: **Qwen2.5-0.5B-Instruct + LoRA**, trained
with **GRPO** on a verifiable GSM8K-style math env (rule-based verifier reward, RLVR),
then evaluated on 5 public benchmarks with lm-evaluation-harness.

## Results (fill from `results/eval_pre` vs `results/eval_post`)

| Benchmark | Base (pre) | GRPO (post) | Delta |
|---|---|---|---|
| GSM8K | | | |
| MMLU | | | |
| ARC-Challenge | | | |
| HellaSwag | | | |
| TruthfulQA | | | |

## Method

- **Env** (`environment/math_env.py`): GSM8K prompts + a rule-based verifier reward (1.0 if the extracted final number is correct, else 0.0, plus a small format reward).
- **Train** (`train_grpo.py`): TRL `GRPOTrainer`, LoRA adapters, vLLM rollouts.
- **Eval** (`eval_harness.sh`): lm-eval-harness on the 5 tasks, base (pre) vs base+adapter (post).

## Honest caveat

Math-only RL reliably lifts GSM8K and math-aligned MMLU-STEM / ARC, but HellaSwag and
TruthfulQA may not move. To lift a broader set, make the env multi-skill (add verifiable
multiple-choice / factual tasks in `environment/math_env.py`).

## Reproduce

```bash
cd phase2_gpu && ./run.sh      # sanity gate -> train -> pre/post eval
```
