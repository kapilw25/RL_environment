"""GRPO + LoRA training on the verifiable multi-skill env (TRL GRPOTrainer).

  python train_grpo.py --smoke   # tiny, M1-runnable: CPU, vLLM off, bf16 off, 3 steps
  python train_grpo.py           # full run (config.yaml), for the CUDA GPU box

The full run uses bf16 + vLLM (config.yaml); the smoke forces CPU so it runs on the M1.
"""
import argparse
import os

import yaml
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from environment import build_dataset, reward_correct, reward_format

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny M1 smoke: CPU, vLLM off, bf16 off")
    p.add_argument("--outdir", default="results/grpo")
    p.add_argument("--steps", type=int, default=None, help="override max_steps (e.g. a short GPU sanity gate)")
    p.add_argument("--n", type=int, default=None, help="override train_n (number of problems to sample)")
    args = p.parse_args()

    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    g, lc, d = cfg["grpo"], cfg["lora"], cfg["data"]
    mix = tuple(d.get("mix", ["math", "mc"]))

    if args.smoke:
        steps, n = 3, 8
        extra = dict(num_generations=4, per_device_train_batch_size=4,
                     gradient_accumulation_steps=1,
                     max_completion_length=64, bf16=False, use_vllm=False, use_cpu=True)
        report = "none"
    else:
        steps = args.steps if args.steps is not None else g["max_steps"]
        n = args.n if args.n is not None else d["train_n"]
        use_vllm = g["use_vllm"]
        env_vllm = os.environ.get("USE_VLLM")            # runbook can force vLLM off for a robust first run
        if env_vllm is not None:
            use_vllm = env_vllm.strip().lower() not in ("0", "false", "no", "")
        extra = dict(num_generations=g["num_generations"],
                     per_device_train_batch_size=g["per_device_train_batch_size"],
                     gradient_accumulation_steps=g["gradient_accumulation_steps"],
                     max_completion_length=g["max_completion_length"],
                     bf16=g["bf16"], use_vllm=use_vllm)
        report = "tensorboard"

    train_ds = build_dataset("train", n, d["seed"], mix=mix)
    peft_config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                             target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    grpo = GRPOConfig(output_dir=args.outdir, learning_rate=g["learning_rate"],
                      max_steps=steps, logging_steps=g["logging_steps"],
                      save_steps=g["save_steps"], report_to=report, **extra)
    trainer = GRPOTrainer(model=cfg["model"], reward_funcs=[reward_correct, reward_format],
                          args=grpo, train_dataset=train_ds, peft_config=peft_config)
    print(f"[train] {cfg['model']} | steps={steps} n={n} mix={mix} smoke={args.smoke} -> {args.outdir}")
    trainer.train()
    trainer.save_model(args.outdir)
    print("[train] saved LoRA adapter to", args.outdir)


if __name__ == "__main__":
    main()
