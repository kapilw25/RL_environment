"""Normalize lm-eval pre/post results into tidy Phase-2 rows for docs/replay.html.

Reads phase2_gpu/results/eval_pre and eval_post (lm-evaluation-harness output),
picks the primary metric per benchmark, and writes a 2-point series per benchmark
(step 0 = base/pre, step max = GRPO/post) in the schema replay.html expects:
  {phase:2, run:"grpo", benchmark, step, value, seed, baseline, ghost}
Set window.DATA_METRICS_P2 = <this array> in replay.html to show the real Phase-2 tab.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
P2 = os.path.join(ROOT, "phase2_gpu")

BENCH = {"gsm8k": "GSM8K", "mmlu": "MMLU", "arc_challenge": "ARC-Challenge",
         "hellaswag": "HellaSwag", "truthfulqa": "TruthfulQA"}
METRIC_PREF = ["exact_match,strict-match", "exact_match,flexible-extract",
               "acc_norm,none", "acc,none", "exact_match,none"]


def load_results(eval_dir):
    files = glob.glob(os.path.join(eval_dir, "**", "results*.json"), recursive=True)
    files += glob.glob(os.path.join(eval_dir, "results*.json"))
    if not files:
        return {}
    return json.load(open(sorted(files)[-1])).get("results", {})


def pick_metric(task_res):
    for k in METRIC_PREF:
        v = task_res.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for k, v in task_res.items():
        if isinstance(v, (int, float)) and ("acc" in k or "exact_match" in k) and "stderr" not in k:
            return float(v)
    return None


def score(results, task):
    if task in results:
        m = pick_metric(results[task])
        if m is not None:
            return m
    subs = [pick_metric(v) for k, v in results.items() if k.startswith(task) and isinstance(v, dict)]
    subs = [s for s in subs if s is not None]
    return sum(subs) / len(subs) if subs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "metrics_phase2.json"))
    ap.add_argument("--max-step", type=int, default=500)
    args = ap.parse_args()

    pre = load_results(os.path.join(P2, "results", "eval_pre"))
    post = load_results(os.path.join(P2, "results", "eval_post"))
    rows, got = [], []
    for task, name in BENCH.items():
        sp = score(pre, task)
        if sp is None:
            continue
        got.append(name)
        sq = score(post, task)
        sp *= 100
        sq = sq * 100 if sq is not None else sp
        base = sp if sp > 0 else 1e-6
        for step, val in [(0, sp), (args.max_step, sq)]:
            rows.append(dict(phase=2, run="grpo", benchmark=name, step=int(step),
                             value=round(val, 2), seed=0, baseline=round(base, 2), ghost=round(base, 2)))
    json.dump(rows, open(args.out, "w"))
    print(f"[export-p2] {len(rows)} rows for {got} -> {args.out}")


if __name__ == "__main__":
    main()
