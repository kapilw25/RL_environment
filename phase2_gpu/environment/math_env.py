"""Verifiable multi-skill RL environment for RLVR (the Phase 2 "environment").

Two verifiable skills, each with a rule-based verifier (no human labels):
  - math (GSM8K):  extract the final number, +1.0 if it matches the gold answer
  - mc (ARC-Challenge science): extract the chosen letter, +1.0 if it matches
plus a small format reward for ending with 'The answer is ...'.

Mixing math with multiple-choice science aims to lift the math-aligned AND
knowledge/science benchmarks (GSM8K, MMLU, ARC). Add more skills the same way:
yield {prompt, answer, task} and extend the verifier switch in reward_correct.
"""
import re

from datasets import concatenate_datasets, load_dataset

MATH_SYS = "Solve the problem. Think step by step, then end with 'The answer is <number>'."
MC_SYS = "Answer the multiple-choice question. Reason briefly, then end with 'The answer is <letter>'."
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _take(ds, n, seed):
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def build_math(n, seed):
    ds = _take(load_dataset("openai/gsm8k", "main", split="train"), n, seed)

    def fmt(ex):
        gold = ex["answer"].split("####")[-1].strip().replace(",", "")
        return {"prompt": [{"role": "system", "content": MATH_SYS},
                           {"role": "user", "content": ex["question"]}],
                "answer": gold, "task": "math"}

    return ds.map(fmt, remove_columns=ds.column_names)


def build_mc(n, seed):
    ds = _take(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"), n, seed)

    def fmt(ex):
        texts, labels = ex["choices"]["text"], ex["choices"]["label"]
        opts = "\n".join(f"{chr(65 + i)}. {t}" for i, t in enumerate(texts))
        key = ex["answerKey"].strip()
        letter = chr(65 + labels.index(key)) if key in labels else key.upper()
        return {"prompt": [{"role": "system", "content": MC_SYS},
                           {"role": "user", "content": ex["question"] + "\n" + opts}],
                "answer": letter, "task": "mc"}

    return ds.map(fmt, remove_columns=ds.column_names)


def build_dataset(split="train", n=2000, seed=0, mix=("math", "mc")):
    """Build a shuffled mix of the verifiable skills. `mix` selects which skills."""
    builders = {"math": build_math, "mc": build_mc}
    per = max(1, n // len(mix))
    parts = [builders[m](per, seed) for m in mix if m in builders]
    return concatenate_datasets(parts).shuffle(seed=seed)


def extract_number(text):
    m = re.search(r"answer is\s*:?\s*\$?(-?\d[\d,]*\.?\d*)", text, re.I)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = _NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def extract_letter(text):
    m = re.search(r"answer is\s*:?\s*\(?([A-E])\)?", text, re.I)
    if m:
        return m.group(1).upper()
    ms = re.findall(r"\b([A-E])\b", text)
    return ms[-1].upper() if ms else ""


def _content(completion):
    # TRL passes conversational completions as a list of message dicts
    return completion[-1]["content"] if isinstance(completion, list) else completion


def reward_correct(completions, answer, task, **kwargs):
    """+1.0 if the verifier for that skill accepts the answer, else 0.0."""
    out = []
    for c, gold, t in zip(completions, answer, task):
        text = _content(c)
        pred = extract_number(text) if t == "math" else extract_letter(text)
        out.append(1.0 if pred == gold else 0.0)
    return out


def reward_format(completions, **kwargs):
    """Small shaping reward for ending with 'The answer is ...'."""
    return [0.2 if re.search(r"answer is", _content(c), re.I) else 0.0 for c in completions]
