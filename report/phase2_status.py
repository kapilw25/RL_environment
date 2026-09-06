"""Derive live Phase-2 pipeline status from the filesystem/logs -> docs/phase2_status.json.

Pure stdlib (no venv needed). The docs/phase2_pipeline.html page polls the JSON.
Each node gets a status in: done, running, pending, blocked, failed, manual.
Run once, or loop it every ~15s to keep the page live.
"""
import glob
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P2 = os.path.join(ROOT, "phase2_gpu")
LOGS = os.path.join(ROOT, "logs")
DOCS = os.path.join(ROOT, "docs")

TASKS = [("gsm8k", "GSM8K"), ("mmlu", "MMLU"), ("arc_challenge", "ARC-Challenge"),
         ("hellaswag", "HellaSwag"), ("truthfulqa", "TruthfulQA")]
METRIC_PREF = ["exact_match,strict-match", "exact_match,flexible-extract",
               "acc_norm,none", "acc,none", "exact_match,none"]


def alive(pat):
    return subprocess.run(["pgrep", "-f", pat], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def newest(pattern, recursive=False):
    fs = sorted(glob.glob(pattern, recursive=recursive))
    return fs[-1] if fs else None


def read(path):
    try:
        return open(path, errors="ignore").read()
    except Exception:
        return ""


def tqdm_step(text, total):
    m = re.findall(r"(\d+)/%d[ \]]" % total, text.replace("\r", "\n"))
    return int(m[-1]) if m else 0


def last_reward(text):
    m = re.findall(r"'reward': '([-0-9.eE]+)'", text)
    return m[-1] if m else None


def has_error(text):
    tail = text[-4000:]
    return ("Traceback (most recent call last)" in tail) or ("\nError" in tail) or (": error:" in tail)


def parse_series(text, keep=90):
    """Pair each per-step metrics dict with the step number from the nearest tqdm bar."""
    lines = text.replace("\r", "\n").split("\n")
    step, rows = 0, []
    stepre = re.compile(r"(\d+)/500")
    for ln in lines:
        m = stepre.search(ln)
        if m:
            step = int(m.group(1))
        if "'loss'" in ln and "'reward'" in ln:
            def g(k):
                mm = re.search(r"'%s': '([-0-9.eE+]+)'" % k, ln)
                try:
                    return round(float(mm.group(1)), 4) if mm else None
                except Exception:
                    return None
            rows.append({"step": step, "loss": g("loss"), "reward": g("reward"),
                         "rc": g("rewards/reward_correct/mean"), "rf": g("rewards/reward_format/mean"),
                         "entropy": g("entropy"), "clen": g("completions/mean_length"),
                         "gnorm": g("grad_norm")})
    return rows[-keep:]


def pick_metric(task_res):
    for k in METRIC_PREF:
        v = task_res.get(k)
        if isinstance(v, (int, float)):
            return round(float(v) * 100, 1)
    return None


def lm_scores(d):
    f = newest(os.path.join(d, "**", "results*.json"), recursive=True) or newest(os.path.join(d, "results*.json"))
    if not f:
        return {}
    try:
        res = json.load(open(f)).get("results", {})
    except Exception:
        return {}
    out = {}
    for task, _ in TASKS:
        if task in res:
            m = pick_metric(res[task])
            if m is not None:
                out[task] = m
        else:
            subs = [pick_metric(v) for k, v in res.items() if k.startswith(task) and isinstance(v, dict)]
            subs = [s for s in subs if s is not None]
            if subs:
                out[task] = round(sum(subs) / len(subs), 1)
    return out


def build():
    nodes = {}

    # 1. install
    have = all(glob.glob("/venv/main/lib/python*/site-packages/%s" % p) for p in ("trl", "peft", "lm_eval", "tensorboard"))
    nodes["install"] = {"icon": "\U0001F4E6", "label": "Install deps",
                        "status": "done" if have else "pending",
                        "detail": "torch 2.11 + trl / peft / lm-eval" if have else "not installed"}

    # 3. full train (needed by others below, compute early)
    ftlog = newest(os.path.join(LOGS, "p2_full_train_*.log"))
    fttext = read(ftlog) if ftlog else ""
    ft_adapter = os.path.exists(os.path.join(P2, "results", "grpo", "adapter_model.safetensors"))
    ft_running = alive("train_grpo.py --outdir")
    ft_step = tqdm_step(fttext, 500)
    if ft_adapter:
        ft = {"status": "done", "detail": "500/500 steps", "progress": 100}
    elif ft_running:
        ft = {"status": "running", "detail": "%d/500 steps" % ft_step, "progress": round(ft_step / 5.0, 1)}
    elif fttext and has_error(fttext) and not ft_adapter:
        ft = {"status": "failed", "detail": "crashed, see log"}
    else:
        ft = {"status": "pending", "detail": "waiting"}
    r = last_reward(fttext)
    if r is not None and ft["status"] in ("running", "done"):
        ft["metric"] = "reward %s" % r
    ft.update({"icon": "\U0001F3CB️", "label": "GRPO train (500)"})
    nodes["full_train"] = ft

    # 2. sanity gate (validated once training actually runs and rewards appear)
    gate_seen = ft_step >= 1 or ft_adapter or os.path.exists(os.path.join(P2, "results", "grpo_gate", "adapter_model.safetensors"))
    nodes["sanity"] = {"icon": "\U0001F9EA", "label": "Sanity gate",
                       "status": "done" if gate_seen else ("running" if ft_running else "pending"),
                       "detail": "rewards move (validated)" if gate_seen else "waiting for first steps"}

    # 4. eval (pre/post on 5 benchmarks)
    pre_s = lm_scores(os.path.join(P2, "results", "eval_pre"))
    post_s = lm_scores(os.path.join(P2, "results", "eval_post"))
    ev_running = alive("lm_eval")

    # Live per-group progress from the eval LOG (lm-eval writes the results JSON only at pass end).
    # generate_until = GSM8K; loglikelihood = the 4 MC tasks pooled into one bar.
    elog = newest(os.path.join(LOGS, "p2_eval_*.log"))
    etext = read(elog) if elog else ""
    epass = "post" if ("== POST:" in etext and etext.rfind("== POST:") > etext.rfind("== PRE:")) else ("pre" if "== PRE:" in etext else None)
    etype = ecur = etot = eeta = None
    for ln in reversed(etext.replace("\r", "\n").split("\n")):
        m = re.search(r"Running (generate_until|loglikelihood) requests:.*?(\d+)/(\d+) \[[0-9:]+<([0-9:]+)", ln)
        if m:
            etype, ecur, etot, eeta = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            break
    epct = round(ecur / etot * 100, 1) if (ecur is not None and etot) else None
    active = {"GSM8K"} if etype == "generate_until" else (
        {"MMLU", "ARC-Challenge", "HellaSwag", "TruthfulQA"} if etype == "loglikelihood" else set())

    benchmarks = {}
    for task, name in TASKS:
        pv, qv = pre_s.get(task), post_s.get(task)
        if qv is not None:
            b = {"status": "done", "value": qv}
        elif ev_running and name in active:
            b = {"status": "running"}
            if eeta:
                b["eta"] = eeta
            if epct is not None:
                b["prog"] = epct
        elif ev_running and pv is not None:
            b = {"status": "running"}  # pre measured, awaiting post
        else:
            b = {"status": "pending"}
        if pv is not None:
            b["pre"] = pv
        if qv is not None:
            b["post"] = qv
        benchmarks[name] = b

    n_post = len(post_s)
    if n_post == len(TASKS) and pre_s:
        ev = {"status": "done", "detail": "5/5 pre + post"}
    elif ev_running:
        grp = "gsm8k gen" if etype == "generate_until" else ("4 MC tasks (shared)" if etype == "loglikelihood" else "loading")
        det = "%s pass . %s" % ((epass or "pre").upper(), grp)
        if epct is not None:
            det += " %s%%" % epct
        if eeta:
            det += " . ETA %s" % eeta
        ev = {"status": "running", "detail": det}
    elif ft["status"] == "failed":
        ev = {"status": "blocked", "detail": "train failed"}
    elif etext and ("OutOfMemory" in etext[-3000:] or "Traceback" in etext[-3000:]):
        ev = {"status": "failed", "detail": "eval crashed, see log"}
    else:
        ev = {"status": "pending", "detail": "waiting for adapter"}
    ev.update({"icon": "\U0001F9EB", "label": "Eval (5 benchmarks)", "benchmarks": benchmarks})
    nodes["eval"] = ev

    # 5. export
    if len(post_s) == len(TASKS):
        ex = {"status": "done", "detail": "metrics + summary table (%d/5)" % len(post_s)}
    elif ev["status"] == "blocked":
        ex = {"status": "blocked", "detail": "eval blocked"}
    elif len(post_s) > 0:
        ex = {"status": "running", "detail": "exporting (%d/5 post)" % len(post_s)}
    else:
        ex = {"status": "pending", "detail": "waiting for eval"}
    ex.update({"icon": "\U0001F4CA", "label": "Export metrics"})
    nodes["export"] = ex

    # 6. wire viz (inject into replay.html)
    rp = read(os.path.join(DOCS, "replay.html"))
    wired = bool(re.search(r"window\.DATA_METRICS_P2\s*=\s*\[\s*\{", rp))
    nodes["wire_viz"] = {"icon": "\U0001F3A8", "label": "Wire replay.html",
                         "status": "done" if wired else ("blocked" if ex["status"] == "blocked" else "pending"),
                         "detail": "DATA_METRICS_P2 injected" if wired else "waiting for export"}

    # 7. push (manual)
    nodes["push"] = {"icon": "\U0001F680", "label": "Commit + push",
                     "status": "manual", "detail": "run: bash git_push.sh"}

    order = ["install", "sanity", "full_train", "eval", "export", "wire_viz", "push"]
    counted = order[:-1]  # push is manual, not part of auto progress
    done = sum(1 for k in counted if nodes[k]["status"] == "done")
    return {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_ts": int(time.time()),
        "order": order,
        "overall": {"done": done, "total": len(counted), "pct": round(done / len(counted) * 100)},
        "series": parse_series(fttext),
        "nodes": nodes,
    }


def write_once(out):
    try:
        json.dump(build(), open(out, "w"), indent=1)
        return True
    except Exception as e:
        import sys
        sys.stderr.write("[status] error: %r\n" % e)
        return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between refreshes; 0 = run once")
    a = ap.parse_args()
    out = os.path.join(DOCS, "phase2_status.json")
    if a.loop > 0:
        while True:
            write_once(out)
            time.sleep(a.loop)
    else:
        write_once(out)
        d = json.load(open(out))
        print("[status] %d/%d done, %d series pts -> %s" % (d["overall"]["done"], d["overall"]["total"], len(d.get("series", [])), out))
        for k in d["order"]:
            print("  %-11s %s" % (k, d["nodes"][k]["status"]))
