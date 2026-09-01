---
name: unattended-run
description: Drive a long, unattended build/train/eval job all the way to a verified deliverable without the user babysitting it. Sets up a gated pipeline, background execution, a stall watchdog, sleep prevention, and a fix-and-relaunch self-heal loop. Use when the user says "don't stop till done", "I'm going to bed", "run overnight", "build without my guidance", or launches any multi-hour job.
---

# 🌙 Unattended run: drive a long job to completion, self-healing

Goal: take a multi-hour build / train / eval job to a verified deliverable while the user is away, catching and fixing failures without them.

## 🧠 How I actually run (be honest about this)

I am event-driven, not a background daemon. Between turns I am idle; something must WAKE me. Three things do:

- a background task finishing (success OR non-zero exit) sends a completion notification,
- a watchdog process exiting re-invokes me,
- the user sending a message.

I do NOT literally execute a check every 5 minutes myself. A separate watchdog process does the periodic checking and wakes me only on a state change (done / crashed / stalled). State this plainly to the user so expectations are right. Do not poll on a timer just to see "still fine"; that wastes turns. Wake on state change only.

## ✅ Pre-flight (before going unattended)

Do these in order, then report the safety net to the user:

1. 🔒 **Prevent sleep** (macOS): `caffeinate -dimsu -t <seconds> &`. Caveat: cannot beat a closed lid on battery, so tell the user to keep the lid open and the charger in. (Linux: `systemd-inhibit --what=idle:sleep <cmd>`, or `caffeine`.)
2. 🧱 **Make the batch resilient:** wrap each unit (per game / file / shard) in try/except so one failure logs and the batch continues. Make it resumable: skip units whose output already exists, so a relaunch only redoes what failed.
3. ▶️ **Launch in the background, logged to `logs/`:** `cd <repo> && set -o pipefail && PYTHONUNBUFFERED=1 <cmd> 2>&1 | tee logs/<name>.log` via the background-run tool; keep the task id. Create `logs/` first. `pipefail` keeps the real exit code so a crash still notifies (plain `| tee` would report exit 0 and mask it); `PYTHONUNBUFFERED=1` (or `python -u`) makes the tee'd log update live instead of block-buffering.
4. 🐕 **Arm the stall watchdog** (template below) as a second background process, in a SEPARATE turn. ⚠️ Do NOT start two background tasks in one message: the harness may reap both within seconds (observed). Launch the run alone first, confirm it survives, then launch the watchdog in its own turn, or skip the watchdog when the run is hang-safe (bounded episodes/steps) and rely on the completion/crash notification.
5. 💾 **Back up before overwriting** a good result (`cp -r out out_backup`). Finish and verify a complete deliverable BEFORE starting a riskier longer run, so the user always wakes to something done.

## 🐕 Stall watchdog template

A completion notification cannot cover a silent hang (process alive, zero progress). This closes that gap. Its EXIT re-invokes me.

```python
import glob, os, subprocess, time
WATCH_DIR = "<absolute dir the job writes files into>"
PROC = "<unique substring of the job command line>"   # e.g. "benchmark.py"
INTERVAL, STALL = 300, 1200        # check every 5 min; 20 min with no file update = stalled

def alive():
    return subprocess.run(["pgrep", "-f", PROC], capture_output=True).returncode == 0

def newest():
    fs = [f for f in glob.glob(os.path.join(WATCH_DIR, "**", "*"), recursive=True) if os.path.isfile(f)]
    return max((os.path.getmtime(f) for f in fs), default=time.time())

while True:
    if not alive():
        print("WATCHDOG_TRIGGER: process gone (done or crashed)", flush=True); break
    age = time.time() - newest()
    if age > STALL:
        print(f"WATCHDOG_TRIGGER: STALL {int(age)}s", flush=True)
        subprocess.run(["pkill", "-f", PROC]); break
    print(f"ok: alive, last progress {int(age)}s ago", flush=True)
    time.sleep(INTERVAL)
```

Set STALL well above the longest healthy quiet gap (eval phase, checkpoint save) to avoid false positives. Make sure PROC does not match the watchdog's own command line.

## 🔁 When I get woken (the self-heal loop)

1. Read the log / traceback and the output dir. Classify: complete, hard crash, per-unit failure, or stall.
2. **Complete** to run the verify / gate steps, self-clearing any review gate by doing the check myself (for a viz: browser screenshot + console read), then continue the pipeline.
3. **Failure** to diagnose from the traceback, FIX the code, relaunch ONLY the failed units (resumable design makes this cheap), re-arm the watchdog, continue.
4. **Stall** to the watchdog already killed it; inspect the last-touched unit, fix or skip it, relaunch the remainder.
5. Update the status / tracker file every wake (flip rows to done, add a one-line findings note on each fix) so the user sees progress and what happened while they slept.

## 🧯 Failure-mode coverage

| Mode | Detection | Response |
|---|---|---|
| Normal completion | task notification | run gates, continue |
| Hard crash | task exits non-zero | read traceback, fix, relaunch failed units |
| Per-unit failure | try/except logs, batch continues | fix that unit, re-run just it |
| Silent hang / stall | watchdog (5-min check, kills on 20-min stall) | diagnose, fix, relaunch remainder |
| Machine sleep | caffeinate + user keeps lid open / on power | physical, cannot be fixed in software |

## 🚧 Guardrails (hold these even when told "don't stop")

- Never do prohibited or irreversible outward actions autonomously: git push, sending messages, publishing, deleting data, spending. Honor standing preferences (for this user: they push git themselves; never use em-dashes).
- Only run the job that was authorized. Do NOT launch an optional heavy run (longer budget, extra seeds) unless the user opted in. Ask before they sleep if unsure.
- Self-clearing a gate means doing the verification yourself and recording it, not skipping it.
- If a decision genuinely needs the user (ambiguous requirement, destructive tradeoff), stop and leave a clear question rather than guessing. Surface it before they sleep when possible.
- Leave the working tree for the user to commit and push; write a clear final summary of what completed, what failed, and every fix applied.

## 📋 Reusable one-liners

- Sleep guard: `caffeinate -dimsu -t 39600 &`  (about 11 hours)
- Launch (logged): background-run `cd <repo> && set -o pipefail && PYTHONUNBUFFERED=1 <cmd> 2>&1 | tee logs/<name>.log`
- Watchdog: background-run `python3 watchdog.py`
- Backup before a risky rerun: `cp -r out out_backup`

Related: gate the pipeline so failures are caught cheaply before the expensive run (see the tracker's review-gate pattern). See also [[audit]] for the pre-write formatting check.
