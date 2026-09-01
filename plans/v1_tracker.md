# 🗂️ Phase 1 (CPU) Build Tracker

> Living checklist for the Phase 1 pipeline (custom env, one PPO recipe, 5 MinAtar benchmarks).
> **Last updated:** 2026-08-31. **Now:** Phase 1 ✅ DONE (2M, 15/15; final % of DQN: Breakout 78, Asterix 76, SpaceInvaders 51, Freeway 44, Seaquest 16). **Phase 2 built + M1-smoke-tested end-to-end** (multi-skill GSM8K+ARC env, GRPO train on CPU, MPS inference, lm-eval on CPU, exporter + P2 viz wiring). The full 500-step run needs the RTX box. README has Phase 2 + M1-smoke commands.

## 🎨 Status legend

| Emoji | Status | Meaning |
|---|---|---|
| ⚪ | **Pending** | queued, not started yet |
| 🔵 | **Running** | in progress right now |
| 🟢 | **Completed** | done and verified |
| 🟡 | **In review** | sitting at a human gate, awaiting eyeball |
| 🔴 | **Blocked** | waiting on a fix or dependency |
| ⏭️ | **Optional** | deferred, nice-to-have |

**Type icons:** 🧩 code (build a file) . ▶️ run (execute) . 🚦 gate (human review)

## 📊 Progress at a glance

> **Completed 19 / 19 core tasks (100%)**  `▰▰▰▰▰▰▰▰▰▰`
>
> 🟢 19 core done + E2 (2M push) done . 🔵 0 running . ⚪ 0 pending . 🔴 0 blocked

## 🚦 The 3 review gates

Cheap checkpoints so a silent bug never wastes the multi-hour run.

| Gate | When | What you eyeball (about 1 min) | Cost if skipped | Status |
|---|---|---|---|---|
| 🚦 **G1** env-check | after Stage B | `check_env` passes, obs shape (10,10,C), action space right | hours of garbage runs | 🟢 Completed |
| 🚦 **G2** smoke-train | after Stage C | 50k-step run finishes, reward trends up, `evaluations.npz` written | full run silently broken | 🟢 Completed |
| 🚦 **G3** viz-wire | after Stage F | real curves render in `replay.html` | wrong data shipped | 🟢 Completed |

## 📋 Phase 1 TODO

### 🏗️ Stage A: Scaffold and setup

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| A1 | 🟢 Completed | `requirements.txt` (sb3, gymnasium, minatar, tbparse, tensorboard, matplotlib, pyyaml) | 🧩 code | deps pinned |
| A2 | 🟢 Completed | venv + `pip install` + import sanity (sb3 2.9.0, gymnasium 1.3.0, minatar ok) | ▶️ run | working env |
| A3 | 🟢 Completed | `config.yaml`: the ONE recipe (lr, n_steps, timesteps, eval_freq, save_freq, seeds, games) | 🧩 code | single source of truth |
| A4 | 🟢 Completed | `.gitignore` + `results/.gitkeep` | 🧩 code | clean repo |

### 🌍 Stage B: Build environments  (ends at Gate G1)

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| B1 | 🟢 Completed | `envs/gridworld_env.py` (custom Env + register `GridWorld-v0`) | 🧩 code | the env you "build" |
| B2 | 🟢 Completed | `envs/minatar_cnn.py` (tiny `BaseFeaturesExtractor` for 10x10xC) + `envs/__init__.py` helpers | 🧩 code | make-or-break piece |
| B3 | 🟢 Completed | 🚦 **G1**: `check_env` on GridWorld (clean) + MinAtar Breakout (obs 4x10x10) + random rollout | 🚦 gate | correctness proof |

### 🎛️ Stage C: Training recipe  (ends at Gate G2)

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| C1 | 🟢 Completed | `train.py` (`--env-id`; MLP / CNN by obs; logger csv+json+tb; `EvalCallback`; `CheckpointCallback`; seed) | 🧩 code | the recipe |
| C2 | 🟢 Completed | smoke: GridWorld learns (0.93 return, optimal 8-step path; needed 100k not 20k) | ▶️ run | env learns |
| C3 | 🟢 Completed | smoke: Breakout 50k in 27s, ours 3.63 vs random 0.40 (9x), npz shape (5,20) | ▶️ run | pipeline runs |
| C4 | 🟢 Completed | 🚦 **G2**: reward up + `evaluations.npz` + `progress.csv` + model zips present | 🚦 gate | go / no-go passed |

### 📏 Stage D: Evaluate and benchmark

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| D1 | 🟢 Completed | `evaluate.py` (N greedy episodes mean±std + random baseline) | 🧩 code | scoring |
| D2 | 🟢 Completed | `benchmark.py` (loop 5 games x seeds; `results.csv` + `summary.md` + per-game curve PNG) | 🧩 code | the Phase 1 result |

### ⏱️ Stage E: Full training runs

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| E1 | 🟢 Completed | 500k x 5 games x 3 seeds. All above random: Breakout 13.0, Freeway 20.8, SpaceInvaders 49.7, Asterix 2.4, Seaquest 1.3 | ▶️ run | 5-row table + curves |
| E2 | 🟢 Completed | 2M x 5 games x 3 seeds. Final (% of DQN): Breakout 21.1 (78%), Freeway 24.3 (44%), Asterix 10.3 (76%), Seaquest 6.0 (16%), SpaceInvaders 95.0 (51%). All up from 500k. Re-plotted (+ Figure E), re-exported (3000 rows), re-reported | ▶️ run | stronger numbers ✓ |

### 📤 Stage F: Export and wire the viz  (ends at Gate G3)

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| F1 | 🟢 Completed | `report/export_metrics.py` (evaluations.npz to tidy `metrics.json` schema; validated against real data at F2) | 🧩 code | feeds the viz |
| F2 | 🟢 Completed | exported 750 rows (aligned) to `metrics_phase1.json`, injected as `window.DATA_METRICS_P1` into replay.html | ▶️ run | real animation |
| F3 | 🟢 Completed | 🚦 **G3**: real curves render (x-axis 500k, seed bands, Freeway ~94% noisy, Breakout ~42%), zero console errors | 🚦 gate | viz verified |

### 📝 Stage G: Report and final verify

| # | Status | Task | Type | Output |
|---|---|---|---|---|
| G1 | 🟢 Completed | `report_phase1.md` written (5-row table, curves, compute, honest caveats) | 🧩 code | deliverable |
| G2 | 🟢 Completed | verified: our >> random on all 5, curves rising, replay animates real data | ▶️ run | headline done |

## 🔁 Runbook order (how they chain)

> 🏗️ A setup ➜ 🌍 B build env ➜ 🚦 **G1** ➜ 🎛️ C recipe + smoke ➜ 🚦 **G2 (STOP)** ➜ ⏱️ E full run ➜ 📤 F export ➜ 🚦 **G3 (STOP)** ➜ 📝 G report

`phase1_cpu/run.sh` (`set -e`) chains everything and pauses at G2 and G3. Pass `--yes` to skip pauses, `--budget=full` for the 5M run.

**Overnight plan (autonomous):** E1 500k ➜ F2/F3/G1/G2 (self-verified) ➜ back up 500k ➜ E2 ~2M push ➜ re-export + re-report.

## 🗒️ Notes and findings (2026-08-30)

- 🔧 **MinAtar + gymnasium 1.3.0:** `import minatar` does NOT auto-register envs anymore. Fix: call `minatar.gym.register_envs()` once (handled in `envs/_ensure_minatar`). Ids are `MinAtar/<Name>-v1`.
- 🔧 **CNN wiring:** obs are `(10,10,4)` bool. `MinatarChannelFirst` wrapper transposes to `(4,10,10)` float32; `MlpPolicy` + custom `MinatarCNN` extractor with `normalize_images=False` (avoids SB3 image-space checks). No NatureCNN failure.
- 🔧 **GridWorld budget:** 20k steps is only ~10 PPO updates and collapses under greedy eval (return -1.0). 100k learns the optimal path (0.93). `config.timesteps_smoke_grid` bumped to 100k.
- ⚠️ **check_env note:** the third-party `minatar.gym` reset seeds its own RNG, so `check_env` warns about `super().reset(seed=...)`. Benign; PPO unaffected.
- 📁 **Extra file vs plan tree:** added `envs/__init__.py` (shared `make_env` / config / wrappers) so train, evaluate, benchmark build envs identically.
- ⏱️ **Timing:** Breakout 50k = 27s on the M1, so 500k ≈ 4.5 min/run, ~75-90 min for 15 runs.
- 🧬 **Bonus anatomy viz:** added a live "Anatomy of the RL environment" section to `docs/replay.html` (6 parts: state, action, reward, transition, episode, task; tabs for GridWorld live / MinAtar / LLM). Verified in-browser, zero console errors. Independent of the F2 data injection.
- 🪵 **Logs:** each new background process now tees to `logs/<name>.log` via `set -o pipefail && PYTHONUNBUFFERED=1 <cmd> 2>&1 | tee logs/<name>.log` (pipefail keeps the real exit code so crash notifications still fire; unbuffered makes the log update live). Live resume log at `logs/benchmark_500k_resume.log`, watchdog at `logs/watchdog.log`.
- 🐛 **Eval hang (caught + fixed by the self-heal loop):** MinAtar's gym env has NO episode step limit, so a deterministic policy on Seaquest produced a non-terminating episode that hung `EvalCallback`. The watchdog detected the ~21 min stall, killed the run, and re-invoked me. Fix: `gym.make(env_id, max_episode_steps=1000)` for MinAtar (guarantees termination; verified Seaquest truncates at 1000). Also made `benchmark.py` resumable (skips games/seeds with a saved `final_model.zip`), so relaunches reuse completed runs. Post-fix, Seaquest finished cleanly (~1.3).
- ⏹️ **Background-task reaping (diagnosed):** every time benchmark + watchdog were launched in the SAME assistant message (two background tasks at once), BOTH were killed within seconds (happened twice). Launched ALONE they survive (the finish-SpaceInvaders run and the 2M run both ran solo). Rule: do not start two background tasks in one turn. Fix applied: launch the training run solo; add any watchdog in a separate turn, or skip it when the run is hang-safe (as here, post-TimeLimit). The 500k deliverable was never at risk (backed up).
- 📊 **Final 500k scores (mean of 3 seeds vs random / DQN):** Breakout 13.0 / 0.6 / 27.1, Freeway 20.8 / 0.1 / 55.9, SpaceInvaders 49.7 / 3.7 / 188, Asterix 2.4 / 0.3 / 13.6, Seaquest 1.3 / 0.2 / 38. All above random.
- 📈 **Eyeball plots:** `report/make_plots.py` writes 4 PNGs to `report/plots/` from the stable 500k data: (A) all-5 learning curves as % of DQN, (B) per-game curves vs random/DQN, (C) final grouped bars (log scale), (D) improvement over random (Freeway 160x, Breakout 23x, SpaceInvaders 14x, Asterix 7x, Seaquest 6x). Fixed an edge-smoothing artifact (windowed mean with a shrinking edge window). Plots read the frozen `results_500k/`, so they stay unchanged until the 2M sweep completes; then regenerate + add a 500k-vs-2M comparison (Figure E).
- 📗 **Docs:** wrote a full `README.md` (embeds plots A + D, a 6-part RL-anatomy table mirroring replay.html, and terminal commands to explore the codebase + run experiments + generate plots). Added a `readme-sync` skill to keep README commands current with the code and minimize README-vs-replay.html drift.
- ⚖️ **Phase comparison table:** added a static "Phase 1 vs Phase 2 (CNN vs LLM)" card to `docs/replay.html` (11 rows: policy, env, state, action, reward, algorithm, hardware, compute, benchmarks, signal, status). Verified at the final refresh.
- 🗂️ **3-tab restructure:** `docs/replay.html` now has 3 top-level tabs (Shared infra / Phase 1 RL agent / Phase 2 LLM GRPO); verified in-browser, zero console errors.
- 🎬 **Animations + GIFs:** the live `replay.html` now auto-plays and loops each run in ~9s (`BASE_MS` 45, loop-on-end, anatomy interval 350ms). `report/make_gifs.py` renders two looping GIFs (loop:0, 100 frames, ~10s) to `report/gifs/`: `learning_curves.gif` (curves growing) and `benchmark_race.gif` (bar-chart race). Screen-recording the live page was not viable (the automation pauses requestAnimationFrame), so the GIFs are matplotlib renders of the same 2M data. **Race fix:** the noisy 2M eval series made the race bars jitter (never settling), so the GIF series is now smoothed (windowed mean), giving a clean 10s race where Asterix overtakes Breakout to finish first (82% vs 77%). Also reduced the live race d3 transitions 240 to 90ms so the on-page race tracks the scrubber and completes in sync. Both GIFs verified at exactly 10,000ms (100 frames x 100ms), loop:0.
- 🚀 **Phase 2 built + M1-tested (2026-08-31):** `phase2_gpu/` = TRL `GRPOTrainer` + LoRA on Qwen2.5-0.5B; **multi-skill verifiable env** (`math_env.py`: GSM8K math + ARC science, rule-based verifier rewards); `train_grpo.py` (`--smoke` = tiny CPU run), `eval_harness.sh` (lm-eval base vs adapter on the 5 tasks), gated `run.sh`, `run_smoke.sh` (M1 end-to-end). `report/export_metrics_phase2.py` turns lm-eval JSONs into `docs/metrics_phase2.json`; `replay.html` reads `window.DATA_METRICS_P2` (P2 tab stays mock until a real run). **Validated end-to-end on the M1:** env + rewards (`[1,1,0]`), GRPO smoke (35MB adapter saved), MPS inference (Qwen generates on `mps:0` in 2.9s), lm-eval pre/post (CPU), exporter (valid GSM8K rows). **M1 limits:** vLLM is CUDA-only, bf16 flaky on MPS, lm-eval segfaults on MPS (used CPU), so the real 500-step run needs the RTX box. Chose TRL over the verifiers lib; trl 1.12 dropped `max_prompt_length`.
- ⚡ **2M throughput tweak + gains:** each 2M run (~18 min) was borderline vs the ~15-20 min reap window, so lowered `n_eval_episodes` 20 to 5 (keeps the 200-point curve grid; results table still uses a separate 30-episode eval). Final gains (500k to 2M): Breakout 13.0 to 21.1 (48% to 78% of DQN), Freeway 20.8 to 24.3, Asterix 2.4 to 10.3 (18% to 76%), Seaquest 1.3 to 6.0 (4.6x), SpaceInvaders 49.7 to 95.0 (nearly doubled). Every game improved; Asterix and Seaquest gained the most. A stray duplicate benchmark process was killed at the end (a relaunch had overlapped a still-running one; results were clean, no corruption). Figure E (`report/plots/E_500k_vs_2M.png`) shows the 500k-vs-2M gain.
