---
name: readme-sync
description: Keep README.md true to the actual codebase and to docs/replay.html. Audits for outdated or redundant terminal commands, undocumented scripts, drifted results/benchmark numbers, missing plot files, and content mismatch with the replay viz. Use after adding/removing/renaming a script, CLI flag, or config key; after a phase transition (Phase 2 lands and some Phase 1 commands go stale or need relabeling); before committing or publishing; or whenever the user asks to check the README.
---

# 📗 README sync: keep the docs true to the code

Goal: the README's terminal commands, file map, results, and RL-component explanations always match the real codebase and `docs/replay.html`. Report drift grouped by severity; fix only when asked.

## When to run

- After adding, removing, or renaming a script, a CLI flag, or a `config.yaml` key.
- After a **phase transition**: Phase 2 code lands, so some Phase 1 commands get superseded, need a "Phase 1 / Phase 2" split, or become redundant.
- Before committing or publishing.
- When the user asks.

## 1. Command accuracy (the main check)

For each fenced shell block in `README.md`:

- Resolve the script/file each command invokes and confirm it EXISTS (`test -f`, `ls`). Flag references to deleted or renamed files.
- Confirm every CLI flag exists in that script's parser: `grep -oE 'add_argument\("[^"]+"' <script>`. Flag removed or renamed flags (e.g. a `--budget` that became `--timesteps`).
- Confirm the working directory is right (in this repo: `train.py` / `evaluate.py` / `benchmark.py` / `run.sh` run from `phase1_cpu/`; `make_plots.py` / `export_metrics.py` from the repo root).
- Where safe and fast, actually run the read-only ones (an import check, `--help`) to confirm they do not error.

## 2. Coverage and redundancy

- List the runnable entry points in the code: scripts with a `__main__`, `run.sh`, config knobs. Every one should appear in the README. Flag anything undocumented.
- Flag REDUNDANT or superseded commands: when Phase 2 arrives, put Phase 1 commands under a clearly labeled Phase 1 section, remove commands a newer one replaces, and mark anything deprecated. A command that still works but is no longer the recommended path is redundant.

## 3. Numbers and plot drift

- The results table and any quoted scores in the README must match the source of truth (`phase1_cpu/results/summary.md`, or `results_500k/summary.md` for the backed-up 500k, whichever the README claims). Flag stale numbers.
- Every embedded plot path (`report/plots/*.png`) and screenshot must EXIST. Flag missing or renamed files. If the underlying data changed, note that the plots need regenerating (`report/make_plots.py`).

## 4. Minimize README vs replay.html discrepancy

`docs/replay.html` is the interactive source of truth for the viz and the RL-anatomy. Cross-check these facts across README, replay.html, and `config.yaml`:

- **Benchmark list**: the 5 MinAtar games match in all three (README table, replay.html Phase 1 `GAMES`, `config.yaml games`).
- **The 6 RL components**: the README anatomy table matches the `DEFS` in replay.html (state, action, reward, transition, episode, task, per env). Wording should agree in substance.
- **Baseline numbers**: published DQN baselines match across `config.yaml`, replay.html, and README.
- **Headline claim**: "one recipe, 5 benchmarks, above random" is stated consistently.

Keep ONE source of truth per fact and have the README quote it: `config.yaml` for games and baselines, `summary.md` for scores, `replay.html` for the anatomy wording. When they drift, fix the README to match the source, unless the README is right and the code changed (then fix the code and say so).

## Output

A grouped report:

- 🔴 **Broken command**: missing file or flag. Location + fix.
- 🟠 **Undocumented / redundant**: entry point not in README, or a superseded command.
- 🟡 **Stale numbers / plots**: mismatched score or missing image.
- 🔵 **README vs replay drift**: benchmark list, anatomy wording, or baselines disagree.

Each finding: where, what is wrong, and the one-line fix. End with a verdict (in sync / minor drift / broken) and the top fixes. Apply fixes only when the user asks.

Related: [[unattended-run]] (long runs that change results, so re-run this after), [[audit]] (formatting audit for the docs you touch).
