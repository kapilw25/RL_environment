# 🧪 Phase 2 ablations: with vs without the RL environment

Here the "RL environment" = the reward source (`phase2_gpu/environment/math_env.py`, the rule-based verifier). Every row below is the SAME setup (GRPO + LoRA on Qwen2.5-0.5B, multi-skill GSM8K + ARC) except for what provides the learning signal. This isolates one claim: the gains come from the verifier reward, not from GRPO/LoRA themselves.

Two senses of "without RL env":
- **Degenerate env** (rows 5 to 6): keep the RL machinery, feed it a useless reward (constant/random). This is the true "no signal" control.
- **No env at all** (rows 8 to 9): a different paradigm (SFT/DPO) whose signal lives in a fixed dataset, so no environment is needed.

| # | 🏷️ Run | 🧬 Paradigm | 🌐 RL env? | 🎁 Learning signal | 🔍 Isolates | 🔮 Predicted vs base | 💰 Cost |
|---|---|---|---|---|---|---|---|
| 1️⃣ | ⚪ Base model | ⬛ none | ⬜ n/a | 🚫 none (no training) | 📍 the starting point | 🔵 baseline | 💤 none |
| 2️⃣ | 🏆 GRPO + verifier (correct + format) | 🟦 RLVR (ours) | ✅ YES (real) | 📏 rule: 1 if correct, + format | 🎯 the full system | 📈 GSM8K/ARC up, some transfer | 🟥 full |
| 3️⃣ | 🎯 GRPO + correctness-only | 🟦 RLVR | ✅ YES (real) | 📏 rule: 1 if correct (no format) | ❓ does format reward help? | 📈 ~= row 2, messier format | 🟥 full |
| 4️⃣ | 🔤 GRPO + format-only | 🟦 RLVR (partial) | 🟡 partial | 📐 format only, ignores correctness | ❓ is correctness the driver? | ➡️ format up, accuracy flat | 🟥 full |
| 5️⃣ | 🧊 GRPO + constant reward (=1) | ⬛ RL only, no signal | ❌ NO (degenerate) | 🟰 every completion scores 1 | 🔬 proves env is causal | ➡️ flat (advantage=0, no update) | 🟩 short |
| 6️⃣ | 🎲 GRPO + random reward | ⬛ RL only, fake | ❌ NO (noise) | 🌀 reward = random 0/1 | ❓ is ANY reward enough? | 📉 flat to down (policy drifts) | 🟩 short |
| 7️⃣ | 🤖 GRPO + reward model | 🟪 RLHF-style | ✅ YES (learned) | 🧠 a trained RM scores answers | ⚖️ verifier vs learned reward | 📈 up but gameable, needs RM | 🟥 full + RM |
| 8️⃣ | 📚 SFT on gold solutions | 🟩 supervised | ❌ NO (dataset) | 📝 copy correct answers | ⚖️ RL vs imitation | 📈 up aligned, weaker transfer | 🟨 medium |
| 9️⃣ | 🥊 DPO on preference pairs | 🟧 preference | ❌ NO (dataset) | 👍 prefer chosen over rejected | ⚖️ RL vs preference | 📈 modest up, needs pairs | 🟨 medium |

Legend: 🌐 env is ✅ real / 🟡 partial / ❌ none / ⬜ n/a . 🧬 paradigm is 🟦 RLVR / 🟪 RLHF / 🟩 SFT / 🟧 DPO / ⬛ none . 🔮 outcome is 📈 up / ➡️ flat / 📉 down / 🔵 base . 💰 cost is 🟥 full / 🟨 medium / 🟩 cheap / 💤 none

## 🎯 How to read it

- **The key contrast is 2 vs 5 vs 6:** if row 2 lifts the benchmarks, row 5 stays flat, and row 6 is flat-to-down, that is direct evidence the **verifier reward (the RL env) is the causal driver**, not GRPO or LoRA.
- **Rows 3 to 4** decompose the reward: how much comes from correctness vs formatting.
- **Rows 7 to 9** place our choice against the alternatives: a learned reward model (RLHF), pure imitation (SFT), and preference learning (DPO).
- All "Predicted" cells are hypotheses until run; the run fills the real numbers (base pre vs post, per benchmark).

## 🛠️ How to build it (cheap)

Add a `--reward-mode {verifier,correct,format,constant,random,rm}` flag to `train_grpo.py` that swaps `reward_funcs`, and add `reward_constant` / `reward_random` to `math_env.py`. Then each ablation is the same command with a different mode and `--outdir`, evaluated with `eval_harness.sh`. To keep it cheap, run the ablations at ~150 to 200 steps with a `LIMIT`-sliced eval, and only after the main baseline + full eval finish (so they do not contend for the GPU). SFT/DPO (rows 8 to 9) are separate small scripts, optional.
