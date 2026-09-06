---
name: emojies
description: Add colorful, distinct-shape emojis to tables, mermaid/flowchart diagram nodes, checklists, and status/comparison grids in ANY document (.md, .html, .txt, and similar) so a reader can eyeball status, category, and outcome at a glance. Use whenever you create or modify a table, a mermaid or other diagram, a checklist, or a comparison/status matrix in a document file, or when the user asks to make one "colorful", "add emojis", "different shapes", or "easy to eyeball". Provides one consistent shape-and-color emoji language plus a required legend, and stays in house style (no em-dashes, no --- horizontal rules).
---

# 🎨 Emoji eyeballing for tables and diagrams

Goal: make every table cell and diagram node carry a leading emoji chosen so the COLUMN or CATEGORY is decodable by shape and color alone, then add a one-line legend. Reuse one code across the whole document.

## 🟢 When to use

- Any time you write or edit a **table**, **mermaid / flowchart diagram**, **checklist**, or **status / comparison grid** in `.md`, `.html`, `.txt`, or similar document files.
- When the user asks to make something "colorful", "add emojis", "different shapes", or "easy to eyeball / scan".

## 📏 The rule

- Put ONE emoji (two max) at the **start** of each cell or node label, never trailing, so columns stay aligned.
- Pick emojis so each COLUMN uses its own shape-and-color family; a reader should decode a column by glyph alone.
- Keep the mapping **consistent** across the whole document (same meaning to same emoji).
- Always finish with a compact **legend** line decoding each emoji family.

## 🧩 Palettes (reuse these consistently)

- **Status / yes-no:** ✅ yes/done . 🟡 partial/warn . ❌ no/fail . ⬜ n/a . 🔵 running/info . ⏳ pending . 🚫 none
- **Category (colored squares):** 🟦 🟩 🟪 🟧 🟥 ⬛ (assign one square per class; keep it stable per class)
- **Outcome / trend:** 📈 up . ➡️ flat . 📉 down . 🔵 baseline
- **Magnitude / cost / risk:** 🟥 high . 🟨 medium . 🟩 low . 💤 none
- **Rank / order:** 1️⃣ 2️⃣ 3️⃣ ... or 🥇 🥈 🥉
- **Row identity:** give each row a unique thematic glyph (🏆 🎯 🧊 🎲 🤖 📚 🥊 🧪 📦 🚀 ...) so rows are scannable too.

## 🕸️ Mermaid and other diagrams

- Prefix every node label with a state emoji (✅ 🔵 ⏳ ❌ ...).
- Color-code with `classDef`, matching the emoji color, e.g. `classDef ok fill:#10b981,color:#04121e;` then `class N ok;` on a node labeled with ✅.
- Keep shapes meaningful: rectangle = process, diamond = decision, cylinder = data/store, stadium = start/end.
- One state emoji per node minimum.

## ✅ House style (must)

- **No em-dashes** (the `—` character or `---` used as a rule/thematic break) anywhere in the content; use `-`, `.`, or the word "to". YAML frontmatter `---` and markdown table separators `|---|` are fine. See the [[audit]] skill and the block_em_dash hook.
- Prefer emoji headings and blockquotes over horizontal rules.
- Do not overload: 1 emoji per cell (2 max), consistent per column; the goal is scanning, not decoration.

## 🔎 Tiny example

```
| # | 🌐 Env? | 🔮 Outcome | 💰 Cost |
|---|---|---|---|
| 1️⃣ | ✅ real | 📈 up | 🟥 high |
| 2️⃣ | ❌ none | ➡️ flat | 🟩 low |

Legend: 🌐 ✅ real / ❌ none . 🔮 📈 up / ➡️ flat . 💰 🟥 high / 🟩 low
```

Related: [[audit]] (checks the same no-em-dash / divider house style before a Write/Edit lands).
