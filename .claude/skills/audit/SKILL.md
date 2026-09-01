---
name: audit
description: Audit Markdown / HTML / LaTeX output for constructs that trip the block_em_dash PreToolUse hook (em-dashes and their HTML entities) and for house-style dividers. Confirms emoji headings + blockquotes are used instead of raw rules, so a Write/Edit passes on the first try. Audit only; fix only when asked.
allowed-tools: Read, Grep, Glob, Edit
argument-hint: <file-path, dir, or glob; default = the file just written/edited>
---

# Formatting audit: em-dashes, dashes, and dividers

Goal: make any file we write pass the deterministic `block_em_dash` hook on the FIRST attempt, and keep a consistent house style for section dividers. Report findings (location, why, safe rewrite). Do not edit unless the user asks.

Why this exists: a PreToolUse hook (`block_em_dash.py`, in `robot_survive_bench/.claude/hooks/`) rejects any `Edit` / `Write` / `MultiEdit` / `NotebookEdit` that would place an em-dash into a file. A rejection costs a wasted round-trip, so catch these constructs BEFORE they are written.

## 🚫 What the hook blocks (hard rules)

| Construct | Blocked in | Note |
|---|---|---|
| Unicode em-dash `U+2014` and horizontal bar `U+2015` | ANY file | the long dash char and its twin |
| LaTeX em-dash `---` (exactly 3 hyphens) | `.tex` `.sty` `.cls` | a longer run `----` stays legal |
| HTML em-dash entities (`mdash`, `#8212`, `x2014` forms) | `.html` `.htm` `.md` `.markdown` | the ampersand + name + semicolon forms |

## ✅ Explicitly allowed (do NOT flag)

- en-dash ranges: `--` or the Unicode en-dash `U+2013`, e.g. `2016--2026`, `Tables 1--3`
- CLI flags: `--env-id`, `--code-only`
- Markdown / YAML `---` front-matter fences and horizontal rules (in `.md` only)
- longer hyphen separators: `----`, `% ------`

> 💡 **Key nuance:** `---` is only a hook problem in LaTeX. In Markdown it is fine by the hook. We avoid it as a house-style choice, not because the hook blocks it. Do not report `---` in a `.md` file as a blocker.

## 🎨 House style for dividers (soft rule)

Prefer, in this order:

1. Emoji section headings (`# 🚀 Phase 2`, `## ✅ Verification`)
2. Blockquote callouts (`> 💡 note`, `> ⚠️ caveat`) as visual separators
3. Blank lines

Avoid raw `---` horizontal rules across all formats so Markdown and LaTeX read the same and content never trips the LaTeX branch of the hook when ported between them.

## 🔁 Safe rewrites for an em-dash

Replace the long dash with, in rough priority:

- a colon (`:`) when introducing or expanding a clause
- a comma pair or parentheses for an aside
- the word "with" / "to" for a spoken range
- two shorter sentences

## 🔎 Audit procedure

1. Resolve the target: the argument, else the file just written / edited.
2. Scan (character checks are case-sensitive):
   - any file: `grep -nP "\x{2014}|\x{2015}" FILE` for the Unicode em-dash / horizontal bar
   - `.tex` / `.sty` / `.cls`: `grep -nP "(?<!-)---(?!-)" FILE` for the LaTeX em-dash
   - `.md` / `.html`: grep for the em-dash HTML entities (the `mdash` / `#8212` / `x2014` forms)
   - `.md`: list any `---` horizontal-rule lines as house-style (soft) findings only
3. Classify each dash: em-dash (block), en-range (ok), hyphen-compound (ok), stray hyphen standing in for a range (flag).

## 📤 Output

Grouped report:

- 🚫 **Hook blockers** (must fix before Write): location + construct + safe rewrite
- 🎨 **House style** (optional): raw `---` rules to convert to emoji headings / blockquotes
- **Verdict:** clean / minor / blockers-present, plus the exact fixes.

Audit only. Apply fixes only when the user asks.
