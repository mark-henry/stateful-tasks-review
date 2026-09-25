# SOURCING.md — grid_navigation (BBH "navigate")

Phase 1, sourcing only. No task.py / generator code written per SPEC.md amendment.

## What this task is

BBH's `navigate`: given a sequence of turn/step instructions, decide whether the walker
returns to the origin. Canonical answer space is binary (Yes/No). The assignment for
`grid_navigation` specifically asks for a **richer, non-binary variant**: state = (x, y,
heading) tracked at every step, answer = final position as a string `"x,y"` (or similar),
not just yes/no. That richer framing is *not* the BBH task as published anywhere I found —
see recommendation at the bottom.

## Candidates found and vendored

### 1. `vendor/BIG-Bench-Hard/` (already present before this pass)
- Source: github.com/suzgunmirac/BIG-Bench-Hard, commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d`, MIT license.
- Shared/symlinked with sibling tasks `dyck` and `boolean_expressions` — not re-cloned.
- Contains:
  - `bbh/navigate.json` — 250 fixed examples, `{input, target}` pairs, target ∈ {"Yes","No"}. Canary-tagged (contamination-flagged).
  - `cot-prompts/navigate.txt` — 3-shot CoT prompt. This is the exact scratchpad style SPEC.md wants replicated: numbered steps `(k) <instruction>: (x, y), facing <axis>.`, ending in `Since (x,y) is/is not (0,0), ... So the answer is Yes/No.`
  - `code-davinci-002-outputs/code-davinci-002-{direct,cot}/navigate_..._eval_metrics.jsonl` — **primary-source published accuracy numbers** for Codex (code-davinci-002) with and without CoT, used directly in desk.json.
- Fixed dataset only, no generator. No reference solver code (just model outputs). CoT format present and matches SPEC's rigid-scratchpad requirement closely, but answer is Yes/No, not (x,y).

### 2. `vendor/google-BIG-bench-navigate/` (newly vendored, sparse checkout of `bigbench/benchmark_tasks/navigate/`)
- Source: github.com/google/BIG-bench, commit `092b196c1f8f14a54bbc62f24759d43bde46dd3b` (main, fetched 2026-09-11), Apache 2.0 license.
- Contains:
  - `README.md` — task description by authors Eric Chu (MIT) and Sneha Priscilla Makini (Google). States the 1000-example set was **"generated synthetically by a script"**, explicitly analogized to SCAN, "a toy task programmatically created." However, **the generator script itself is not in this directory** — only the already-materialized `task.json` output is checked in. I did not find `generate_examples.py` or equivalent anywhere in the BIG-bench repo for this task (checked the directory listing directly; only README.md, task.json, results/ are present). So: procedurally-generated *in spirit*, but the procedure/code is not published/recoverable from this repo.
  - `task.json` (308KB) — 1000 examples total, two `inst_type`s: `"turns"` (relative turning, matches BBH's `navigate.json` style) and `"face_forward"` (always-forward, directional steps — this is actually a *third* of BBH's examples, since BBH sampled 250 from this same underlying pool). Scored as `multiple_choice_grade` (True/False), still binary.
  - `results/scores_*.json` — published per-model multiple_choice_grade scores at 0/1/2/5-shot (no CoT — this predates the BBH CoT-prompting paper) for GPT-3 family, Gopher family, PaLM (8b/64b/535b), BIG-G/BIG-G-sparse family. E.g. PaLM 535B: 0-shot 0.517, 1-shot 0.558, 2-shot 0.56, 5-shot 0.553 (all barely above the 0.5 random baseline — this is the *original* BIG-bench no-CoT eval, distinct from BBH's CoT-prompting eval of the same task).
- Same binary answer space as BBH. No richer state variant. No solver code, only eval scores.

### 3. `vendor/lm-evaluation-harness-bbh/` (newly vendored, sparse checkout of `lm_eval/tasks/bbh/`)
- Source: github.com/EleutherAI/lm-evaluation-harness, commit `ad8737ae7fad24cf64e50fc7fc31397bff586b9e` (main, fetched 2026-09-11), MIT license.
- Contains YAML task configs for `navigate` in four modes: `fewshot/`, `zeroshot/`, `cot_fewshot/`, `cot_zeroshot/`. These are **harness glue, not new data** — `dataset_name: navigate` points at the `lukaemon/bbh` HF dataset (i.e. the same BBH 250 examples), with `doc_to_text` templates and regex answer-extraction filters (`\b(Yes|No|yes|no)\b`). The `cot_fewshot/navigate.yaml` embeds the identical 3-shot CoT prompt seen in BBH's `cot-prompts/navigate.txt`.
- No generator, no new examples, no solver beyond regex-match grading. Confirms BBH's CoT prompt format is the field-standard scratchpad for this task. Still binary answer space.

### 4. `vendor/hf-mirrors/` (newly vendored — small JSON samples only, not full dataset dumps)
- `lukaemon-bbh-navigate-first-rows.json` — first 100 rows of `lukaemon/bbh`, config `navigate`, split `test`, fetched via HF datasets-server API 2026-09-11. Byte-identical in content/format to `BIG-Bench-Hard/bbh/navigate.json`'s first example (verified by direct comparison) — confirmed to be a straight re-export of BBH, not independent data. License: BBH's MIT (HF card defers to source).
- `maveriq-bigbenchhard-navigate-first-rows.json` — same check against `maveriq/bigbenchhard`, config `navigate` (this dataset has all 27 original BIG-bench task names — a superset packaging of BBH, not a 27-*additional*-variant collection). First row also byte-identical to BBH. Note: as of recent `datasets` library versions this repo's loading script is broken (`RuntimeError: Dataset scripts are no longer supported`) per its own HF discussion thread — another reason not to depend on it.
- I did not do a full download of either HF dataset since the first-100-rows sample already establishes they are re-exports with no unique content; fetching the rest would just duplicate `bbh/navigate.json` already vendored under BIG-Bench-Hard.

## Candidates searched for but NOT found / NOT vendored

- **"Liu et al. 2022" gridworld/compositional-generalization paper with a navigation generator** — I searched extensively (multiple query variants: "gridworld language model compositional generalization", "Liu 2022 SCAN gSCAN", "Liu 2022 COUNTERFACTUAL compositional generalization") and could not identify a paper matching this description authored by a first-author "Liu" in 2022 that ships a navigate-style generator. The closest adjacent line of work is **gSCAN** (Ruis et al., 2020, "A Benchmark for Systematic Generalization in Grounded Language Understanding," NeurIPS) and follow-ups (Ruis & Lake 2022; Hein & Diepold 2022) — these are grid-world *language-conditioned action-sequence* benchmarks (agent must execute a described action in a grid with objects), not turn/step dead-reckoning tasks, and their answer format is an action trajectory, not a final coordinate. I judged these to be a different enough task family (grounded instruction-following vs. pure spatial-arithmetic tracking) that vendoring the gSCAN repo would not usefully shortcut grid_navigation, so I did not clone it. Flagging this explicitly in case "Liu 2022" refers to a specific paper I'm not finding — happy to vendor gSCAN or something more specific on request.
- **A programmatic generator for BBH/BIG-bench's navigate specifically** — none exists in either google/BIG-bench or suzgunmirac/BIG-Bench-Hard. Both ship only the materialized JSON.
- **Any published implementation of the richer (x,y,heading)-as-answer variant** — none found anywhere. Every implementation I found (BIG-bench, BBH, lm-eval-harness, both HF mirrors) treats this as a binary yes/no classification task. Nobody publishes it with the final coordinate as the target answer.

## Fit against the common interface (SPEC.md)

| requirement | status |
|---|---|
| `Instance.steps` (one line per step) | BBH's CoT format already does this via numbered `(k) ...` lines — directly adaptable |
| `Instance.states` (tracked state per step) | Present in the CoT text as `(x, y), facing <axis>` — would need parsing out of the vendored prompt strings, not structured data |
| `Instance.answer` exact-match | BBH/BIG-bench answer is `Yes`/`No` only. The assignment wants final `"x,y"` position. **No vendored source provides this** — must be derived from scratch by re-deriving the final coordinate from each instruction sequence (trivial to compute, since we'd be writing generate() and solve() anyway) |
| generator with seed, non-contaminated instances | Not present in any candidate — BIG-bench's own generator script is not published, only its output. A generator would need to be written from scratch, reverse-engineering the instruction-sampling scheme from the README's description ("turns" vs "face_forward" types, sub-instruction counts 3–10, balanced Yes/No) |
| reference solver | Not present anywhere (BBH/BIG-bench ship model outputs, not a ground-truth solver) — trivial to write independently (this is a simple prefix-sum over displacement vectors, see desk.json theory_class) |
| published baseline numbers | Yes — see desk.json `published_data`, sourced directly from vendored eval_metrics.jsonl and results/scores_*.json files (primary source, not paper-table transcription) |

## Recommendation

**Nothing usable as-is; usable with a thin wrapper only for the canonical binary BBH task, but the assignment's richer (x,y,heading) variant must be written from scratch.** For the plain binary "return to origin?" task, BBH's `navigate.json` + `cot-prompts/navigate.txt` are essentially ready to wrap (parse the CoT prompt's step lines into `Instance.steps`/`states`, treat Yes/No as `answer`) — that's a thin-wrapper job, not new data. But that is *not* what this assignment asked for: the assignment explicitly wants the answer space to be the final position "x,y" rather than yes/no, and no vendored or found candidate — BIG-bench, BBH, lm-evaluation-harness, HF mirrors, or the searched-for "Liu 2022" line of work — publishes or generates that richer variant anywhere. The generator, the (x,y,heading)-tracking solve(), and the non-binary answer format all have to be written from scratch; only the CoT scratchpad *style* (numbered step lines with position+heading, borrowed directly from BBH's cot-prompts/navigate.txt) and the underlying instruction vocabulary (turn left/right/around, take N steps, forward/backward/left/right variants) can be reused as design reference from the vendored material.
