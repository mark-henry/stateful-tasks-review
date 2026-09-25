# SOURCING — cup_shuffling (BBH tracking_shuffled_objects)

Phase 1 sourcing only, per SPEC.md AMENDMENT. No task.py / generator written yet.

## Candidates vendored

### 1. `vendor/bbh/` — Suzgun et al. 2022, BIG-Bench-Hard (canonical)
- Source: https://github.com/suzgunmirac/BIG-Bench-Hard, commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d` (2022-10-22), MIT license (see `vendor/bbh/LICENSE`).
- Contents vendored: `data/tracking_shuffled_objects_{three,five,seven}_objects.json` (the fixed eval sets, JSON `{input, target}` pairs, multiple-choice), and `cot-prompts/tracking_shuffled_objects_{three,five,seven}_objects.txt` (the 3-shot CoT prompt prefix used in the paper, with a rigid numbered scratchpad: `(0) At the start: ...`, `(1) X and Y swap Z: ...`, ..., `So the answer is (X).`).
- **What it contains**: a fixed dataset (not a generator — no data-generation script is in this repo, just the outputs) plus the exact CoT few-shot prompts used to produce Table 3's numbers. No reference solver.
- **How close to common interface**: the CoT prompt format is close to spec's rigid-scratchpad requirement — one line per swap, state visible at a fixed position (`Alice: X, Bob: Y, Claire: Z`). But it's a **fixed, contamination-risked dataset** (canary GUID embedded, published on GitHub/HF since 2022) with only 3/5/7-object instances and no seed/knob control. It has multiple-choice targets `(A)/(B)/(C)`, not free-form state.
- **Missing for common interface**: `generate(depth, seed)`, `solve()` (independent reference solver), `check()`, `step_spans()`, and a KNOBS-controllable depth (BBH depth = N-1 swaps always, fixed at N=3/5/7 — no independent depth knob separate from object count).

### 2. `vendor/bigbench/` — google/BIG-bench original task (same underlying data)
- Source: https://github.com/google/BIG-bench, commit `092b196c1f8f14a54bbc62f24759d43bde46dd3b`, subpath `bigbench/benchmark_tasks/tracking_shuffled_objects`.
- Contents vendored: `README.md` (detailed task spec, generation *description* — "a .py script (not included but available upon request) to swap in objects and names") and `task.json` (task metadata + canary; the actual per-N examples are the same content as BBH's `data/*.json`, since BBH's task JSONs were themselves extracted from BIG-bench).
- **What it contains**: this is the *origin* of the BBH dataset — same fixed instances, no generator script actually included (explicitly noted "not included but available upon request" in the README), no reference solver. Useful only for the human-rater/design-rationale documentation (swap-selection constraints: every person swaps at least once, no two consecutive swaps between the same pair — worth replicating in our generator for realism).
- **Recommendation weight**: nothing new beyond BBH; keep for the design-constraint documentation only.

### 3. `vendor/lm-evaluation-harness/` — EleutherAI harness YAML task configs
- Source: https://github.com/EleutherAI/lm-evaluation-harness, commit recorded in `vendor/lm-evaluation-harness/COMMIT.txt`.
- Contents vendored: `bbh/cot_fewshot/tracking_shuffled_objects_{three,five,seven}_objects.yaml` (CoT prompt as YAML, includes the same 3-shot scratchpad demonstrations as the BBH repo, machine-readable), `bbh/cot_zeroshot/*.yaml`, and `bigbench/{multiple_choice,generate_until}/tracking_shuffled_objects.yaml` (harness wrapper around the original BIG-bench task, defines `doc_to_target`/answer parsing regex worth reusing for `check()`).
- **What it contains**: harness *wrappers*, not a generator or solver. Confirms the exact answer-parsing convention (`So the answer is (X).`, regex-extracted).
- **Recommendation weight**: useful as a second, independently-maintained copy of the CoT prompt/scratchpad format (cross-check against BBH's own copy) and for the answer-extraction regex. No generator.

### Considered but not vendored
- HuggingFace mirrors (`lukaemon/bbh`, `maveriq/bigbenchhard`, `lighteval/big_bench_hard`, etc.): confirmed via web search to be Parquet re-packagings of the identical fixed BBH JSON data — no generator, no new content. Not cloned since they add nothing over `vendor/bbh/data/*.json`.
- No standalone reimplementation of a *procedural generator* for this exact task form (N people, pairwise swap sequence, "who has what") was found anywhere on GitHub. The task is simple enough (a few lines of `random.sample` + templated sentences) that nobody appears to have packaged a reusable generator for it separately from the fixed BBH release.

## Reference: `vendor/paper/suzgun2022.pdf`
Suzgun et al. 2022, "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them" (arXiv:2210.09261). Table 3 (page 8) has the exact per-task Random / SOTA / Human-Rater / InstructGPT(AO,CoT) / Codex(AO,CoT) / PaLM-540B(AO,CoT) numbers used in `desk.json.published_data`. Section 4.3 / Figure 5 specifically calls out Tracking Shuffled Objects as one of three BBH tasks where CoT produces "emergent" performance (flat/near-random scaling under answer-only prompting, steep improvement under CoT) — directly relevant to why this task is a good stateful-CoT probe.

## Recommendation

**Nothing usable as-is; a thin wrapper is not viable either — must write a fresh generator**, for two reasons:
1. The only real data source (BBH/BIG-bench) is a **fixed, canary-tagged, publicly-crawled dataset** with no generation script included — SPEC explicitly requires vendoring it (done) AND writing a fresh generator so instances aren't contaminated.
2. No candidate anywhere (BBH, BIG-bench origin, lm-eval-harness, HF mirrors) exposes `generate(depth, seed)`, an independent `solve()`, or a depth knob decoupled from N. The task itself is simple (composition of transpositions in S_n) — implementation from scratch, following the vendored CoT scratchpad format and BIG-bench's swap-selection constraints (every participant swaps ≥1 time; no immediate repeat of the same pair), is the correct next phase.

Useful salvage for that future generator: the exact scratchpad line format and answer-sentence phrasing from `vendor/bbh/cot-prompts/*.txt` (to keep our fresh instances stylistically matched to the published CoT numbers), and the swap-sampling constraints from `vendor/bigbench/.../README.md`.
