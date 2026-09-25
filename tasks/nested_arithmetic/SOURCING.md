# SOURCING.md — nested_arithmetic (BBH `multistep_arithmetic_two`)

Phase 1 (sourcing only, per SPEC.md AMENDMENT). No task.py / generator / wrapper code was written.

## Candidates found and vendored

### 1. `vendor/bbh-suzgun/` — suzgunmirac/BIG-Bench-Hard (the canonical BBH source)
- URL: https://github.com/suzgunmirac/BIG-Bench-Hard
- Commit: `9ee07bd481feebf959a6b59d61ea57bdcf30964d` (shallow clone, then pruned to the
  relevant files; full clone was 42MB, mostly unrelated Codex output logs and figures
  for other BBH tasks — those were deleted, `.git` kept for provenance of this one commit).
- License: MIT (LICENSE file vendored in full).
- Contents kept: `bbh/multistep_arithmetic_two.json` (250 fixed examples, `{"input":..., "target":...}`,
  canary-GUID-tagged so it must never be used verbatim as eval data / training data), and
  `cot-prompts/multistep_arithmetic_two.txt` (the paper's 3-shot chain-of-thought prompt,
  same canary GUID).
- What it contains: a **fixed dataset** (not a generator) + a **fixed 3-shot CoT prompt** with a
  worked, prose-style scratchpad ("Let's calculate A = ... = ...", "Let's calculate B = ...",
  "Then, the final equation is A op B = ..."). No reference solver code (the "solver" is
  whatever computed the fixed `target` values — presumably Python `eval`). No published
  per-task numbers file (numbers live only in the paper, see below).
- Distance from common interface: the CoT text is close in spirit to the `steps`/`format_cot`
  idea (each intermediate reduction reported), but it is prose, not a rigid one-line-per-step
  template, and the 250 examples are fixed/finite and canary-tagged (contamination risk),
  which the SPEC explicitly wants us to route around by writing a fresh generator "of the
  same form." Missing: generator, rigid step format, `state` tracking, reference solver
  independent of the fixed answers, `check()`/tolerant parsing, KNOBS.

### 2. `vendor/bigbench-google/` — google/BIG-bench, `benchmark_tasks/multistep_arithmetic/`
- URL: https://github.com/google/BIG-bench
- Commit: `092b196c1f8f14a54bbc62f24759d43bde46dd3b` (shallow clone; full repo is ~2.3GB —
  hundreds of tasks' worth of result logs/plots — so only `task.py` and the task's `README.md`
  plus the repo's top-level `LICENSE`/`README.md` were kept; `.git` dropped, commit hash
  recorded in `VENDORED_FROM.txt` instead).
- License: Apache 2.0 (LICENSE file vendored in full).
- Contents kept: `bigbench/benchmark_tasks/multistep_arithmetic/task.py` and its `README.md`.
- What it contains: this is the **original, pre-"Hard"-curation BIG-bench task** that
  `multistep_arithmetic_two` in BBH was drawn from/inspired by. It IS a real **procedural
  generator**: `MultistepArithmeticTask.generate_string(depth_levels, length)` recursively
  builds nested-parenthesis expressions parameterized by `depth_level_list` (nesting shape,
  e.g. `[2,2]` = two levels of two-way splits) and `lengths` (operator count in each innermost
  parenthesis), sampling from `operations=['+','-','*']` and `numbers=range(-9,10)` via a
  `numpy.random.RandomState(seed)`. The "solver" is a bare `eval(code)` call on the generated
  string (no independent solver, no CoT/scratchpad at all — this is a zero-shot
  answer-only task, scored by exact string match against `eval()`'s output via regex
  `[-+]?\d+`). No published per-task CoT numbers (this is pre-BBH; BBH's contribution was
  adding CoT prompts and re-measuring).
- Distance from common interface: closest thing to a real *generator* we found, and its
  grammar/operand-range choices (small ints -9..9, tree-shaped nesting via depth-level lists)
  are a reasonable, well-precedented basis for a fresh generator. But: uses `numpy` (SPEC
  requires stdlib only — trivial to replace with `random.Random(seed)`), has zero scratchpad/
  step-tracking/`Instance`/`KNOBS` scaffolding, and its "solver" is unsafe raw `eval()` on
  the generated string rather than an AST-based evaluator. Everything from the `Instance`
  dataclass down through `format_cot`/`step_spans`/`check`/`KNOBS` would need to be written
  from scratch; only the recursive expression-shape idea is reusable.

### 3. `vendor/lm-eval-harness-bbh/` — EleutherAI/lm-evaluation-harness, `lm_eval/tasks/bbh/`
- URL: https://github.com/EleutherAI/lm-evaluation-harness
- Not cloned (repo is large and this is config-only); individual files fetched via
  `raw.githubusercontent.com` at commit `ad8737ae7fad24cf64e50fc7fc31397bff586b9e` (HEAD of
  `main` at fetch time, recorded in `VENDORED_FROM.txt`); per-file history SHA for
  `cot_fewshot/multistep_arithmetic_two.yaml` also recorded (`c2be72110ff2c5bd764a4b655d178c522cbc51a2`).
- License: MIT (`LICENSE.md` vendored in full).
- Contents kept: `cot_fewshot_multistep_arithmetic_two.yaml` (3-shot CoT config, same 3 examples
  as the Suzgun repo's CoT prompt, `include: _cot_fewshot_template_yaml`), `fewshot_...yaml`,
  `zeroshot_...yaml`, `cot_zeroshot_...yaml`, and the shared `_cot_fewshot_template_yaml.yaml`
  (declares `dataset_path: SaylorTwift/bbh` — a HF mirror of the fixed BBH data — plus the
  answer-extraction regex `(?<=the answer is )(.*)(?=.)`, `exact_match` metric, `num_fewshot: 3`,
  greedy decoding).
- What it contains: a **thin YAML eval-harness wrapper** around the same fixed BBH dataset
  (via a HF mirror), not a generator, no reference solver, no new numbers — it's plumbing for
  running an LLM against the fixed set and scoring with an answer-extraction regex.
- Distance from common interface: gives us a concrete, working answer-extraction regex worth
  imitating for `check()`'s tolerant parsing (`"the answer is X"` at the end of the CoT), and
  confirms the field names/split. Otherwise adds nothing beyond what's already in `bbh-suzgun/`.

### HuggingFace dataset mirrors (not vendored, no unique content)
Search surfaced several HF mirrors of the same fixed BBH data with no generator, no solver, no
new format: `maveriq/bigbenchhard`, `Joschka/big_bench_hard`, `chiayewken/bbh-cot`,
`SaylorTwift/bbh` (the one lm-eval-harness points at), `RUCAIBox/bbh`. All are re-packagings of
the identical 250 `multistep_arithmetic_two` examples from `bbh-suzgun/bbh/multistep_arithmetic_two.json`
in parquet/dataset-script form. Not cloned — zero marginal information over what's already vendored.

## Published numbers (Suzgun et al. 2022, Table 3 — per-task BBH results)

Sourced via ar5iv HTML rendering of arXiv 2210.09261 (cross-checked across two separate fetches
after the first fetch returned the InstructGPT/Codex CoT numbers transposed; the two later fetches
agreed with each other). **This should still be spot-checked against the actual PDF/HTML table
before being treated as ground truth for any published claim** — automated table extraction from
this paper's PDF is unreliable (the table renders as much wider than the text column).

Row: "Multi-Step Arithmetic [Two]", Table 3 ("Few-shot prompting performance of several large
language models on BIG-Bench Hard (BBH)"):

| | No-CoT (Answer-Only) | CoT |
|---|---|---|
| Random guessing | 0.0% | — |
| Prior SOTA (per BIG-Bench paper) | 5.7% | — |
| Average human-rater | 9.7% | — |
| Max human-rater | 25.0% | — |
| InstructGPT (text-davinci-002) | 1.2% | 53.2% |
| Codex (code-davinci-002) | 1.2% | 47.6% |
| PaLM 540B | 1.6% | 19.6% |

Headline finding restated in the abstract/intro: CoT prompting produces the largest gains on
this task of any BBH task category, with Codex/InstructGPT going from near-random (~1%) to
47–53% with CoT — i.e. this is one of the tasks BBH highlights as showing CoT is *necessary*,
not just helpful, for multi-step arithmetic.

## Recommendation

**Nothing usable as-is; usable with a thin-to-moderate wrapper for the generator half only.**

- The fixed BBH/BIG-bench dataset and CoT prompt (`bbh-suzgun/`) are exactly what the SPEC
  says NOT to rely on directly (fixed, canary-tagged, contamination-risked) — they're useful
  only as ground truth for "does our fresh generator produce the same *form* of problem" and
  as the source of published comparison numbers.
- The google/BIG-bench `task.py` generator (`bigbench-google/`) is the one candidate with real
  reuse value: its recursive nested-expression construction (depth-level lists × innermost
  length × operator/number sampling) is a solid, precedented basis for `generate()`, and could
  be ported to stdlib `random.Random(seed)` with modest effort. But it has none of the required
  scaffolding (`Instance`, rigid-template `steps`/`states`, `format_cot`, `step_spans`,
  `check()`, `KNOBS`, an independent AST-based `solve()`) — that's all new code, not a thin
  wrapper, even though the underlying grammar idea is borrowed.
- lm-eval-harness contributes only a proven answer-extraction regex pattern, nothing structural.

So: **"nothing usable as-is; must write generate/solve/check/format_cot/step_spans/KNOBS from
scratch"**, but do not start from a blank page on the generator's expression grammar — reuse the
depth-levels × operand-range design documented above from `bigbench-google/task.py`, and reuse
the fixed BBH examples/CoT prompt purely as a form-reference and as the source of the published
comparison numbers in `desk.json`. This decision (to write code) is for human review to confirm
per the SPEC amendment; this file only recommends.
