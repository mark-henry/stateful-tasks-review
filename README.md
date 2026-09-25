# stateful-tasks

Code and results for [POST TITLE](POST_URL), a comparative review of chain-of-thought tasks from the
literature, asking which ones force a language model to carry load-bearing state in its CoT tokens. The review
reimplemented 16 published tasks (addition, Dyck completion, S5 composition, Towers of Hanoi, program traces,
and others) behind one common interface. Each gold trace uses the source's published format. The tasks were
then swept over depth with and without CoT on Qwen3.5-9B, Llama-3.3-70B and DeepSeek-V4-Flash. Survivors
went through trace knockout, filler, prompt blinding and mistake-propagation tests. This repository holds the
tasks, the [Inspect AI](https://inspect.aisi.org.uk/) harness that ran them, and every result table.

## Layout

    SPEC.md                  the task contract (read this before writing a new task)
    tasks/<slug>/            one directory per task; 16 implemented, 8 screened out at the desk stage
      task.py                the task: generator, solver, checker, gold trace, exemplars, redaction, corruption
      README.md              source, citation, license, format decision, depth semantics, caveats
      SOURCING.md            the implementations found upstream, with pinned commits and licenses
      desk.json              desk metrics of the task as published
      published_trace.txt    the verbatim published CoT exemplar
      examples.txt           rendered instances (`python task.py --demo`)
      vendor/                only the upstream files task.py reads (8 tasks; see below)
    harness/                 Inspect AI harness, drivers, plans and results; runbook in harness/README.md
      results/               every result table (markdown + CSV); results/tidy/ is the chart-ready version
    docs/                    the working record (below)

## Picking a task

| task | source | state carried | depth = |
|---|---|---|---|
| addition † | Nye et al. 2021 | partial-sum digits and carry | digit columns |
| blocksworld | Stechly et al. 2024 | block-stacking configuration | actions executed |
| boolean_expressions | Suzgun et al. 2022 (BBH) | partially reduced expression | sub-expression resolutions |
| cellular_automaton * | Neary & Woods 2006 (rule 110) | row of 8 binary cells | generations |
| cruxeval | Gu et al. 2024 | program variable values | executed statements |
| cup_shuffling | Suzgun et al. 2022 (BBH) | objects-to-people permutation | pairwise swaps |
| dyck | Suzgun et al. 2022 (BBH) | bracket stack | input symbols |
| entity_tracking_boxes * | Kim & Schuster 2023 | contents of every box | box operations |
| hanoi | Shojaee et al. 2025 | disks on three pegs | moves |
| multiplication | Dziri et al. 2023 | partial products, running sum | digits_y × (digits_x + 1) |
| nested_arithmetic | Suzgun et al. 2022 (BBH) | values of evaluated sub-expressions | sub-expression reductions |
| random_lookup_table † | Ramesh et al. 2024 | current symbol | table applications |
| s5_composition † | Liu et al. 2022; trace format from Li et al. 2025 | permutation of 5 | permutations composed |
| synthetic_program_trace † | Nye et al. 2021 | integer variables | executed statements |
| threesum † | Pfau et al. 2024 | enumeration position (implicit), hit flag | candidate triples checked |
| turing_machine | Wu et al. 2025 (TMBench; a tag system) | symbol queue | rewrite steps |

**Markers.** `†` = the source taught the trace format by fine-tuning or from-scratch training, so a prompted
model has never seen it. `*` = no published CoT trace exists and the format is ours. Both are defined in
`harness/stateful/registry.py`. Per-task verdicts, working depths and the format that works best are in
`harness/results/tidy/master.csv` and `tasks.csv`. multiplication was dropped after batch 1: the 9B scores 1.0 without CoT
through 3×3 digits, so there is no CoT gap to study.

Screened out at the desk stage (sourcing and desk metrics only, no `task.py`): affine_mod_p, grid_navigation,
khop_pointer_chasing, last_letter_concat, modn_counter, parity_coinflip, river_crossing_checkers, web_of_lies.

## The task contract, in brief

`SPEC.md` has the full contract. Each `task.py` is pure Python + stdlib and exposes:

- `Instance(prompt, steps, states, answer, depth, meta)`: the problem statement only, the gold trace split one
  element per serial step, the tracked state after each step, and the exact-match answer.
- `generate(depth, seed, **knobs)`: deterministic. `solve(inst)`: an independent reference solver.
- `check(inst, completion)`: reads the last `Answer:` line. `format_cot(inst)`: the gold trace in the
  published format, then `Answer: X`.
- `exemplars(k, seed)`: few-shot instances; `[0]` is the verbatim published exemplar where one exists.
- `DEPTHS`: the recommended sweep. `KNOBS`: name → (default, description). `ANSWER_FORMAT`: one line for the
  no-CoT instruction.
- `redact_prompt(inst, k)`: the prompt with the initial state and the first k operators replaced by `[…]`
  (prompt blinding). `corrupt_step(inst, k, seed)`: step k rewritten to report a plausible wrong state
  (mistake propagation; on the ten batch-3 tasks).
- Format knobs: `format="ergonomic"` (addition, random_lookup_table, s5_composition, threesum) is an SFT-free
  trace that writes the full state in plain language at every step. hanoi `formulation` is `"execute"`
  (apply a given move list; the default) or `"plan"` (batch 1). cellular_automaton `format` is `"cells"`
  (per-cell with a running row; the default) or `"rows"` (batch 1).
- `python task.py --selftest` checks all of this; `python task.py --demo` renders examples.

## Quickstart: one task, one model

    cd harness && uv sync && cp .env.example .env        # then put your Together key in .env
    uv run python run_batch1.py --slugs dyck --n 20 --log-dir results/logs/dyck-9b
    uv run python collect.py --log-dir results/logs/dyck-9b --out results/dyck-9b

This runs dyck at every depth in `DEPTHS`, with and without CoT, on the model in `.env` (Qwen3.5-9B on
Together, thinking off). It writes `results/dyck-9b/BENCH1.md`: accuracy per arm, the CoT gap and a paired
McNemar p per depth. `uv run inspect view --log-dir results/logs/dyck-9b` browses the samples. Without a key,
`uv run python check_task.py dyck` runs the task's self-test and a mock eval.

**Another provider.** The model is any Inspect model string. Pass `--model`, or set `STATEFUL_MODEL` in
`.env`. An OpenAI-compatible server is `openai-api/<name>/<model>` with `<NAME>_BASE_URL` and
`<NAME>_API_KEY` set, e.g. `openai-api/vllm/Qwen/Qwen2.5-7B-Instruct` with `VLLM_BASE_URL`. Set
`STATEFUL_EXTRA_BODY=` (empty) unless the server takes `chat_template_kwargs`. Batches 2 and 3 also need the
server to continue a prefilled assistant turn; see `harness/README.md`.

## Batches and result tables

All paths are under `harness/results/`. `harness/README.md` has the command that produced each one.

| batch | what | tables |
|---|---|---|
| 1 | acc(CoT) vs acc(no-CoT) over depth, n=50 per cell, 4 models | `qwen35-9b/`, `llama33-70b/`, `deepseek-v4-flash/`, `deepseek-v4-pro/` (each `BENCH1.md` + `batch1.csv`); survival verdicts in `SUMMARY.md` |
| 1, second chance | the same sweep in variant formats: ergonomic † formats, hanoi execute, cellular per-cell | `ergonomic-{9b,ds}/`, `second-chance-{9b,ds}/`, `second-chance3-{9b,ds}/`; comparison in `SECOND_CHANCE.md` |
| 2 | knockout of the model's own trace, filler, redacted vs plain continuation | `BENCH2.md`, `batch2.csv` |
| 3 | step-aligned gold knockout and mistake propagation, 9B | `BENCH3.md` |
| derived | load-bearing token density estimate | `DENSITY.md` |
| all | chart-ready long-format CSVs recomputed from the logs and cross-checked against every table above | `tidy/` (`tidy/README.md` documents each file and every disagreement found) |

Pass rule for batch 1: gap ≥ 0.2, paired McNemar p < 0.05, and acc(CoT) ≥ 0.7 at the same depth.

**Logs.** Every table was built from the Inspect `.eval` logs (about 300 MB), which are not in the
repository. They are available on request. `tidy.py` and `density.py` need them; the other scripts need them
only to rebuild tables.

**Known quirks in the tables.** `SUMMARY.md` predates the `†` marker: it marks the five † tasks with `*`
and uses the old verdict wording. It also takes the shallowest of tied best depths, while everything later
takes the deepest, and it misses boolean_expressions' d3 pass through a floating-point comparison. All of
this is itemised in `tidy/crosscheck.csv`, and `tidy/` has the corrected values. The hanoi execute and
cellular per-cell runs used changed task defaults rather than a knob; `tidy/README.md` explains how they were
told apart.

## Vendored code and data

The upstream clones (1.2 GB) are not shipped. Each task's `SOURCING.md` records the repository, pinned
commit and license of everything that was cloned. Eight tasks read upstream files at runtime; only those
files are in `tasks/<slug>/vendor/`, with the upstream LICENSE where there is one:

| task | files | upstream |
|---|---|---|
| boolean_expressions, dyck, nested_arithmetic, cup_shuffling | BBH task JSON and CoT prompt | suzgunmirac/BIG-Bench-Hard @ 9ee07bd, MIT |
| cruxeval | `data/cruxeval.jsonl` (800 rows) | facebookresearch/cruxeval @ 190faf1, MIT |
| entity_tracking_boxes | two 100-noun object vocab CSVs | sebschu/entity-tracking-lms @ 8400de0, no license file |
| multiplication | `generate_scratchpads.py` (imported) | nouhadziri/faith-and-fate @ 1e90edb, MIT |
| turing_machine | `src/tag_generate.py` (imported; pure-Python fallback if absent) | HaitaoWuTJU/Turing-Machine-Bench @ 9e04fca, no license file (the HF dataset is Apache-2.0) |

Task READMEs and SOURCING.md files mention other `vendor/<repo>` paths. Those refer to the full clones, which
you can recreate from the recorded commits.

## Docs

- `docs/DESK.md`: the desk-metrics table for every candidate task, as published (the post's desk table).
- `docs/REVIEW.md`: the phase-1 sourcing verdicts, one row per candidate task.
- `docs/IMPL.md`: implementation status at the start of batch 1, with each task's deviations from its source.

`DESK.md` at the top level is a symlink to `docs/DESK.md`, kept because `harness/tidy.py` reads it from there.
