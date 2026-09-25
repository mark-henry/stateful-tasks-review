# SOURCING — cruxeval

Phase 1 (sourcing only, per SPEC.md AMENDMENT). No task.py, generator, or wrapper code has been written for
this task directory. Nothing was here before this pass (no pre-existing draft to flag as untrusted).

## Candidate 1: facebookresearch/cruxeval (the canonical repo) — vendored

- `vendor/cruxeval_repo/` — `git clone --depth 1 https://github.com/facebookresearch/cruxeval`
- Commit: `190faf16d175b5847b0af05d937872b1fb395942` (HEAD of default branch at clone time, 2026-09-11)
- License: MIT (Copyright 2023 Meta), see `vendor/cruxeval_repo/LICENSE`
- Size: ~27MB total (~11MB is `.git`)
- Paper: Gu, Rozière, Leather, Solar-Lezama, Synnaeve, Wang. "CRUXEval: A Benchmark for Code Reasoning,
  Understanding and Execution." arXiv:2401.03065 (2024). ICML 2024.
- Also on HuggingFace: `cruxeval-org/cruxeval` (same 800 rows, MIT). Not separately vendored since the
  GitHub repo's `data/cruxeval.jsonl` is byte-identical in schema and the repo also gives us the eval
  harness and prompt templates the HF dataset card doesn't.

**Contents:**
- `data/cruxeval.jsonl` — 800 rows (`wc -l` reports 799 because the final line has no trailing newline;
  verified by parsing: ids `sample_0`..`sample_799`, 800 total), fields: `code` (str, a short `def f(...):`
  function, 3-13 lines), `input` (str, literal args as they'd appear inside `f(...)`), `output` (str, literal
  repr of the return value), `id`.
- `prompts.py` — the four canonical prompt templates used in the paper: `make_direct_output_prompt_phind`,
  `make_cot_output_prompt` (the one we want — 1-shot CoT with `[PYTHON]/[THOUGHT]/[ANSWER]` tags, manually
  written natural-language execution trace, NOT a settrace-based mechanical trace), `make_direct_output_prompt`,
  and the mirror-image input-prediction versions.
- `evaluation/utils_execute.py` — an execution-based correctness checker (`check_correctness`): builds a
  `code + "\n" + assertion` string, execs it in a subprocess with a timeout and a `reliability_guard()`
  (disables destructive builtins/os calls) — this is the paper's own differential-check mechanism, adapted
  from OpenAI HumanEval's `execution.py`. It checks that an assertion PASSES, not that it produces a specific
  literal — so as a "solve"-equivalent it would need adaptation (we want the produced value, not pass/fail).
- `evaluation/evaluate_generations.py`, `read_results.py` — batch scoring / pass@1,5 aggregation over
  pre-generated model outputs. Not directly reusable for a single-instance `check()`.
- `samples/` — example generations + scored results for CodeLlama-7B (both directions), useful only as a
  format reference.
- No reference to computing per-instance execution-trace line counts or var-dict snapshots (the "depth"
  proxy and `states` list required by SPEC.md) — that machinery does not exist here and would have to be
  written (via `sys.settrace`) if this task moves to implementation.

**Distance from common interface:** dataset + prompt templates + an execution-based verifier are all present
and license-clean, but: (a) there is no `sys.settrace`-based line/state tracer — the paper's own CoT is a
*manually authored* natural-language trace only used as a 1-shot example, not something generated per-instance;
(b) no depth/trace-length bucketing exists; (c) the verifier checks assert-pass, not produces a literal value,
so `solve()` would need a thin independent rewrite (plain `exec` + call + `repr`); (d) no notion of `steps`/`states`
lists aligned 1:1 with `format_cot()` as SPEC.md requires.

## Candidate 2: EleutherAI/lm-evaluation-harness `lm_eval/tasks/cruxeval/` — vendored (sparse checkout)

- `vendor/lm_eval_harness_cruxeval/` — sparse-checkout of `lm_eval/tasks/cruxeval/` only, from
  `git clone --filter=blob:none --sparse https://github.com/EleutherAI/lm-evaluation-harness`
- Commit: `ad8737ae7fad24cf64e50fc7fc31397bff586b9e` (HEAD of `main` at fetch time, 2026-09-11)
- License: MIT (lm-evaluation-harness repo license)
- Contributed by @ThomasHeap per the PR history (search result; PR #3699), added as a lm-eval-harness task.
- Contents: `cruxeval_common_yaml` (points at HF `cruxeval-org/cruxeval`, `test` split, `unsafe_code: true`),
  `cruxeval_output.yaml` / `cruxeval_output_08.yaml` / `cruxeval_output_cot.yaml` (and mirror `input` variants),
  `utils.py` (execution sandbox nearly identical to the original repo's `reliability_guard`/`check_correctness`,
  plus `pass_at_k` aggregation, plus regex/string-based `extract_code_output`/`extract_code_input` answer
  parsers that pull the value out of `[ANSWER]...[/ANSWER]` tags or after `==`), `README.md` (task doc +
  citation), fewshot examples in code (`list_fewshot_samples_output/input`).

**Contents assessment:** This is a harness-config wrapper around the *same* dataset and the *same* prompt
style as Candidate 1 (in fact the CoT prompt text embedded in `cruxeval_output_cot.yaml` is a byte-for-byte
copy of `make_cot_output_prompt` from the original repo). Its main added value over Candidate 1 is: (a) a
cleaner, more tolerant answer-extraction function (`extract_code_output`, handles missing tags, stray
quotes/comments, multi-line garbage) that is a good reference for writing `check()`'s tolerant parser, and
(b) YAML-declared generation configs (temperature, `until` stop sequences) that document exactly how the
CoT variant is invoked. It still does **not** provide a settrace-based line tracer, a depth notion, or
`steps`/`states` lists — it's built for a different harness (lm-eval-harness's generate-until + regex-filter
pattern, not this repo's Instance/generate/solve/check interface), so integrating it would mean extracting the
parsing logic, not "using it as-is."

**Distance from common interface:** same gaps as Candidate 1, minus needing to write a tolerant answer parser
from scratch (this file gives a solid model for it).

## Candidate 3 (not vendored — no downloadable code): Nye et al. 2021 "Show Your Work: Scratchpads for
Intermediate Computation with Language Models" (arXiv:2112.00114)

This is the paper CRUXEval itself cites as precedent for scratchpad-style execution traces (line source +
local-variable-dict state after each line — exactly the `steps`/`states` shape SPEC.md asks for). Searched
for an official implementation; found none — no linked GitHub repo turned up in search results, and the
technique (alternating source-line / JSON-dict-of-locals) is described in the paper text only. **Not vendored**
because there is nothing installable/clonable; it's useful only as a written-format precedent, cited here for
the README's justification of the settrace-based `steps`/`states` design if this task is later implemented.

## Candidate 4 (noted, not vendored): "What I cannot execute, I do not understand" (arXiv:2503.05703) and
"Think Like You Execute" (arXiv:2512.00127) — both report using dynamic execution-trace scratchpads and
evaluate on CRUXEval/MBPP (the former reports ~80% on CruxEval/MBPP with trace-augmented training), but
neither search turned up a public code/data repo. Same disposition as Candidate 3: cited as prior art only.

## Recommendation

**Usable with a thin wrapper.** The canonical `facebookresearch/cruxeval` dataset (MIT, 800 code/input/output
triples) is exactly the right fixed dataset for CRUXEval-O and needs no filtering or re-scraping. Nothing
vendored provides the settrace-based per-line tracer, depth-bucketing, or `Instance.steps/states` structure
that SPEC.md's common interface requires — that portion is genuinely missing from every candidate found and
would need to be written from scratch (a `sys.settrace` line tracer is straightforward stdlib Python, on the
order of ~50-100 lines). The lm-eval-harness task's `extract_code_output` parser is a good template to adapt
for `check()`'s tolerant-parsing requirement, saving some of that work. In short: dataset = vendor as-is;
prompt style = adapt the paper's own `make_cot_output_prompt` template; verifier/tracer/depth-bucketing =
write new, informed by the two vendored candidates' parsing/execution-safety conventions. No candidate is
usable as-is against the Instance/generate/solve/check/format_cot/step_spans interface.

## Contamination note (relevant to desk.json)

CRUXEval is a well-known, actively-leaderboarded public benchmark (homepage, HF dataset, lm-eval-harness
integration, X/Twitter announcement thread, PapersWithCode entry) that has been in circulation since Jan
2024. High likelihood of appearing in pretraining/RLHF/eval-mix data for any model trained after that date.
Flagged as **high** contamination risk in desk.json.
