# Sourcing report — synthetic_program_trace

Task family: tiny straight-line-plus-loop integer programs (2-3 variables, `+`/`-`/`*`, one bounded
`for` loop) with a line-by-line scratchpad trace and a final-variable-value answer. Inspired by
Zaremba & Sutskever 2014 "Learning to Execute" (arXiv:1410.4615) and Nye et al. 2021 "Show Your Work"
(arXiv:2112.00114).

Search performed: web search (general + site-scoped to github.com, huggingface.co, arxiv.org,
paperswithcode.com) for the original papers' code/data, BIG-bench / BBH tasks, lm-evaluation-harness
tasks, HF datasets for "learning to execute" / "program execution" / "CodeExecution", and newer
(2024-2026) synthetic execution-trace / scratchpad-reasoning papers and their repos. GitHub's code-search
API required auth and wasn't usable unauthenticated; relied on web search + direct repo/directory probes
instead (e.g. HTTP 404 on `google-research/google-research/scratchpad` to confirm no such directory).

## Candidates found

### 1. `wojciechz/learning_to_execute` — vendored at `vendor/learning_to_execute/`
- **What it is:** The original code release for Zaremba & Sutskever 2014 (arXiv:1410.4615), linked
  directly from the paper. Torch7/Lua, trains an LSTM to predict the printed output of tiny generated
  Python-like snippets.
- **Commit:** `dc6baf6835337f0acea76c8e8233f767c50b058f` (2016-04-09, last commit on the repo).
- **License:** Apache 2.0 (file headers say "Copyright 2014 Google Inc.").
- **Source URL:** https://github.com/wojciechz/learning_to_execute
- **Contents relevant to us:** `data.lua` + `utils/operations.lua` implement a program generator with
  operation primitives `pair_opr`, `smallmul_opr`, `equality_opr`, `vars_opr`, `small_loop_opr`,
  `ifstat_opr` — i.e. variable assignment, addition/pairing, small multiplication, a bounded `for` loop,
  and an `if` statement, composed via a stack machine (`compose()` in `data.lua`) driven by a "hardness"
  (nesting/length) curriculum function. This is structurally the closest thing found to what SPEC.md
  wants (small integer programs with a loop, varying difficulty).
- **Gaps vs. the common interface:**
  - No CoT/scratchpad at all — the target is only the final printed numeric string (`to_data()` builds
    `x`/`y` character sequences ending in the program's stdout, nothing line-by-line). We'd need to
    invent the entire step/state trace format; nothing here produces `steps`/`states`/`step_spans`.
  - No reference solver independent of the generator — correctness is checked by piping the generated
    snippet through `python2.7` (`torch data.lua` self-check), i.e. it "solves" via a real Python
    interpreter subprocess, not a small pure-Python differential solver. Requires network-free stdlib
    execution per SPEC (`ast`/`exec` on generated code would work, but isn't provided here as reusable
    code).
  - Torch/Lua, not Python — would need a full reimplementation of `compose()`'s logic in Python (~150
    lines), not a straight port.
  - No `depth` knob in SPEC's sense; uses "nesting" and "length" (digit count) instead, and difficulty
    is sampled from a curriculum distribution rather than a clean `generate(depth, seed)` call.
  - Answer is a single printed value already (not a specific *variable's* final value out of several
    tracked variables), and multiplication inside nested composition is essentially unbounded in
    magnitude by design (this is exactly the "state growth" risk SPEC flags).
  - No published per-step accuracy figures — the paper reports overall sequence accuracy, not step-wise.
- **Verdict:** good conceptual reference and confirms the operation vocabulary (assign / +,- / small `*`
  / bounded loop) is exactly the 2014 paper's vocabulary too, but nothing here is reusable as code for
  our interface — it would all need to be rewritten in Python from scratch.

### 2. Nye et al. 2021 "Show Your Work" (arXiv:2112.00114) — **no repo found, nothing vendored**
- Searched arXiv abstract/HTML, OpenReview PDF, research.google publication page, and probed
  `google-research/google-research` for a `scratchpad` directory (404 — doesn't exist) and for a repo
  search (GitHub's code-search API requires auth, so this is a best-effort negative, not exhaustive).
  Found no code or data release linked from the paper, OpenReview, or Google Research's own page. The
  paper's scratchpad tasks (long addition, polynomial evaluation, Python program execution) are
  described in prose/figures only as far as could be found.
- **Verdict:** nothing to vendor; the paper is useful only as a citation for scratchpad format design
  and published numbers (see desk.json), not as a code/data source.

### 3. BIG-bench `auto_debugging` — vendored at `vendor/BIG-bench-tasks/bigbench/benchmark_tasks/auto_debugging/`
- **What it is:** A small **fixed** dataset (34 free-response examples in `task.json`), authored by
  Mo Tiwari, Chris Waites, Medina Baitemirova. Each example gives a short straight-line Python snippet
  (assignments, one arithmetic reassignment, sometimes a `for` loop) and asks "What is the value of X at
  line N?" or "...the k-th time line N is executed?" — e.g.
  `{"input": "for i in range(10):\n\tpass\nWhat is the value of i the third time line 2 is executed?", "target": "2"}`.
- **Commit:** `092b196c1f8f14a54bbc62f24759d43bde46dd3b` (google/BIG-bench main, 2024-01-18).
- **License:** Apache 2.0.
- **Source URL:** https://github.com/google/BIG-bench (sparse-checked out `auto_debugging`,
  `cs_algorithms`, `simple_arithmetic`, `modified_arithmetic` only — full repo is large).
- **Gaps:** fixed 34-item dataset, no generator, no scratchpad/CoT (asks for one intermediate value only,
  no step-by-step trace field), no reference solver code, canary-GUID-tagged (BIG-bench explicitly marks
  these "should never appear in training corpora" — i.e. this exact set is a known contamination target
  if reused verbatim; also its small size and public GitHub presence means high memorization risk for
  any model trained post-2023). Directly overlaps our task's *spirit* (variable-state reasoning without
  execution) but is answer-only, single-question, and not remotely procedurally generated.
- **Verdict:** useful as a validation of the task's face-validity and prompt phrasing style ("What is the
  value of X at line N?"), not usable as a base.

### 4. BIG-bench `simple_arithmetic` — vendored at `vendor/BIG-bench-tasks/bigbench/benchmark_tasks/simple_arithmetic/`
- **What it is:** `task.py`, a real (non-JSON) BIG-bench programmatic task: a seeded numpy generator
  producing `"3 + 4 = "` style single-operation arithmetic strings at variable difficulty
  (`ArithmeticTask(seed=42, num_trials=100)`), documented explicitly as "a template task to be used as an
  example during task development."
- **Contents relevant to us:** the cleanest small example in the BIG-bench corpus of a *seeded,
  parameterized Python generator* for arithmetic strings — useful as a style reference for how to
  structure `generate(depth, seed)`, even though it only covers one binary op, no variables, no loop, no
  scratchpad, no `Instance`-shaped output.
- **Verdict:** building-block/style reference only, not a base to wrap.

### 5. BIG-bench `modified_arithmetic` — vendored alongside (three_digit_addition/subtraction ± "plus one",
  two_digit_multiplication ± "plus one")
- Fixed JSON multiple-target datasets testing whether models can do standard arithmetic with a
  perturbed semantics (e.g. every digit shifted). No variables, no loop, no scratchpad. Not close to our
  task; kept only because it was in the same sparse-checkout pull and costs nothing extra.

### 6. BIG-bench `cs_algorithms` (`lcs`, `valid_parentheses` subtasks) — vendored alongside
- Fixed algorithmic-reasoning tasks (longest common subsequence, balanced-parens checking) with their
  own generators, but the algorithms themselves (LCS, bracket matching) are a different computational
  family from straight-line arithmetic + bounded loop tracing. Not close enough to adapt; kept only as a
  neighbor for reference on how BIG-bench structures a procedurally-generated `task.py`.

### Other things checked, nothing found
- HuggingFace Hub: no dataset titled or tagged "learning to execute" / "learning_to_execute". Newer
  execution-trace work that does exist on HF (e.g. `PatronusAI/trace-dataset`) is about a different
  thing (agent/tool-call traces / reward-hack detection), not integer-variable program tracing.
- lm-evaluation-harness (EleutherAI): no task matching straight-line program execution / variable
  tracing found in the docs or task guide; its tasks are overwhelmingly NLP benchmarks, not synthetic
  execution traces.
- Meta/FAIR paper "What I cannot execute, I do not understand" (arXiv:2503.05703) — studies execution
  traces for *real* Python functions (MBPP/CruxEval-style) mined from GitHub + fuzzed inputs, not a
  from-scratch tiny synthetic straight-line+loop generator, and no public code repo was found linked
  from the paper. Noted as a neighbor/related-work citation only, not vendored (nothing to vendor —
  no released dataset or generator found).
- CRUXEval (arXiv:2401.03065) is the separately-assigned neighbor task per the instructions; not
  duplicated here, but it's worth noting CRUXEval programs are *general* Python (mined/LLM-generated),
  not the constrained 2-3-variable arithmetic+loop family this task needs, so even if it were in-scope it
  wouldn't fit without heavy filtering.
- Papers-with-code "Learning to Execute" page just mirrors the arXiv listing and the same
  `wojciechz/learning_to_execute` repo above; no additional reimplementations with released data were
  found on the first several pages of results.

## Recommendation

**Nothing usable, must write.** The task as specified — a *pure-Python*, seed-deterministic generator
over 2-3 integer variables with a rigid per-line scratchpad template, a `states` list, `step_spans`
byte-offsets, and a solver independent of the generator — does not exist anywhere found in this search.
The closest real artifact, `wojciechz/learning_to_execute`, shares the operation vocabulary (assign,
add/subtract, small multiply, one bounded loop) but is 2016-era Lua/Torch code whose output target is a
single final printed value with no intermediate scratchpad, checked against a live Python interpreter
subprocess rather than a portable solver — none of that transfers as code, only as a confirmation that
the chosen operation set is reasonable and paper-precedented. BIG-bench's `auto_debugging` confirms the
prompt framing ("what is the value of X at line N") is a natural, previously-used phrasing, but it's a
fixed 34-item public dataset with no generator and no CoT field. Building blocks worth reusing when
`task.py` is eventually written: (a) the `wojciechz/learning_to_execute` operation vocabulary/curriculum
idea (nesting depth as a nudge for how to define `depth`), and (b) Python's own `ast`/`compile`/`exec` (or
just direct interpretation of a tiny custom AST) as a natural, dependency-free reference solver, which
none of the vendored candidates provide but which is easy to write correctly and independently from a
line-generator.
