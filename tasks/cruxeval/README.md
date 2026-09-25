# cruxeval — CRUXEval-O (output prediction)

Given a short Python function `f` and a concrete input, predict the value `f(input)` returns.
This is the **output-prediction** direction of CRUXEval (CRUXEval-O). Implemented against
SPEC.md AMENDMENT 3.

## Source, citation, license

- **Paper:** Alex Gu, Baptiste Rozière, Hugh Leather, Armando Solar-Lezama, Gabriel Synnaeve,
  Sida I. Wang. *CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution.*
  arXiv:2401.03065 (2024), ICML 2024.
- **Repo:** https://github.com/facebookresearch/cruxeval —
  commit `190faf16d175b5847b0af05d937872b1fb395942`, **MIT** (© 2023 Meta Platforms).
- **Secondary (parsing reference only):** EleutherAI/lm-evaluation-harness
  `lm_eval/tasks/cruxeval/`, commit `ad8737ae7fad24cf64e50fc7fc31397bff586b9e`, MIT.

Both are in `vendor/`. See `SOURCING.md` for the full phase-1 write-up and `published_trace.txt`
for the verbatim published CoT exemplar.

## Vendored vs written

**Vendored and used at runtime**
- `vendor/cruxeval_repo/data/cruxeval.jsonl` — the fixed 800 rows (`code`, `input`, `output`, `id`).
  This is the only file `task.py` reads.
- `vendor/cruxeval_repo/prompts.py` — the published prompt template and the single hand-written
  CoT exemplar, transcribed verbatim into `task.py` (`PUBLISHED_*` constants) and rendered by
  `exemplars(k, seed)[0]`.

**Written from scratch** (nothing in either vendored repo provides these — see SOURCING.md)
- the `sys.settrace` line tracer that produces per-instance `steps` / `states`;
- the executed-line depth index and bucketing;
- `solve()` (the vendored `check_correctness` answers *pass/fail on an assertion*, not *what value
  came out*);
- `check()`'s extraction + literal comparison (mirroring, not importing, the official scorer;
  see "Ground truth and scoring").

## Format decision

The published CRUXEval-O CoT format is reproduced exactly: a `[THOUGHT]` block opening with
`Let's execute the code step by step:`, a blank line, a numbered list of execution steps, then
`[/THOUGHT]`, then `[ANSWER] assert f(<input>) == <literal> [/ANSWER]`. `format_cot()` appends the
one harness-imposed normalization, a final `Answer: <literal>` line.

`Instance.prompt` is the published problem statement only, with no instruction and no exemplars:

```
[PYTHON]
def f(...):
    ...
assert f(<input>) == ??
[/PYTHON]
```

**Where we deviate, and why.** The published `[THOUGHT]` block is *free prose* — the repo's one
hardcoded exemplar happens to number its sentences 1–5, but nothing in the prompt instruction
mandates numbering, a per-line template, or any particular granularity (`desk.json:
format_description`). There is exactly one published trace in existence and it was written by hand,
so there is no published *procedure* for producing a trace for an arbitrary instance. Per the task
brief, `steps` are therefore rendered **mechanically** from the real execution, one step per
executed statement, in the published numbered-sentence shape:

```
3. line 4: key = d.popitem()[0] -> d = {'f': 1, 'h': 2}, key = 'j'
```

i.e. `<n>. line <lineno>: <source of that line> -> <what changed>`. When a step changes nothing
observable the effect is `no change`; when the statement returns, it is `returns <value>`. Under
recursion the frame ordinal is shown (`line 3 (call 2): ...`). The closing sentence
`N+1. The return value of the function is therefore <literal>.` copies the published exemplar's
final sentence and is *not* counted as a step (it carries no new state).

`exemplars(k, seed)[0]` is the published exemplar verbatim — its five hand-written sentences, not
the mechanical trace. Its `depth` is therefore 5 (the number of narrated sentences) while its
executed-line count is 2; both are recorded in `meta`. This is the one place where the narrated and
mechanical step counts disagree, deliberately, so that the few-shot prompt shows the model the
published example exactly as published.

## Depth semantics

Arbitrary-depth *generation* is not meaningful here: CRUXEval is a fixed, hand-filtered set of 800
instances, and synthesising new functions would produce a different benchmark. Per AMENDMENT 3,
**depth = the number of executed statements** — `sys.settrace` `line` events raised inside frames of
`f` — when the reference code is run on the reference input. The 800 instances are bucketed by that
count and `generate(depth, seed)` samples from the bucket. `len(steps) == len(states) == depth`.

Selection is deterministic and *distinct-per-seed*: the bucket is shuffled with a fixed
bucket-derived seed and indexed by `seed % len(bucket)`, so seeds `0 .. len(bucket)-1` give
different dataset rows before wrapping.

Tracing counts only frames whose code object is `f` itself. Nested `lambda` / generator-expression
frames (95 and 39 line events across the whole dataset) are treated like any other callee and do
not add steps; on CPython ≥ 3.12 list/dict/set comprehensions are inlined by PEP 709 and so *are*
counted, in `f`'s own frame.

### Bucket sizes (CPython 3.12.4, all 800 rows traced, 0 skipped)

| depth | n | depth | n | depth | n | depth | n |
|---|---|---|---|---|---|---|---|
| 1 | 91 | 11 | 17 | 21 | 5 | 33 | 6 |
| 2 | 142 | 12 | 20 | 22 | 6 | 34 | 2 |
| 3 | 111 | 13 | 18 | 23 | 5 | 35 | 3 |
| 4 | 62 | 14 | 11 | 24 | 5 | 36 | 1 |
| 5 | 44 | 15 | 16 | 25 | 5 | 37 | 2 |
| 6 | 37 | 16 | 8 | 26 | 2 | 38 | 1 |
| 7 | 32 | 17 | 6 | 27 | 2 | 39 | 4 |
| 8 | 25 | 18 | 10 | 28 | 4 | 40 | 1 |
| 9 | 30 | 19 | 7 | 29 | 4 | 43 | 3 |
| 10 | 12 | 20 | 7 | 30–32 | 1,1,3 | 47+ | 1–3 each |

Long tail: 47(2), 50, 53, 56, 57, 61, 63(2), 70, 73(2), 76(3), 78(2), 84, 88, 90, 98, 101, 112,
131, 136, 157, 205, 625 — one or two instances each. `python3 task.py --buckets` prints the full
histogram from the live index.

### DEPTHS

`DEPTHS = [1, 2, 4, 6, 9, 12, 15]` — sizes `91, 142, 62, 37, 30, 20, 16`.

Chosen as a roughly geometric sweep over the buckets that hold **at least 16 instances**, which is
the practical floor for getting a non-degenerate accuracy estimate per depth point. 15 is the
deepest such bucket; everything above it has ≤ 10 members. Depth 1 is trivial (a single `return`
line — direct prediction should already solve most of it); by depth 12–15 the trace involves 4–5
loop iterations with live accumulator state, which is where a 7B is expected to fall apart (the
paper reports 46% pass@1 for Code Llama **34B** with CoT on CRUXEval-O overall, so a 7B at the top
of this sweep should be well under that).

To probe deeper than the dataset comfortably allows, use the `tolerance` knob
(`generate(20, 0, tolerance=4)` draws from buckets 16–24). With `tolerance > 0`, `len(steps)` no
longer equals the requested `depth` — `meta["executed_lines"]` carries the true count.

## ANSWER_FORMAT

`"a Python literal, the value returned by the function"` — this is the published instruction's own
wording ("Complete the assertion with a literal (no unsimplified expressions, no function calls)
containing the output"), compressed to one line for the no-CoT prompt. The answer space is
unbounded (any Python value's `repr`), which is a real weakness of this task as a CoT probe
(no chance floor, no multiple-choice fallback) but is intrinsic to the benchmark.

## Ground truth and scoring

- `solve(inst)` runs `code; f(input)` in a **separate subprocess** with a timeout (`timeout` knob,
  default 10 s) and returns `repr(result)`. It never reads the dataset's `output` column, so
  `solve() == inst.answer` is a genuine differential check against the published labels. **All 800
  rows round-trip exactly** (`repr(f(input)) == row["output"]` for every row).
- `check(inst, completion)` extracts the answer with the AMENDMENT 3 rule — last `Answer:` line;
  if there is none, fall back — with one task-specific step inserted into the fallback branch: the
  `[ANSWER] … [/ANSWER]` block is tried before "last non-empty line", because a model emitting the
  published format ends on the literal `[/ANSWER]` tag. An `assert f(...) ==` prefix is stripped.
- Comparison mirrors the official scorer (`evaluation/utils_general.py::evaluate_score`, which
  executes `assert <gold> == <pred>`): both sides go through `ast.literal_eval` and are compared
  with `==`, falling back to `repr` equality and then to a whitespace-normalized string compare.
  All 800 gold outputs are `literal_eval`-able; a small fallback handles non-literal reprs a model
  might emit (`set()`, `frozenset(...)`, `float('inf')`, `float('nan')`). The official anti-cheat
  guard is kept: a prediction containing `f(<input>)` is rejected.
- **Deliberate deviation from the official scorer: we never `exec` model output.** The official
  scorer executes the model's text inside an `assert`. `check()` only `literal_eval`s it. The
  practical difference is that we reject unsimplified expressions (`2*3` for `6`) that the official
  scorer would accept — but the prompt explicitly forbids those, so this is a tightening, not a
  loosening. `1` for `True` and `1` for `1.0` are still accepted, matching `==` semantics.
- Nothing untrusted is executed in a subprocess either: `solve()` only ever runs the vendored
  dataset's own code. The vendored `reliability_guard()` is therefore not reproduced; if this
  module is ever pointed at unvetted code, add it.

## Redaction

`redact_prompt(inst, k)` (SPEC.md AMENDMENT 4) returns the problem statement with everything that
would let the model *recompute* the state after step k removed. A CRUXEval prompt has only two
pieces: the function source and the call `assert f(<input>) == ??`. The source is **static
material** — the rule table of this task, needed by every remaining step — so it is kept intact;
the input argument value is the initial state and the only thing a model could read that state
back out of, so for `k >= 1` it is replaced by a single `[…]`. `redact_prompt(inst, 0)` returns
`inst.prompt` unchanged; `REDACTION_MEANINGFUL = True`.

**This is the closest available analogue of the amendment's operator-wise redaction, not the real
thing.** CRUXEval has no per-step operator list in the prompt: the "operators" are the statements
of the function body, which are static and must stay, and the input is consumed by the first
executed statement rather than one item per step. So the redaction is *all-or-nothing* — every
`k >= 1` yields the same string, and `redact_prompt(inst, 1) == redact_prompt(inst, inst.depth)`.
A multi-argument call gets one placeholder for the whole argument span (`f([…])`), since the
removed items are consumed as a single initial binding. The graded-`k` part of the probe is
therefore carried entirely by the length of the trace prefix the harness supplies, not by the
prompt.

Rendered example, `sample_210` (depth 6), at `k = depth // 2 = 3`:

```
[PYTHON]
def f(n, m, num):
    x_list = list(range(n, m+1))
    j = 0
    while True:
        j = (j + num) % len(x_list)
        if x_list[j] % 2 == 0:
            return x_list[j]
assert f([…]) == ??
[/PYTHON]
```

Unredacted, that last line reads `assert f(46, 48, 21) == ??`.

`--selftest` checks, over 20 instances at `k ∈ {0, 1, depth//2, depth}`: `k = 0` is the identity;
every `k > 0` differs from the prompt and contains `[…]`; all `k >= 1` agree; and at `k = depth`
the result is exactly `make_prompt(code, "[…]")`, the code and the `== ??` question survive, and
the input string appears nowhere in the redacted prompt (unless it also occurs inside the code
itself, e.g. the literal `1`) and nowhere on the assertion line.

## Knobs

| knob | default | meaning |
|---|---|---|
| `tolerance` | `0` | widen the bucket to `[depth-t, depth+t]`; reaches sparse deep buckets, breaks `len(steps) == depth` |
| `state_in_step` | `"delta"` | how state is shown in a step line: `delta` (only what changed), `full` (all locals), `none` |
| `max_state_chars` | `120` | truncation for the rendered state string |
| `max_repr_chars` | `80` | truncation for a single variable's `repr` inside the tracer |
| `timeout` | `10.0` | seconds per reference-execution / tracing subprocess |

`depth` is not a knob.

## Caveats

1. **Contamination risk: high.** CRUXEval has been a public, leaderboarded, HF-hosted benchmark
   since January 2024, and these exact 800 `(code, input, output)` triples are almost certainly in
   the post-2024 pretraining mix. This is the single biggest weakness of this task for the review
   and is not fixable without regenerating the dataset. The functions were themselves generated by
   Code Llama 34B from a seeded prompt (`data/generate_function_prompts.py`), so a fresh
   generator is *possible* but would need an LLM at generation time — out of scope for a
   stdlib-only module, and it would no longer be the published benchmark.
2. **Depth is interpreter-dependent.** PEP 709 (CPython 3.12) inlines comprehensions into the
   enclosing frame, so their body lines count as `f` steps on ≥ 3.12 but would be separate
   (uncounted) frames on ≤ 3.11. Bucket membership will shift if the harness runs on an older
   interpreter. The interpreter version is recorded in `meta["python"]` and printed by
   `--selftest`. Validated on CPython 3.12.4.
3. **Multi-line statements render only their first physical line.** `settrace` reports the
   statement's starting line, so a step for a statement spanning lines 2–5 shows only line 2's
   text. The `line <n>` marker disambiguates it, but a long comprehension reads awkwardly
   (see `sample_649` in `examples.txt`). Fixing it by splicing the whole logical statement would
   repeat the same multi-line block on every step of an inlined comprehension, which is worse.
4. **State is not at a fixed position and does not tokenize atomically.** The tracked state is an
   arbitrary Python value embedded in prose. `desk.json` already records
   `state_at_fixed_position: false`, `state_atomic_tokens: false`, `state_bounded: false`.
   For per-step attention-blinding, `states[k]` gives the canonical short form and `step_spans()`
   gives the char offsets of each step within `format_cot()`.
5. **`answer` quoting.** Answers are the dataset's `output` column verbatim, which is `repr`-style
   (single-quoted strings). The published exemplar uses double quotes (`"bhihia"`), so
   `solve(exemplar) != exemplar.answer` as raw strings; the selftest compares those two as
   *literals*. Generated instances match exactly.
6. **Unbounded answer space, no chance floor.** A wrong answer does not identify which step was
   dropped in general, though the trace is fully diagnostic when it *is* emitted.
7. Not every depth is reachable: `generate()` raises `ValueError` (listing the populated buckets)
   for an unpopulated bucket, e.g. `generate(41, 0)`.

## Running

```
python3 task.py --selftest        # 200 random (depth, seed) pairs; exit 0 on pass (~6 s)
python3 task.py --demo            # 3 instances at DEPTHS[0] and 3 at DEPTHS[-1]
python3 task.py --buckets         # executed-line histogram over the 800 rows
```

`examples.txt` is the saved output of `--demo`.

`desk.json` was reviewed against this implementation and left unchanged; no errors were found.
(Its `knob.name` prose anticipated exactly this executed-line bucketing.)
