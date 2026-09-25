# boolean_expressions

Evaluate a nested Boolean formula over `True` / `False` / `not` / `and` / `or`.
Canonical form: BIG-Bench Hard task `boolean_expressions`.

Implements the AMENDMENT 3 contract of `../../SPEC.md`. Pure python stdlib, no network.

    python3 task.py --selftest     # exits 0 on pass
    python3 task.py --demo         # rendered instances -> examples.txt
    python3 task.py --depth 6 --seed 3

## Source and citation

Primary source (format, fixed set, published numbers):

> Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,
> Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, Jason Wei.
> *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* arXiv:2210.09261, 2022.
> Repo: <https://github.com/suzgunmirac/BIG-Bench-Hard> @ `9ee07bd481feebf959a6b59d61ea57bdcf30964d`, **MIT**.

Upstream task definition:

> Aarohi Srivastava et al. *Beyond the Imitation Game.* arXiv:2206.04615, 2022.
> `google/BIG-bench`, `bigbench/benchmark_tasks/boolean_expressions/` @ `092b196c1f8f14a54bbc62f24759d43bde46dd3b`, **Apache 2.0**.

Complexity class:

> Samuel R. Buss. *The Boolean Formula Value Problem Is in ALOGTIME.* STOC 1987.
> Boolean formula evaluation is the canonical NC¹-complete (ALOGTIME-complete) problem.

Full sourcing write-up, including the candidates that were rejected, is in `SOURCING.md`;
desk metrics for the published implementation are in `desk.json`.

## What was vendored vs. written

**Vendored** (`vendor/`, see SOURCING.md for commits/licences):

| path | used for |
|---|---|
| `vendor/BIG-Bench-Hard/bbh/boolean_expressions.json` | the fixed 250-example canonical set, exposed as `bbh_examples()` / `bbh_instances()`; also a 250-case differential test of the reference evaluator |
| `vendor/BIG-Bench-Hard/cot-prompts/boolean_expressions.txt` | the published 3-shot CoT prompt; the **first exemplar is read verbatim at runtime** to build `exemplars(k, seed)[0]` |
| `vendor/BIG-bench-google/.../boolean_expressions/` | consulted only, for the upstream knob names (`expression_lengths`, `allow_parentheses`, `binary_ops`) |
| `vendor/lm-evaluation-harness/lm_eval/tasks/bbh/` | consulted only, confirms the `"the answer is X"` extraction convention |

**Written from scratch** (no upstream code copied): the generator, the explicit-parenthesis
expression AST and renderer, the innermost-first reducer that produces the `= ... = ...` chains,
the trace renderer, the independent recursive-descent evaluator used by `solve()`, and the
selftest. No boolean-expression generator with a CoT scratchpad existed anywhere (SOURCING.md).

`exemplars()[0]` is read from the vendored file at import time; `published_trace.txt` is the
fallback, and a verbatim string constant in `task.py` is the last-resort fallback, so the module
still works if `vendor/` is stripped. The selftest asserts the exemplar is byte-identical to the
published lines.

## Format decision

The gold trace reproduces the **published BBH CoT format** verbatim in structure. Every trace is:

```
Let's think step by step.
Remember that (i) expressions inside brackets are always evaluated first and that (ii) the order of operations from highest priority to lowest priority is "not", "and", "or", respectively.
We first simplify this expression "Z" as follows: "Z = <full expression> = <skeleton>" where "A = ...", "B = ..." and "C = ...".
Let's evaluate A: A = <A's text> = <reduction> = ... = True.
Let's evaluate B: B = <B's skeleton, containing A> = <A substituted> = ... = False.
Plugging in C, we get: Z = <Z's skeleton> = ... = False. So the answer is False.
Answer: False
```

The first two lines and the `We first simplify ... where "A = ..."` sentence are copied word for
word from `cot-prompts/boolean_expressions.txt`. At depth 2 — the depth of the published exemplar
— the rendered trace is structurally identical to it, down to the punctuation.

Two things had to be **generalised**, because the published prompt only ever shows one level of
naming (its exemplars have named sub-expressions `A` and `B` that are both evaluated directly):

1. **Recursive naming.** For depth *d* the trace names *d−1* sub-expressions `A, B, C, …`,
   innermost first, each defined in terms of the previous one. The published one-level case is the
   *d = 2* special case of this, unchanged.
2. **The `where` clause gives each name's *skeleton*, not its full text** — i.e. `"B = ( A ) and True"`
   rather than `"B = ( not not True ) and True"`. At *d = 2* these coincide (A is a leaf), so this
   again matches the published exemplar exactly. Using skeletons keeps the trace Θ(depth) instead of
   Θ(depth²) tokens, which matters at depth 16.

Two deliberate small departures from the published text, both in the direction of correctness:

* **The published exemplars' inner reductions are sloppy and in one case wrong.** Exemplar 2 writes
  `B = not True and True = not (True and True) = not (True) = False`, which mis-applies precedence
  and gets the right answer by luck; exemplar 3 writes `not not (True) = not not False = True`.
  Generated traces use a principled leftmost-innermost reducer, so every `=` in a generated trace is
  a true equality. The selftest re-parses every intermediate expression with the independent
  evaluator and asserts it. The verbatim published exemplar is of course left untouched.
* **Redundant parentheses around a literal are dropped silently**, exactly as the published exemplar
  does (`not ( ( True ) ) = not True = False`), rather than being unwound one pair at a time.

The one harness-imposed normalisation from AMENDMENT 3 is applied: the trace ends with the
published closing `So the answer is X.` and *then* a final line `Answer: X`.

`Instance.prompt` is the problem statement only, in BBH's own wording — the bare expression with
its trailing ` is`, e.g. `not ( True ) and ( True ) is`. BBH's one-line task description is exposed
separately as `TASK_DESCRIPTION` for the harness to use as an instruction; it is not in the prompt.

## Depth semantics

**`depth` = the number of named sub-expression resolutions in the gold trace = `len(steps)` =
`len(states)`.** Equivalently: the length of the longest chain of dependent evaluations, measured at
the granularity of named sub-expressions.

This is exact rather than approximate, because the generator builds every instance as a **path, not
a tree**. Level *k* is produced by wrapping level *k−1* in new operators, so `A₁ → A₂ → … → A_{d−1} → Z`
is a chain with no branching: the dependency DAG over named sub-expressions *is* a path of length *d*,
so its critical path, its node count and `len(steps)` are all the same number. Nothing in the trace
can be evaluated out of order or in parallel.

Two further properties make the depth knob load-bearing rather than cosmetic:

* **Every level genuinely depends on the one below it** (knob `require_dependence`, on by default).
  A candidate level is rejected unless substituting `True` and `False` for the inner sub-expression
  gives different values — which rules out the absorbing forms `A and False` and `A or True` that
  would silently truncate the chain and make all the deeper state dead. Consequently the model must
  actually carry one bit forward through all *d* steps; there is no early exit.
* **Off-path work stays constant.** The literal operand introduced at each level
  (`… and not False`, `… or ( True )`) is a side computation of bounded size, so it adds width, not
  depth. Measured on 50 seeds per depth, prompt length and gold-trace length are both linear in
  depth (≈ 3.0 prompt words and ≈ 26 CoT words per level) and the elementary critical path — the
  longest chain of *primitive* reductions, `meta["elementary_depth"]` — grows at ≈ 1.6 primitive
  reductions per level:

  | depth | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
  |---|---|---|---|---|---|---|---|
  | prompt words | 11 | 14 | 17 | 23 | 29 | 45 | 56 |
  | gold-CoT words | 114 | 141 | 167 | 225 | 278 | 403 | 507 |
  | elementary critical path | 4.6 | 5.8 | 7.7 | 10.9 | 13.8 | 20.2 | 26.4 |

  So `depth` is within a constant factor (~1.6–2.3, flat across the sweep) of the true elementary
  serial depth. The named-sub-expression granularity is the right one to report because it is the
  granularity at which the *published* format emits state: one value per named sub-expression, one
  line per step. `meta["elementary_depth"]` carries the finer number if the analysis wants it.

`depth = 1` is supported (a single unnamed reduction line) but is outside `DEPTHS`; its rendering is
necessarily degenerate, since the published format has nothing to name.

The **fixed BBH set** is mapped onto the same scale by `bbh_instances()`, which parses each of the
250 published expressions, cuts named levels at the binary nodes on its critical path, and renders
the same trace format. The canonical set lands at depth 1–4 (9 / 96 / 121 / 24 examples), which is
the concrete statement of how shallow canonical BBH is. Note that BBH ships **no per-example CoT**
(only the 3 prompt exemplars), so those `steps` are ours; `answer` is BBH's own target, and the
selftest asserts all 250 agree with the independent evaluator.

## DEPTHS

    DEPTHS = [2, 3, 4, 6, 8, 12, 16]

* **2** is canonical BBH difficulty — the depth of the published exemplar, and the second-commonest
  depth in the 250-example set (96 of 250; the mode is 3, with 121). It anchors the sweep to the
  published numbers.
* **3–4** covers the rest of the canonical distribution (all 250 BBH items are ≤ 4).
* **6–8** is already past anything in the published data.
* **12–16** is where a 7B should fail. The reason for pushing this far: Suzgun et al. Table 3 has
  *text-davinci-002 at 90.0 % answer-only and 87.6 % with CoT*, PaLM-540B at 83.2 / 80.0, Codex at
  88.4 / 92.8 — i.e. at canonical depth the task is near ceiling and CoT does not help two of three
  model families. A sweep that stops at canonical depth would measure nothing. At depth 16 there
  are 16 strictly serial dependent reductions and ~56 tokens of prompt with 15 nesting levels; the
  no-CoT condition requires holding and flipping a bit 16 times with no scratchpad.
* Seven values, roughly geometric at the top, so the accuracy curve is sampled densely where it
  starts to bend and sparsely in the flat regions. Gold-CoT length stays under ~700 tokens at
  depth 16, so the whole sweep fits comfortably in context.

## ANSWER_FORMAT

    ANSWER_FORMAT = "exactly one word, either True or False"

The canonical BBH target is literally the string `True` or `False`, so the answer space is the
published one. `check()` follows the AMENDMENT 3 rule — take the last `Answer:` line, else the last
non-empty line — and then normalises tolerantly: strips markdown emphasis, backticks, quotes and a
trailing period, case-folds, takes the last whitespace-separated token, and additionally honours the
BBH convention by extracting the text after a trailing `the answer is`. Anything that does not
normalise to exactly `true` or `false` (e.g. `Answer: maybe`, an empty completion) is rejected rather
than scored, so a non-answer is not silently counted as a wrong answer.

## Knobs

`depth` is not a knob. `KNOBS`:

| knob | default | effect |
|---|---|---|
| `leaf_size` | `2` | operators in the innermost named sub-expression `A` |
| `wrap_ops` | `1` | operators added at each subsequent level; raises tokens/step without raising serial depth |
| `operand_ops` | `1` | max operators in the literal operand introduced at each level (so operands are `not False`, `( True )`, … and not just bare literals) |
| `extra_parens` | `0.30` | probability of a redundant parenthesis pair at a node, matching BBH's `( ( … ) )` surface style |
| `require_dependence` | `True` | reject levels whose value does not depend on the inner level — see Depth semantics |
| `ops` | `("and", "or")` | binary operators available |

The selftest sweeps 9 knob settings × 3 depths × 6 seeds.

## Redaction

`redact_prompt(inst, k)` implements SPEC AMENDMENT 4 (prompt blinding), and
`REDACTION_MEANINGFUL = True`. Here the expression *is* the operator list, so what gets redacted are
the sub-expressions steps 1..k have already resolved — the named groups `A`, `B`, ... in evaluation
order — replaced **in place** by the literal placeholder `[…]`, with the enclosing structure and
every not-yet-evaluated operand left byte-identical. Because the generator builds a path rather than
a tree, those *k* levels are nested inside one another and therefore form a single contiguous span
(the outermost of them, level `k-1`), so one placeholder covers the removed span, as the amendment
allows. `redact_prompt(inst, 0)` returns `inst.prompt` unchanged; `redact_prompt(inst, inst.depth)`
returns `[…] is`, since the last step resolves `Z` itself — no literal, no operator and no
parenthesis of the original survives, only the question. The static material (BBH's two boilerplate
"Remember that ..." lines) is not part of `prompt` in the first place and is unaffected.

A model given the redacted prompt plus the published-format trace for steps 1..k can still finish:
step *k* ends with `... = <value>.`, which is exactly the one bit the placeholder stands for. A model
that was silently re-reading the expression instead of carrying state cannot.

Rendered example, `generate(depth=6, seed=1)` at `k = depth // 2 = 3`:

```
prompt              : not not ( not ( not not ( True or False ) ) and True ) is
redact_prompt(·, 3) : not not ( […] and True ) is
states              : A=False | B=True | C=False | D=False | E=True | Z=False
```

Steps 1–3 resolved `A`, `B`, `C`; `C = not ( not not ( True or False ) )` is the span that
disappears, and its value (`False`) is the last thing the trace prefix asserts. `D`, `E` and `Z` —
the `and True`, the two `not`s and the brackets — are still there to be applied.

The levels are recovered from `inst.steps` (the first `=` clause of each step is that level's
skeleton) rather than from a saved tree, so redaction works identically for generated instances, the
verbatim published exemplar and all 250 vendored BBH examples; the selftest checks all three.

## Caveats

* **Binary answer space.** `answer_space = 2`, so the no-CoT floor is 50 % and the CoT gap is
  compressed by chance accuracy relative to tasks with a large answer space. This is a property of
  the canonical published task, not of this implementation; per SPEC "What NOT to do", the canonical
  version is implemented as-is. If a wider answer space is wanted, that is a separate slug
  (e.g. reporting the vector of level values, or a *k*-valued logic). Two mitigations are already
  available without changing the task: `states` gives a per-step gold value, so a step-level
  diagnostic is possible even when the final bit is right by luck, and `meta["level_values"]` lets a
  scorer localise where a wrong trace diverged.
* **The state really is one bit.** Because `require_dependence` forces every level to be either the
  identity or the negation of the level below, `Z = A₁ XOR (parity of effective negations)`. The
  serial state a model must carry is exactly 1 bit — which is the point (it is the cleanest possible
  instance of load-bearing serial state) but also means there is an available shortcut: track
  polarity instead of values. The shortcut is still Θ(depth) serial work and still requires
  evaluating each level's literal operand to know whether that level is identity or negation, so it
  does not collapse the depth knob; but an interpretability analysis should expect polarity-tracking,
  not value-tracking, to be one plausible learned algorithm. Setting `require_dependence=False`
  restores absorbing operators (and with them instances whose deep state is dead), if a contrast is
  wanted.
* **Contamination.** `desk.json` records `contamination_risk: medium` for the canonical set — the
  250 examples are mirrored across at least six HuggingFace dataset repos and are almost certainly
  in pretraining corpora, despite the canary string. The generator exists precisely so the sweep can
  be run on fresh instances; `bbh_instances()` is provided so the two can be compared directly.
* **`solve()` independence.** `solve()` uses a separate direct-evaluating recursive-descent parser
  over the surface string and never touches the generator's AST, `steps`, `states`, `answer` or
  `meta`. It is verified against all 250 published BBH targets in the selftest — an external ground
  truth, not a self-consistency check. The module does contain a *second* string parser
  (`_parse_ast`), but it is used only to build instances from the fixed BBH set, never by `solve()`.
* **Reconstructed traces for the fixed set.** `bbh_instances()[i].steps` are generated by this
  module, not published; only `exemplars()[0]` is verbatim. `meta["trace_is_reconstructed"]` flags it.
* **`desk.json` was not modified** — no errors found in it against the implementation. Its
  `tokens_flat_across_knob: null` could now be filled in as `true` for the `depth` knob (the table
  above shows ~26 CoT words per level, flat across the sweep), but that is a token count, not a desk
  estimate, so it is left for whoever re-runs the tokenizer pass.
* **Depth ceiling.** Names are single letters, so `depth ≤ 27`; `generate()` raises above that.
