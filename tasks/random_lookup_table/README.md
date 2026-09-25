# random_lookup_table

Iterated application of random lookup tables over a small alphabet. The prompt gives a pool of
full lookup tables over `[N]` (default `N = 10`), a start symbol, and a sequence of `depth` task
tokens naming which table to apply at each step. The model carries one alphabet symbol across the
steps; the answer is the final symbol.

Two renderings of the same instance are available via the `format` knob: `"published"` (default,
the paper's token stream) and `"ergonomic"` (AMENDMENT 5, plain language). See "Ergonomic
variant".

## Source and citation

Primary source (as recorded in `desk.json`):

> Rahul Ramesh, Ekdeep Singh Lubana, Mikail Khona, Robert P. Dick, Hidenori Tanaka et al.,
> *Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable
> Tasks*, arXiv:2311.12997, ICML 2024. https://arxiv.org/abs/2311.12997

Related / contrast (lookup-table & associative-recall lineage, not the source of this format):

> Simran Arora, Sabri Eyuboglu et al., *Zoology: Measuring and Improving Recall in Efficient
> Language Models*, arXiv:2312.04927.

Theory:

> David A. Barrington, *Bounded-width polynomial-size branching programs recognize exactly those
> languages in NC1*, JCSS 1989.

## What was vendored vs. written

Vendored (phase 1, unmodified):

- `vendor/compositional_capabilities/` — https://github.com/rahul13ramesh/compositional_capabilities
  @ `54256841ef5f3af3a39d47288d071f7d04b6da5e`, **MIT**. `synthetic/functions.py` +
  `synthetic/generator.py` are the origin of the task's structure (fresh pool of bijective lookup
  tables per composition depth, `T{d}_{i}` task tokens, step-by-step vs. direct document format)
  and the source of `published_trace.txt`.
- `vendor/zoology/` — https://github.com/HazyResearch/zoology @ `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`,
  **Apache-2.0**. MQAR is a parallel-query retrieval task, not serial composition; cited for
  lineage only. No code from it is used.

Written from scratch: all of `task.py`. Per `SOURCING.md` the vendored generator needs more than a
thin wrapper — it is numpy/torch/omegaconf-bound (the spec forbids those), its `reduce_functions`
composes over the whole alphabet in batched-array form rather than tracking one scalar state, its
documents contain no lookup tables (the paper's from-scratch models learn the tables during
training, whereas an off-the-shelf LLM must read them from the prompt), and it has no
`solve`/`check`/`Instance` plumbing. `task.py` reimplements the same construction in stdlib and
reads nothing from `vendor/` at runtime.

## Format decision

The published step-by-step document (`config/gen/conf.yaml` with `direct: False`) is one flat
token stream:

```
S T0_3 T1_0 T2_0 T3_3 T4_2  X5X9X1X5X7X8  X9X7X5X9X3X2  X9X7X5X9X3X2  X9X7X5X9X3X2  X7X9X5X7X0X3  X8X7X4X8X6X1
```

`S`, then one `T{d}_{i}` task token per composition step naming the pool member applied at depth
`d`, then the initial length-`seq_len` vector, then one vector per step. `task.py` keeps that
vocabulary (`S`, `T{d}_{i}`, `X{a}`) and that per-step emission, with three deviations, all
deliberate:

1. **Plain ASCII digits, not the repo's Unicode subscripts.** `X₅`/`T₀_₃` come from the repo's
   `SyntheticData.decode()` pretty-printer, not from the data: the trained model consumes integer
   token ids, so the subscripts are a display choice with no standing in the published task.
   `desk.json`'s own notes measured them at 18.6–25.2 tokens per depth-step under
   gemma2/llama3/qwen2 (rare multi-byte glyphs) and flag them as "not what should be used if
   actually prompting an open-weight LLM". Plain digits are the same content, ~6× cheaper.
2. **One step per line, prefixed by that step's task token** (`T2_1 X6`), instead of all vectors
   space-separated on one line. Same content — the task token is already in the document header —
   re-laid out. On one line the model must align the *k*-th header token with the *k*-th emitted
   vector by counting; that is a positional-alignment burden, not the state-tracking the bench is
   trying to isolate. Re-emitting the token beside its output also puts the state at a fixed
   position (last token of the line), which `step_spans()` and per-step attention-blinding need.
3. **The lookup tables are written into the prompt.** The published documents contain no tables at
   all; the tables are weights, learned across 100k training documents. An LLM evaluated zero-shot
   cannot have them, so they must be stated. This is the "non-trivial wrapper" `SOURCING.md`
   anticipated. The surrounding English (the `Alphabet:` line, the `T{d}_{i} means ...` gloss, the
   closing question) is ours: the published task has no natural-language wording whatsoever.

Per AMENDMENT 3's one normalization, `format_cot()` ends with a final `Answer: X8` line. The
published format has no closing sentence, so nothing else is appended.

Everything in this section describes `format="published"`, the default. AMENDMENT 5 adds a second
rendering of the identical instance for models that were never trained on this token stream; see
"Ergonomic variant".

## Depth semantics

`depth` = number of function applications = number of task tokens in the document = number of
state updates. `len(steps) == len(states) == depth` exactly. This is the repo's own
`function.depth` knob (`config/gen/conf.yaml`: `depth: 5`).

`DEPTHS = [2, 4, 8, 16, 24, 32, 48]`.

- 2 is trivial and should be at ceiling with and without CoT — it calibrates format compliance.
- 4 and 8 bracket the paper's own `function.depth = 5`.
- No-CoT should collapse by ~8: composing 8 random permutations of a 10-symbol alphabet in a
  single forward pass with no intermediate tokens is the thing the bench is measuring.
- With CoT each step is a table lookup plus a copy, so accuracy decays roughly geometrically in
  per-step error; 24–48 is where a 7B's per-step slips should compound into visible failure while
  the prompt stays a constant ~30 table entries (see the knob note below).

## ANSWER_FORMAT

`"a single alphabet symbol written like X7"`. The answer space is exactly the alphabet, so
`answer_space = 10` at the default `n_alphabets` (matching `desk.json`) and the chance floor is
`1/N`. Measured over 3000 instances the marginal answer distribution is flat (best constant guess
0.104–0.115 across all depths), so there is no free lunch for a model that ignores the document.

`check()` takes the last `Answer:` line, falling back to the last non-empty line, and normalizes:
case-insensitive, trailing period stripped, internal spaces ignored, and a bare digit accepted for
`X<digit>` (so `9`, `x9`, `X9`, `Answer: X9.` all match).

## Knobs

| knob | default | note |
|---|---|---|
| `n_alphabets` | 10 | repo `n_alphabets`. `state_bits = log2(N)` |
| `n_functions` | 3 | repo `function.n_functions`. Pool size |
| `bijective` | **True** | repo ships `np.random.permutation`. See caveats |
| `fresh_tables_per_step` | False | `True` = repo's shipped `function.repeat: False` |
| `include_identity` | False | repo's choice 0. `True` reinstates it |
| `seq_len` | 1 | repo `seq_len: 6`. 1 = the single-tracked-symbol definition |
| `format` | `"published"` | `"ergonomic"` = the SFT-free rendering, below (AMENDMENT 5) |

`depth` is not a knob. At the defaults the prompt is flat across depth (three tables, ~30 entries)
except for the depth-long task-token header, so a depth sweep is not confounded by prompt length.

## Ergonomic variant

AMENDMENT 5. This is a `†` task: the published format was taught by training from scratch, and
batch 1 showed small models cannot pick it up from exemplars. `format="ergonomic"` is a second
rendering of the *same* instance designed to be computed in by a prompted model.

**What changes.** Only the wording of `prompt` and `steps`. The prompt drops the `S`/`T{d}_{i}`
document vocabulary for plain language: the alphabet as a sentence, each lookup table as a
readable mapping (`F1: X0 -> X5, X1 -> X8, ...`), the start symbol on its own line, and the
operator sequence as an explicit numbered list, one line per step, so the model never has to count
positions to find out which table step *k* uses. The trace line restates everything a step
consumes and produces:

```
step 2: apply F3 to X7. F3: X7 -> X0. Now at X0.
```

— the step index, the operator, the state going in, the single table entry that is being read, and
the state coming out. The state appears twice per line and is never implicit, which is the point:
the model is asked to *write down* the thing it must carry. Under `include_identity=True` a no-op
step reads `step 2: apply F0 to X7. F0 is the identity. Now at X7.`; under `seq_len > 1` the line
lists one entry per distinct symbol in the vector (`F2: X0 -> X8, X6 -> X9, X3 -> X0`). Under
`fresh_tables_per_step=True` the tables keep their `T{d}_{i}` names, since they genuinely differ
per step.

**What does not change.** `generate(depth, seed, **knobs)` draws the tables, the table order and
the start symbol from an rng keyed on everything *except* `format` (`_RNG_IRRELEVANT`), so a given
`(depth, seed)` is literally the same instance in both formats: same `answer`, same `states`, same
`depth`, same `check()`, same `solve()`. `--selftest` asserts this on all 200 instances — each
trial generates both renderings and compares `meta["start"]`, `meta["choices"]`, `answer`,
`states`, `depth`, `solve()`, `check(gold)` and `check(wrong)`, and requires the two prompts and
traces to actually differ. (Deliberately breaking the rng exclusion makes those assertions fire.)

`solve()` dispatches on the rendered prompt — `Document: S ...` present means published — and each
branch parses its own prompt from scratch, so the differential check still covers both renderings.

`exemplars(k, seed, **knobs)` forwards `format` (and every other knob) to `generate`. The
AMENDMENT 3 published exemplar is a reconstruction of the published token stream and has no
ergonomic counterpart, so under `format="ergonomic"` it is skipped and all `k` exemplars are
generated at depth 4.

Rendered instance, `generate(depth=4, seed=1000, format="ergonomic")` — the same instance whose
published rendering is the redaction example below:

```
There are 10 symbols: X0, X1, X2, X3, X4, X5, X6, X7, X8, X9.

There are 3 lookup tables, and the same ones are used at every step.  A lookup table says what each symbol turns into:

F1: X0 -> X5, X1 -> X8, X2 -> X6, X3 -> X0, X4 -> X3, X5 -> X2, X6 -> X1, X7 -> X7, X8 -> X4, X9 -> X9
F2: X0 -> X2, X1 -> X9, X2 -> X7, X3 -> X0, X4 -> X4, X5 -> X1, X6 -> X5, X7 -> X3, X8 -> X6, X9 -> X8
F3: X0 -> X9, X1 -> X1, X2 -> X3, X3 -> X8, X4 -> X5, X5 -> X2, X6 -> X6, X7 -> X7, X8 -> X4, X9 -> X0

Start at X7.

Apply the tables in this order, one table per step, each one to what you are holding after the previous step:
step 1: F2
step 2: F1
step 3: F1
step 4: F1

What do you end at after step 4?
```

with gold CoT

```
step 1: apply F2 to X7. F2: X7 -> X3. Now at X3.
step 2: apply F1 to X3. F1: X3 -> X0. Now at X0.
step 3: apply F1 to X0. F1: X0 -> X5. Now at X5.
step 4: apply F1 to X5. F1: X5 -> X2. Now at X2.
Answer: X2
```

**Cost.** The ergonomic trace is ~5x the published trace in characters per step (`step 3: apply F1
to X0. F1: X0 -> X5. Now at X5.` vs `T2_1 X6`), which matters for the depth-48 rows; it is still
far cheaper than the repo pretty-printer's Unicode subscripts that `desk.json` measured at
18.6-25.2 tokens/step. `desk.json` describes the published format and was not changed.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4) removes what steps 1..k consumed and leaves everything
needed to continue from step k+1. Removed, each replaced by one literal `[…]`: the **start
symbol** — the initial state, consumed by step 1 — in both places the prompt states it (the
`Document:` line and the `Starting from the symbol ...` sentence), and the **first k task tokens**
of the document header, which are what say which table each of steps 1..k applied. Kept: the
lookup tables, which are static material the model still needs for steps k+1..depth and are not
state; the task tokens for steps k+1..depth; the alphabet; the `T{d}_{i}` gloss; and the question.
`redact_prompt(inst, 0)` returns `inst.prompt` unchanged; at `k = depth` the document header is all
placeholders and the only way to answer is to read the state out of the supplied trace prefix.

Both formats are redacted the same way, on the two places each states the removed material. In
the ergonomic rendering that is the `Start at ...` line and the table name on the first `k` lines
of the `step i: <table>` order list:

```
Start at […].

Apply the tables in this order, one table per step, each one to what you are holding after the previous step:
step 1: […]
step 2: […]
step 3: F1
step 4: F1
```

`REDACTION_MEANINGFUL = True`. Under `fresh_tables_per_step=True` the table block is keyed
`T{d}_{i}` rather than `F{i}`, but it lists *every* pool member for *every* step and so discloses
nothing about which member step `d` applied — that choice exists only in the document header — so
the block stays intact there too.

Rendered example, `generate(depth=4, seed=1000)` at `k = depth//2 = 2` (the answer is `X2`; a
model given this plus the gold trace for steps 1..2 can finish, one given this alone cannot):

```
Alphabet: X0 X1 X2 X3 X4 X5 X6 X7 X8 X9

Lookup tables (the same tables are used at every step):
F1: X0->X5 X1->X8 X2->X6 X3->X0 X4->X3 X5->X2 X6->X1 X7->X7 X8->X4 X9->X9
F2: X0->X2 X1->X9 X2->X7 X3->X0 X4->X4 X5->X1 X6->X5 X7->X3 X8->X6 X9->X8
F3: X0->X9 X1->X1 X2->X3 X3->X8 X4->X5 X5->X2 X6->X6 X7->X7 X8->X4 X9->X0

Document: S […] […] T2_1 T3_1 […]

The task token T{d}_{i} means "at step d, apply table F{i}".
Starting from the symbol […], apply the tables named by T0_*, T1_*, ... in that order, each to the result of the previous step.
What is the symbol after the last step?
```

`--selftest` checks, over 20 instances x both formats at `k` in `{0, 1, depth//2, depth}`: `k = 0` is the
identity; every `k > 0` changes the prompt and contains `[…]`; no `X<digit>` survives on the
`Document:` or `Starting from` lines; the first `k` header slots are placeholders and the
remaining ones are verbatim; at `k = depth` no `T{d}_{i}` token is left in the document and the
question is still there; the alphabet and table lines are byte-identical at every `k`; and `k`
outside `[0, depth]` raises. The ergonomic branch asserts the analogue: no `X<digit>` survives on
the `Start at` line, the order list still has exactly `depth` entries, its first `k` are `[…]` and
the rest are byte-identical to the original.

## Corruption

`corrupt_step(inst, k, seed) -> (step_text, corrupted_state)` (AMENDMENT 6) rewrites step `k`
(1-based) so that the state it *reports* is wrong, and returns that wrong state in the same
canonical form as `inst.states`. It is the mistake-propagation probe: prefill the gold trace with
step `k` replaced by `step_text`, let the model continue, and score against the **original**
answer — a model that is genuinely carrying the state forward reads the corrupted symbol and ends
somewhere else, while one that is re-deriving the answer (or guessing) is unmoved.

The corruption is one **misread table entry**: a distinct input symbol of the state going into
step `k` is chosen, and the output the step reports for it is moved to a different symbol of the
same alphabet. At the default `seq_len=1` that is exactly "the step names the wrong result
symbol". At `seq_len > 1` every position holding that input symbol moves together, because that
is what a single wrong lookup does to a vector — and it keeps the ergonomic line's table entry and
its `Now at` telling one story.

The step's **action is untouched** in both formats; only the reported result moves. In the
published rendering that means the task token survives and the state token is replaced:

```
gold       T1_1 X0
corrupted  T1_1 X9
```

In the ergonomic rendering the state is written twice — in the table entry and in `Now at` — so
both places are rewritten together, never one of them:

```
gold       step 2: apply F1 to X3. F1: X3 -> X0. Now at X0.
corrupted  step 2: apply F1 to X3. F1: X3 -> X9. Now at X9.
```

(`generate(depth=4, seed=1000)`, `k=2`, `seed=7` — the same instance as the sections above. The
true answer is `X2`; F1 fixes X9, so a model that actually reads the corrupted state answers `X9`
and the propagation is visible.) Under `include_identity=True` a no-op step states no table entry
— `F0 is the identity.` is action text, not a result — so only `Now at` moves there.

Determinism is in `(inst, k, seed)`, and the rng key deliberately excludes `format`, exactly as
`generate`'s does: the same instance corrupts to the **same wrong state** in either rendering, so
a published and an ergonomic mistake run are comparable. `k` outside `[1, depth]` raises.

`--selftest` checks, over 20 instances x both formats at `k` in `{1, depth//2, depth}`: the
corrupted state differs from `states[k-1]`; the step text differs from `steps[k-1]`; the state is
well formed (right number of positions, every symbol in the alphabet, round-trips through the
canonical spelling); the step line still parses against the format's anchored step-line regex
with its action groups byte-identical to the gold line, and the state it reports equals the
returned state; in the ergonomic branch the rewritten table entry covers exactly the distinct
input symbols and maps the input vector onto the reported output vector (so the line is internally
consistent, not merely changed); both formats return the same corrupted state; two calls agree.
Mutation-tested: not rewriting the table entry, returning the true state, ignoring `meta["format"]`,
touching the action text, keying the rng on `format`, and dropping the range check each fire.

## Caveats

**`bijective` defaults to True, against `SOURCING.md`'s recommendation, and this is the one place
this implementation overrides the sourcing pass.** `SOURCING.md` argued the bijection restriction
should be dropped to get arbitrary `[N] -> [N]` maps. Non-bijective maps are implemented
(`bijective=False`) but cannot be the default, because the task destroys itself as depth grows —
arbitrary maps are not information-preserving, so the composition collapses toward a constant
function. Measured over 2000 instances at `N=10, n_functions=3` (after rejecting pools with a
common fixed point):

| depth | P(perturbing one step's choice changes the answer) | P(answer does not depend on the start symbol) |
|---|---|---|
| 8  | 0.64 / 0.90 | 0.16 / 0.00 |
| 16 | 0.47 / 0.90 | 0.51 / 0.00 |
| 32 | 0.30 / 0.90 | 0.84 / 0.00 |

(non-bijective / bijective). At depth 32 the non-bijective task is answerable without tracking
state 84% of the time: a model that guesses the absorbing symbol is right, and the depth knob
stops measuring serial state. Bijections hold per-step sensitivity flat at 0.90 = (m-1)/m at every
depth, which is exactly the "every step is load-bearing" property the bench needs — and they are
also what the paper actually published. `desk.json`'s `theory_citation` already flags the
collapse risk; these numbers are its quantification. Nothing in `desk.json` was changed.

**Overlap with `s5_composition`.** Both are word problems in a symmetric group. They differ in
alphabet size (10 vs 5), in that the generators here are given as explicit lookup tables in the
prompt rather than as cycle/permutation notation, and in the published format they reproduce. If
the review wants a genuinely non-group variant, `bijective=False` at depth ≤ 8 is the usable range.

**`exemplars(k, seed)[0]` is a partial reconstruction.** AMENDMENT 3 requires the published
exemplar first. The published document applies one function elementwise to `seq_len=6`
*independent* positions, so any single column of it is itself a verbatim published single-symbol
trace of that instance; `exemplars()[0]` takes column 0. Verbatim from `published_trace.txt`:
depth 5, the task tokens `T0_3 T1_0 T2_0 T3_3 T4_2`, the start symbol `X5`, and every state
`X9 X9 X9 X7 X8`. Reconstructed: the lookup tables. The trace pins 5 of 10 entries of each applied
table (all six columns are used as constraints and are mutually consistent); the rest, and the
pool members the trace never exercises, are completed deterministically from a fixed seed. This is
recorded in that instance's `meta["reconstruction"]`.

This applies to `format="published"` only; `exemplars(k, seed, format="ergonomic")` skips the
reconstruction and generates all `k`, so the caveat below about a heterogeneous few-shot prompt
does not arise there.

Consequence for the harness: because the published instance uses `fresh_tables_per_step=True` and
`include_identity=True`, `exemplars()[0]`'s prompt lists 15 per-step tables and has a different
shape from the generated instances (which share one 3-table pool). A harness that wants a
homogeneous few-shot prompt should use `exemplars(k + 1, seed)[1:]`; the remaining exemplars are
generated at depth 4 with the default knobs.

**Identity steps are off by default.** The paper includes identity as choice 0 of each pool
(`T1_0`, `T2_0` in the published trace are no-ops), because it studies compositional
generalization over a task algebra where the identity matters. For a depth sweep it dilutes the
knob — at `n_functions=3` a quarter of steps would be no-ops, so nominal depth would overstate
effective depth by 4/3. `include_identity=True` restores the published behaviour.

**Upstream bug, not fixed here.** `vendor/compositional_capabilities/synthetic/functions.py`'s
`get_train_functions()` calls `random.sample()` on a `set`, which Python 3.11+ rejects. It was
worked around with a runtime-only monkey-patch when `published_trace.txt` was generated; the
vendored copy is left untouched. `task.py` does not import it.

## Running

```
python3 task.py --selftest      # 200 instances across 7 knob settings x both formats; exit 0 on pass
python3 task.py --demo          # 3 at DEPTHS[0]=2, 3 at DEPTHS[-1]=48, then one instance per
                                # format at depth 6 -> examples.txt
```

Pure python + stdlib; no network; nothing is read from `vendor/` at runtime.
