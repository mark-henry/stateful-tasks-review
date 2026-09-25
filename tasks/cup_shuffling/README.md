# cup_shuffling — BBH `tracking_shuffled_objects`

N people each start holding one object; a sequence of pairwise swaps is described; the model must say
which object one named person ends up with. This is composition of transpositions in S_n.

## Source and citation

Primary source (the published implementation this reproduces):

> Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,
> Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, Jason Wei.
> *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* arXiv:2210.09261 (2022).
> https://github.com/suzgunmirac/BIG-Bench-Hard

Underlying BIG-bench task:

> James Simon, `tracking_shuffled_objects`, in Srivastava et al., *Beyond the Imitation Game* (BIG-bench),
> https://github.com/google/BIG-bench — task README documents the generation constraints.

Licenses: BBH is MIT (`vendor/bbh/LICENSE`); BIG-bench is Apache-2.0. `task.py` is original work.

## Vendored vs. written

**Vendored** (phase 1, see `SOURCING.md` for the full survey):

| path | what it is |
|---|---|
| `vendor/bbh/data/tracking_shuffled_objects_{three,five,seven}_objects.json` | the fixed BBH eval sets, 250 items each (750 total), `{input, target}` |
| `vendor/bbh/cot-prompts/*.txt` | the 3-shot CoT prompt prefix used for the paper's numbers. **All three files are byte-identical** (md5 `60bed715…`) — BBH ships the 3-object exemplars for the 5- and 7-object subtasks too |
| `vendor/bigbench/tracking_shuffled_objects/` | origin task: README with the swap-sampling constraints and context list; `task.json` |
| `vendor/lm-evaluation-harness/` | harness YAML wrappers; confirms the `So the answer is (X).` extraction convention |
| `vendor/paper/suzgun2022.pdf` | Table 3 / Figure 5, cited in `desk.json.published_data` |
| `published_trace.txt` | the first BBH CoT exemplar, verbatim |

**Written** (`task.py`, pure stdlib):

- A fresh procedural generator at arbitrary depth and n ∈ {3..7}, reproducing all five published
  contexts and BIG-bench's two swap-sampling constraints. No generation script was ever released
  (BIG-bench's README: "a .py script (not included but available upon request)"), so this is from scratch.
- An independent reference `solve()`, `check()`, `format_cot()`, `step_spans()`, `exemplars()`.
- `load_bbh(objects)` — reads the vendored fixed sets and rebuilds full Instances **with gold CoT**
  (the published data ships only `input`/`target`, no per-item scratchpad). Used as a differential test.

## Format decision

The published format is reproduced exactly. `format_cot()` emits:

```
(0) At the start: Alice: yellow, Bob: blue, Claire: pink.
(1) Claire and Alice swap balls: Alice: pink, Bob: blue, Claire: yellow.
(2) Alice and Bob swap balls: Alice: blue, Bob: pink, Claire: yellow.
(3) Claire and Bob swap balls: Alice: blue, Bob: yellow, Claire: pink.
At the end of the game, Bob has the yellow ball. So the answer is (A).
Answer: (A)
```

Decisions inside that:

- **`steps` are the numbered swap lines only.** `(0) At the start: …` is the *initial* state, not a state
  update, so it lives in `meta["initial_line"]` and is emitted by `format_cot()` as the trace header.
  This keeps `len(steps) == len(states) == depth` as the AMENDMENT-3 selftest requires.
- The plain-English closing sentence + `So the answer is (X).` is `meta["closing"]` + the published tail;
  the trailing `Answer: (X)` line is the one harness-imposed normalization, per AMENDMENT 3.
- **State token form** follows the published scratchpad: colour-based items are abbreviated to the
  colour (`yellow ball` → `yellow`, `red present` → `red`); names, book titles and soccer positions are
  written out (`Lola`, `Ulysses`, `goalkeeper`). BBH's own exemplars only demonstrate the ball and dance
  contexts; the other three follow the same rule by extension.
- `states[k]` is the compact permutation, `"pink blue yellow"` (short item per person, person order) —
  the same information as the step line, without the person labels.
- `Instance.prompt` is the BBH `input` field verbatim in form: problem statement + `Options:` block,
  no scratchpad instruction, no exemplars. The harness adds `A: Let's think step by step.`
- `exemplars(k, seed)[0]` is parsed out of `published_trace.txt` at runtime and is **verbatim** — the
  selftest round-trips it back to the file byte-for-byte, including BBH's stray double space in
  `(2)  Alice and Bob swap balls`. The remaining exemplars are generated at depth 3, n = 3.

## Depth semantics

**depth = the number of pairwise swaps.** BBH itself always uses depth == n (3 swaps for 3 objects,
5 for 5, 7 for 7); here depth is decoupled from n so the sweep can go deeper than the published sets.
`len(steps) == len(states) == depth` exactly.

Swap sequences follow BIG-bench's two documented constraints:

1. every person is involved in at least one swap;
2. the same two people never swap twice in a row.

Constraint 1 is unsatisfiable when `2 * depth < n`, so it is enforced only when `2 * depth >= n`
and silently relaxed below that (e.g. n = 7, depth = 2). Constraint 2 is why `objects >= 3`: with
n = 2 there is only one possible pair, so "never twice in a row" cannot hold past depth 1.

### DEPTHS

`DEPTHS = [2, 3, 5, 8, 12, 16, 24]`

- **2** — below the published minimum; a sanity floor where even no-CoT should be well above chance.
- **3** — exactly the published 3-object setting, so the sweep contains the BBH operating point and the
  fresh generator's numbers are directly comparable to Suzgun Table 3.
- **5, 8** — the published 5- and 7-object depths and just past them; this is where Suzgun's answer-only
  prompting is already at the random baseline while CoT is not.
- **12, 16** — beyond anything published; the regime where the CoT gap should be largest.
- **24** — long enough that a 7B model is expected to drop a step or lose track of the running state even
  with CoT (24 serial state updates, ~55 CoT tokens each ≈ 1.3k scratchpad tokens).

Note that with n = 3 the *state space* stays at 3! = 6 regardless of depth — extra depth buys serial
steps, not state width. Widening the state is the `objects` knob's job, not depth's. A full study should
sweep both.

## ANSWER_FORMAT

```
a multiple-choice option letter in parentheses, like (A)
```

The published task is multiple-choice and BBH's `target` is the parenthesized letter (`"(B)"`), so
`Instance.answer` is the letter and the `Options:` block stays in the prompt. The underlying object is
kept in `meta["answer_object"]` (`"yellow ball"`), `meta["answer_object_short"]` (`"yellow"`) and the
full letter→object map in `meta["options"]`, so a harness that wants free-form scoring has it.

Option letters follow BBH's convention: **option k is the object initially held by the k-th person
listed**. That is also what makes `solve()` cheap.

`check()` takes the last `Answer:` line (else the last non-empty line) and accepts `(A)`, `A`, `A.`,
`(A).`, a `…answer is (A)…` sentence, a leading `(A) yellow ball`, or the bare option text; anything
else is a miss. Confirmed to reject `Answer: banana`.

## Reference solver

`solve()` reads **only `inst.prompt`** — never `meta`, `steps` or `states` — and uses a different
algorithm from `generate()`: rather than simulating the permutation forward, it walks the swap list
*backwards* from the queried person to whoever started with the object that person ends up with, and
returns that person's index as the option letter. Independent parse, independent direction.

## Redaction

`redact_prompt(inst, k)` (SPEC AMENDMENT 4, `REDACTION_MEANINGFUL = True`) returns the prompt with
everything needed to *recompute* the state after swap `k` removed, and everything needed to *continue*
from swap `k+1` left in place. Two things go: the initial assignment clause — everything after the colon
in the opening sentence, i.e. who starts with which object — and the first `k` swap clauses, each
replaced by one literal `[…]` (so a redacted prompt carries exactly `k + 1` placeholders). Everything
else is untouched: the cast and the scenario lead-in, swaps `k+1 … depth`, the question
(`At the end of the game, Claire has the`) and the whole `Options:` block. `redact_prompt(inst, 0)` is
`inst.prompt` verbatim; at `k = depth` no operator and no initial state survive. The redaction is purely
textual (it works on `load_bbh()` items and the verbatim published exemplar too), and the published CoT
format already restates the swap on every step line, so the trace prefix the harness supplies is
self-sufficient — which is the point.

One caveat, unavoidable given the amendment says to keep the options block: by BBH convention option `k`
is the object initially held by the `k`-th person listed, so the *order* of the options still encodes the
initial assignment. Nothing in the redacted prompt states that convention, but a model that has memorized
BBH could exploit it. A stricter variant would shuffle the options (and re-letter the answer); that is
left out here because it changes `answer`, which AMENDMENT 3 fixes.

Rendered example, `generate(5, 3, objects=3, scenario="ball")` at `k = depth//2 = 2`:

```
Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: […].
As the game progresses, pairs of players trade balls. […] […] Then, Alice and Bob swap balls. Then, Alice and Claire swap balls. Finally, Bob and Claire swap balls. At the end of the game, Claire has the
Options:
(A) brown ball
(B) white ball
(C) blue ball
```

The harness pairs that with the published-format trace prefix for steps 1..k, which carries the state:

```
(0) At the start: Alice: brown, Bob: white, Claire: blue.
(1) Alice and Claire swap balls: Alice: blue, Bob: white, Claire: brown.
(2) Bob and Claire swap balls: Alice: blue, Bob: brown, Claire: white.
```

## Corruption

`corrupt_step(inst, k, seed)` (SPEC AMENDMENT 6) returns `(step_text, corrupted_state)` for step `k`
(1-based) with the reported assignment rewritten to a plausible *wrong* one. The state here is a
permutation, so the minimal edit is a transposition: two people's objects are exchanged relative to the
truth. The result is still a legal assignment — same people in the same order, same object set, each
object held exactly once — just not the right one. The action clause (`(3) Alice and Bob swap balls:`)
is left byte-identical; only what follows the colon changes. `corrupted_state` is in the same canonical
form as `inst.states` (short object names, space-joined, in the prompt's person order). The transposition
is drawn deterministically from `(inst.prompt, k, seed)`, uniformly over all `n(n-1)/2` pairs — which
includes the pair the step itself swaps, i.e. the "forgot to apply this swap" slip, and pairs the step
never touched, i.e. the "applied it to the wrong people" slip. Both are on-distribution mistakes for this
task. The rewrite is purely textual, so it works on `load_bbh()` items and the verbatim published
exemplar too; there is only one format here, so the `format`-knob clause of the amendment is vacuous.

Rendered example, `generate(5, 3, objects=3, scenario="ball")`. True trace:

```
(0) At the start: Alice: brown, Bob: white, Claire: blue.
(1) Alice and Claire swap balls: Alice: blue, Bob: white, Claire: brown.
(2) Bob and Claire swap balls: Alice: blue, Bob: brown, Claire: white.
(3) Alice and Bob swap balls: Alice: brown, Bob: blue, Claire: white.
...
```

`corrupt_step(inst, 3, 0)`:

```
(3) Alice and Bob swap balls: Alice: blue, Bob: brown, Claire: white.
```

→ `corrupted_state = "blue brown white"` against the true `"brown blue white"`. That particular draw
picked the (Alice, Bob) pair, so the line reads as if the swap it announces was never applied — the
canonical failure. `corrupt_step(inst, 1, 0)` instead picks (Bob, Claire) and yields
`(1) Alice and Claire swap balls: Alice: blue, Bob: brown, Claire: white.` — the swap was applied to the
wrong pair. Neither line is distinguishable from a correct one without carrying the previous state.

## Selftest

`python3 task.py --selftest` runs:

- 200 random (depth, seed, objects, scenario) instances: `solve() == answer`, `check(gold)` true,
  `check(wrong letter)` false, determinism, `len(steps) == len(states) == depth`, `step_spans()` slices.
- 7 tolerant-parser probes.
- **All 750 vendored BBH items**: parse → rebuild gold trace → `solve()` and our simulated answer must
  both equal BBH's own published `target`. 0 mismatches. This is the strongest available validation that
  the format and option convention are right.
- `exemplars(3, 0)` renders and `[0]` round-trips to `published_trace.txt` byte-for-byte.
- Knob validation rejects `objects=9`, `objects=2`, `scenario="x"`, `depth=0`, unknown knobs.
- **Redaction**: 20 instances x `k in {0, 1, depth//2, depth}` — `redact_prompt(inst, 0) == inst.prompt`
  and inserts nothing; `k > 0` differs from the prompt and carries exactly `k + 1` `[…]` placeholders;
  no initial item phrase survives anywhere outside the `Options:` block; the surviving swap clauses are
  exactly `k+1 … depth`, in order; the question and options block are byte-identical to the original; at
  `k = depth` no swap clause, connective or extra name remains; `k < 0` and `k > depth` raise. Also run
  over the published exemplar and one vendored BBH item per `n`.
- **Corruption**: 20 instances x `k in {1, depth//2, depth}` — `corrupted_state != states[k-1]`,
  `step_text != steps[k-1]`, the line still matches the published step shape (`(k) <action>: <name>:
  <object>, ...` with the action clause byte-identical to the true step), the reported assignment is a
  permutation of the true objects with no duplicates and the same person order, `corrupted_state` agrees
  with the line and has the same shape, exactly 2 of the `n` positions differ (minimal edit), determinism
  and purity in `(inst, k, seed)`, `k` outside `1..depth` raises, and different seeds give different
  corruptions. Also run over the published exemplar and one vendored BBH item per `n`.

```
generated instances: 200 cases, 0 failures
parser probes: 7
vendored BBH items: 750 checked, 0 mismatches
exemplars: 3 rendered, [0] verbatim round-trip OK
redaction: 20 instances x k in {0, 1, depth//2, depth}
corruption: 20 instances x k in {1, depth//2, depth}
SELFTEST PASSED
```

## Knobs

| knob | default | description |
|---|---|---|
| `objects` | `3` | number of people/objects n, 3–7. BBH publishes 3, 5, 7. Sets the state width: log2(n!) = 2.58 / 6.91 / 12.30 bits. Tokens per step grow with n (each step line lists every person), so tokens/step is **not** flat across this knob. |
| `scenario` | `"mixed"` | `ball`, `gift`, `book`, `dance`, `soccer`, or `mixed`. `mixed` samples one of the five published contexts per instance, which is what the BBH files themselves do. Pin it to hold surface form constant across a depth sweep. |

`depth` is not a knob; it is the first positional argument.

## Caveats

- **Contamination (high).** The vendored fixed sets are canary-tagged but have been on GitHub and
  mirrored on HF (`lukaemon/bbh`, `maveriq/bigbenchhard`, …) since 2022 and are in lm-eval-harness.
  Use `generate()` for actual eval; `load_bbh()` is there for validation and for a contamination
  A/B (same form, same depth, published vs. fresh), not as the primary eval set.
- **Small answer space.** n = 3 means a 33% random floor, and BIG-bench's own README warns a naive
  "person X still has object Y" model scores at or below chance. At depth ≥ 8 with n = 3 the answer
  is close to uniform, but accuracy is still bounded below by guessing. Raise `objects` to shrink the
  floor (1/5, 1/7) at the cost of longer step lines.
- **Published grammar bugs not replicated.** BBH's generator emits `a orange ball` (wrong article) and,
  in the white-elephant context, leaks `black ball` / `orange ball` / `pink ball` where presents were
  meant. Our generator writes `an orange ball` and `black present`. The vendored data is untouched, so
  `load_bbh()` still reproduces the bugs faithfully.
- **Deep instances read as unnatural.** A 24-swap prompt is one very long sentence of `Then, X and Y
  swap balls.` clauses. This is the published phrasing scaled up; it is grammatical but no human wrote
  a BBH item that long. If that surface artifact matters, cap the sweep at 8–12.
- **CoT step lines are long.** ~55 tokens per step at n = 3 (see `desk.json.tokens_per_step`) because
  the full state is restated every line. That is exactly the property that makes this a good stateful-CoT
  probe (the state is written down, at a fixed position, every step) but it makes deep sweeps expensive.
- **`depth` vs `objects` confound.** BBH conflates them (depth always == n). Anyone comparing to the
  published numbers should use `objects=n, depth=n`.

`desk.json` was reviewed against this implementation and found correct; it is unchanged. (Note that
`SOURCING.md` says "BBH depth = N-1 swaps" — that is wrong, it is N swaps; the count of *numbered CoT
lines* is N+1 because of the `(0) At the start:` line. `desk.json` reports `steps_in_trace: 4` for that
reason and is consistent with itself.)
