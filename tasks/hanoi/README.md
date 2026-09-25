# hanoi — Tower of Hanoi (Apple, "The Illusion of Thinking")

## Source and citation

Primary source: Shojaee, Mirzadeh, Alizadeh, Horton, Bengio, Farajtabar, **"The Illusion of Thinking:
Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity"**,
Apple, 2025, [arXiv:2506.06941](https://arxiv.org/abs/2506.06941). Apple released **no code or data**;
the only published exemplar for this task is the worked 3-disk solution embedded in the paper's own
appendix system prompt, recorded verbatim in `published_trace.txt`.

Secondary: `vendor/illusion_of_thinking/` — `taisazero/illusion_of_thinking`, commit
`fc1bc18128b914b54579397c36bb88cf281ef404`, **Apache-2.0** — a third-party reproduction whose
`SYSTEM_PROMPT` / `USER_PROMPT_TEMPLATE` constants reproduce the paper's appendix text byte-for-byte,
and whose `HanoiSimulator` encodes the legality rules. See `SOURCING.md` for the full candidate survey
and `desk.json` for the desk metrics. Critique worth knowing when reading results:
[arXiv:2506.09250](https://arxiv.org/abs/2506.09250) argues part of the reported "collapse" is
token-budget truncation and over-strict exact-match scoring of the `moves = [[...]]` format.

## Vendored vs written

* **Vendored:** the reproduction repo (reference only) and `published_trace.txt` (the verbatim Apple
  system prompt, user prompt template, and the 7-move worked example).
* **Written from scratch:** all of `task.py`. No vendored code is imported or copied; nothing in the
  vendored repo is a generator, a solver, or `Instance`-shaped (per `SOURCING.md`'s verdict, "nothing
  usable, must write"). The vendored legality rules were used only as a cross-check when writing
  `_legal_moves`.
* `task.py` reads `published_trace.txt` at import and exposes the Apple wording as
  `APPLE_SYSTEM_PROMPT`, `APPLE_USER_PROMPT_TEMPLATE` and `PUBLISHED_EXEMPLAR_MOVES`.

## The formulation: EXECUTION (default), not planning

**Default (`formulation="execute"`).** The prompt gives a random legal start configuration in the
paper's bracket notation and a list of `depth` legal moves in the paper's `[disk_id, from_peg, to_peg]`
notation; the question is *what configuration the pegs are in after those moves*. Same shape as
`tasks/blocksworld` (PlanBench plan-execution): the operator sequence is given, the model only has to
carry the state.

**Why this replaced the batch-1 formulation.** Batch 1 asked the model to *plan* — given a random start
and a single-peg goal, work out the first `depth` moves of the optimal solution and report where they
land. That failed on every model tested, and not by getting the simulation wrong: it failed by not
attempting the instance. At depth 2, DeepSeek opened with

> "We need to move the entire stack of 3 disks from peg 2 to peg 1"

and then recited the textbook 7-move solution from the standard start, ignoring the random start
configuration in front of it. The prompt is contaminated territory — the n-disks-from-a-full-first-peg
solution is in every textbook and all over GitHub — so a planning prompt retrieves the memorised
solution instead of engaging with the given state. A floor of 0 % produced by pattern completion
measures nothing about state tracking, which is the axis this library exists to compare. Execution
removes the retrieval affordance: there is no "the" solution to recall, only the given moves to apply.

The old formulation is still reachable as `generate(..., formulation="plan")`, unchanged in prompt
wording, trace format and sampling, so the two can be run head to head.

## Format decision

There is **no published trace for the execution formulation** — Apple only ever published a bare move
list for the planning formulation (`moves = [[1, 0, 2], [2, 0, 1], ...]`), which carries no per-step
state at all. Per AMENDMENT 3, where no published trace exists we choose the simplest format
consistent with the paper's own notation and record it here. The choice:

```
move 1: [2, 1, 2] -> [[3, 1], [4], [2]]
move 2: [1, 0, 1] -> [[3], [4, 1], [2]]
Answer: [[3], [4, 1], [2]]
```

One line per move: the step index (free, so the model never has to count), the move triple in the
paper's `[disk_id, from_peg, to_peg]` notation, `->`, and the resulting configuration in the paper's
bracket notation (`[[3, 2, 1], [], []]` is how the appendix system prompt writes a state). Both
notations are published; only the `move k:` prefix and the `->` join are ours. `len(steps) == depth`,
`states[i]` is the configuration after move `i+1`, and `answer == states[-1]`.

This is a deliberate departure from the planning format's *state-free* trace. In the planning
formulation the state is load-bearing but invisible (`desk.json`: `state_at_fixed_position: false`);
here the point is that the state is written down, so the CoT / no-CoT gap has something to measure.
Running `formulation="plan"` still emits the published state-free move list, so the "state written vs
state implicit" contrast is available inside this one task.

`format_cot()` for the plan formulation keeps its published wrapper (`moves = [` … `]`) followed by
the harness-imposed `Answer:` line; for the execute formulation the trace is just the move lines
followed by `Answer:`.

## Depth semantics

**depth = number of moves applied.** `generate(depth, seed)`:

1. `n = 4` disks (knob `disks`), independent of depth;
2. samples a start configuration uniformly over all `3**n` legal configurations (every assignment of
   disks to pegs is legal — within a peg the order is forced);
3. takes a uniform random legal walk of `depth` moves, never immediately reversing the previous move
   (knob `allow_undo`), and asks for the configuration reached.

`DEPTHS = [2, 4, 8, 16, 32, 64]`. Execution is far cheaper per step than planning was — one disk
changes peg, ~20 tokens of trace — so the ladder goes deeper than the batch-1 `[2, 5, 10, 20, 40, 80]`
while being much easier per step. Depth 2 is trivial for anything; a 9B-class instruct model should be
comfortable at 4–8, visibly degrading by 16–32 (where blocksworld's tracking accuracy falls apart),
and at floor by 64. The ladder is geometric so the curve has resolution wherever the collapse actually
lands. A depth-64 instance is ~64 prompt move lines plus a ~64-line trace: well inside context, so the
critique in arXiv:2506.09250 (that Apple's "collapse" is partly output truncation) does not apply here.

Decoupling n from depth is an improvement over batch 1, where n was a function of depth and so trace
length and state width were confounded. Here depth is the only thing that moves unless you move
`disks` yourself.

**Knobs** (`KNOBS`):

| knob | default | effect |
|---|---|---|
| `formulation` | `"execute"` | `"plan"` restores the batch-1 planning task (prompt, trace format and instance distribution all revert) |
| `disks` | `None` → 4 | number of disks; state width, independent of depth. Under `plan`/`optimal_prefix`, `None` → `max(3, smallest n with 2**n - 1 > depth)` |
| `optimal_prefix` | `False` | execute formulation, but the move list is the first `depth` moves of the optimal solution to a single-peg goal instead of a random walk — the bridge to the old formulation (same moves, but handed to the model) |
| `allow_undo` | `False` | permit a move that immediately reverses the previous one (trivially trackable, hence off) |
| `goal_peg` | `None` | `plan` / `optimal_prefix` only: fix the goal peg |

Under `optimal_prefix=True` (and under `plan`) the start is sampled uniformly among configurations
whose optimal distance to the goal is **strictly greater than `depth`**, so the answer is never just
the goal configuration printed in the prompt. Uniformity comes from the bitwise decomposition of the
distance: bit *i−1* of D is 1 exactly when disk *i* is not on the running target peg, and each 1-bit
leaves two peg choices, so there are `2**popcount(D)` configurations at distance D; sampling D with
that weight and filling in the free choices is uniform over the eligible set.

## ANSWER_FORMAT

```
ANSWER_FORMAT = "a peg configuration written like [[3, 2], [1], []] — three lists (peg 0, peg 1,
                 peg 2), each listing that peg's disks from bottom to top"
```

The paper's own answer is the *move list*, which is useless for these metrics: under the execution
formulation the move list is given in the prompt, and under the planning formulation the answer and
the chain of thought would be the same tokens (so acc(no-CoT) is undefined — you cannot withhold the
scratchpad and then ask for it). Scoring the **resulting configuration** gives a short, exact-match
answer that a no-CoT model must reach by simulating internally. The notation is published
(`[[3, 2, 1], [], []]` in the appendix system prompt), so nothing is invented.

Exact-match: within a peg the disk order is forced by the rules, so `check()` canonicalises each peg
to descending (bottom-to-top) order before comparing; a model that lists a peg top-to-bottom is not
penalised. `check()` takes the last `Answer:` line (else the last non-empty line), accepts
bold/markdown decoration around it, and falls back to a `Peg 0: … Peg 1: … Peg 2: …` prose parse if no
bracket structure is present. Everything else is strict.

## Redaction (AMENDMENT 4)

`REDACTION_MEANINGFUL = True`. `redact_prompt(inst, k)` removes exactly what would let a model
recompute the state after step *k*: the **initial configuration** (one `[…]`) and the **first k move
lines** (one `[…]` each) — `k + 1` placeholders in all. Kept: the disk count and notation sentence,
the three rules (static, not state), moves *k+1..depth*, and the question — everything needed to
continue from step *k+1* given the trace so far. `redact_prompt(inst, 0)` returns `inst.prompt`
unchanged; at `k == depth` no configuration and no move triple survives anywhere in the prompt.

Rendered example, depth 4, `k = depth//2 = 2` (from `generate(4, 0)`):

```
I have a puzzle with 4 disks of different sizes on three pegs. The disks are numbered from 1
(smallest) to 4 (largest). The positions are 0-indexed (the leftmost peg is 0). A configuration is
written as three lists — the disks on peg 0, then peg 1, then peg 2 — each list running from the
bottom of that peg to the top.

Initial configuration: […]

Rules:

• Only one disk can be moved at a time.

• Only the top disk from any stack can be moved.

• A larger disk may not be placed on top of a smaller disk.

I made the following moves, in order, each written as [disk_id, from_peg, to_peg]:

moves = [
  […],
  […],
  [1, 1, 0],
  [1, 0, 2],
]

Report the configuration of the three pegs after these moves.
```

(The unredacted prompt has `Initial configuration: [[4, 2], [], [3, 1]]` and `[1, 2, 1]`, `[2, 0, 2]`
in the first two move slots. The trace for steps 1–2 ends at `[[4], [1], [3, 2]]`, which is all the
model needs to finish — but only if it read it out of the trace.)

For `formulation="plan"` the model is given no moves at all, so the only redactable material is the
initial configuration; it is replaced by a single `[…]` for any `k > 0` (goal, rules and question
stay). The redaction is coarse there and does not vary with *k* — one more reason `plan` is not the
default.

## Corruption (AMENDMENT 6)

`corrupt_step(inst, k, seed) -> (step_text, corrupted_state)` rewrites step *k* (1-based) so that the
state it reports is plausibly wrong, and returns that state in the same bracket form as
`inst.states`. In the default **execute** formulation the move triple is left exactly as written —
only the configuration after the arrow changes — and the wrong configuration is one of two, chosen
deterministically from `(inst, k, seed)`:

* **wrong peg** — the moved disk is put on the *third* peg instead of `to_peg`. Offered only when
  that peg is empty or carries a larger disk, so the reported configuration is one that some legal
  move out of the previous configuration could have produced.
* **move not applied** — the configuration from before move *k* is reported again (the dropped
  update). Always available, always legal, and the only option when the third peg is blocked.

Both are legal configurations of the same *n* disks in canonical order, so nothing in the line looks
malformed; the error is only visible by re-applying the move. From `generate(4, 0)` — gold trace
`move 2: [2, 0, 2] -> [[4], [1], [3, 2]]`, `move 3: [1, 1, 0] -> [[4, 1], [], [3, 2]]`:

```
corrupt_step(inst, 2, 0) -> ('move 2: [2, 0, 2] -> [[4, 2], [1], [3]]', '[[4, 2], [1], [3]]')
                             move not applied: disk 2 is still on peg 0 (peg 1 holds disk 1, so the
                             wrong-peg variant is not offered here)

corrupt_step(inst, 3, 0) -> ('move 3: [1, 1, 0] -> [[4], [], [3, 2, 1]]', '[[4], [], [3, 2, 1]]')
                             wrong peg: disk 1 went to peg 2 instead of peg 0
```

**`formulation="plan"` is the documented exception.** Its step lines are bare `[disk, from, to]`
triples inside `moves = [ ... ]` and report no state at all, so there is nothing after an arrow to
corrupt. As the amendment's "apply to every format the task supports" requires a corruption here
too, the **destination peg of the triple** is corrupted instead — to the third peg, which is the
only other choice — and `corrupted_state` is the configuration that altered move reaches. From
`generate(8, 3, formulation="plan")`, whose fourth gold move line is `  [2, 2, 0],` (state
`[[3, 2], [4, 1], []]`):

```
corrupt_step(inst, 4, 0) -> ('  [2, 2, 1],', '[[3], [4, 2, 1], []]')
```

Two consequences of that exception, stated plainly: the action text *does* change (it cannot not),
and the corrupted move may violate the size rule, because the third peg is precisely the one the
optimal move avoids and it often holds a smaller disk. The *configuration* is still well formed —
in Hanoi every assignment of disks to pegs is a legal state, since stacking order is forced — so
`corrupted_state` never shows a larger disk resting on a smaller one. A harness that needs the
corrupted action to be legal as well should use the execute formulation, which is the default.

## Interface notes

* `solve()` is written independently of `generate()`. For the execute formulation it re-parses the
  start configuration and the move list **out of the prompt text** and applies them with an explicit
  stack simulation (push/pop with precondition checks), where `generate()` works from an internal
  disk→peg array it never re-reads — so a prompt-rendering bug cannot hide behind a matching
  generator bug. For the plan formulation `solve()` keeps the batch-1 route: no recursion, just the
  closed-form distance-to-goal of every legal neighbour, taking the unique distance-decreasing move.
  The selftest adds a *third* implementation (full BFS over the 3^n state graph) as a cross-check.
* `exemplars(k, seed)[0]` is the paper's published worked example — 3 disks, standard start, the seven
  move triples parsed out of `published_trace.txt` rather than recomputed — re-cast as an *execution*
  instance (moves given, resulting configuration asked for). The remaining exemplars are generated at
  depth 3–6.
* `step_spans(inst)` is provided (not required by AMENDMENT 3) for per-step attention-blinding runs.
* `python3 task.py --selftest` covers: 200 execute instances (solve==answer, check(gold),
  check(wrong)==False, determinism, `len(steps)==len(states)==depth`, move legality re-simulated with
  stacks, no immediate undo, every move and the start round-tripping through the prompt text); 60
  instances each of `optimal_prefix`, `plan` (BFS cross-check, prefix optimality, answer ≠ goal) and
  `disks`/`allow_undo`; the AMENDMENT 4 redaction checks over 21 instances at
  `k ∈ {0, 1, depth//2, depth}`; the AMENDMENT 6 corruption checks over 26 instances (20 execute,
  5 plan, plus the published exemplar) at `k ∈ {1, depth//2, depth}` — corrupted state ≠ true state,
  corrupted line ≠ gold line, step-line structure and (in `execute`) the move triple preserved,
  the state a well-formed configuration of the same disks and one of the two documented
  corruptions, determinism in `(inst, k, seed)`; `exemplars(3, 0)`; and tolerant-parse spot checks.

## Caveats

1. **Deviation from the paper, stated plainly:** Apple score the complete move list generated from the
   standard start. We give the moves and score the configuration they reach, from a random start. The
   *notations* are the paper's; the *task*, the *scored quantity* and the *instance distribution* are
   not. Numbers from this task are not comparable to the paper's accuracy-vs-N curves. (The `plan`
   knob is closer to the paper but still scores a configuration, not a move list.)
2. **A closed-form shortcut exists for `optimal_prefix`/`plan`** — the configuration after k optimal
   moves is computable in O(n) from the bits of the remaining distance — so a model that knows the
   trick could skip the serial simulation. It does **not** apply to the default random walk, which has
   no structure to exploit: every move must be applied. A further argument for the execute default
   (`desk.json` lists the published task as `solvable/TC0`).
3. **Answer space** is `3**n` = 81 configurations at the default `disks=4`, but only configurations
   reachable in `depth` moves from the start are plausible answers, and shallow instances concentrate
   near the start. Chance accuracy is small but not zero at depth 2; raise `disks` if that matters for
   a given sweep.
4. **Contamination is much lower than for the published task** (`desk.json` rates it *high*): a random
   start plus a random legal walk appears nowhere in the training corpus, and there is no "the
   solution" to recall. The residual risk is that Hanoi *notation* is familiar, which is a feature (no
   format-learning burden) rather than a leak.
5. **Error diagnosis is partial.** A wrong final configuration usually identifies which disk was
   mistracked, but not which move it was dropped at; `states` gives the per-step ground truth for a
   harness that wants to score prefixes.
6. Token cost is ~9–10 tokens per move triple for all three reference tokenizers (`desk.json`) plus
   the configuration string; an execute trace line is roughly 20–25 tokens at n = 4 and grows with n,
   not with depth. Total trace tokens grow linearly in depth.
7. **`disks=1` or `2` with `allow_undo=False`** leaves very few legal non-undo moves (at n = 1 the walk
   is forced), so those settings are near-deterministic. The selftest exercises them; they are not
   sensible sweep points.

`desk.json` describes the task **as published** (AMENDMENT 2) and therefore still describes the
planning formulation; a one-line NOTE has been appended to its `notes` field recording that batch 2's
default differs. `examples.txt` is the verbatim output of `python3 task.py --demo` (3 instances at
depth 2, 3 at depth 64).
