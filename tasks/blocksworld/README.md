# blocksworld — Blocks World state tracking

Given an initial Blocks World configuration and a fully specified sequence of actions, report
the state that results from executing them. This is the **state-tracking** (plan-execution)
formulation, not planning: the model is handed the actions and only has to simulate them.

## Source and citation

**Primary source (CoT format).** Stechly, Valmeekam & Kambhampati, *"Chain of Thoughtlessness?
An Analysis of CoT in Planning"*, NeurIPS 2024, [arXiv:2405.04776](https://arxiv.org/abs/2405.04776).
Companion repo `karthikv792/cot-planning`, the `blocksworld_state_tracking` domain; the step
format is taken from `example_query_prompts/blocksworld_state_tracking/upb.txt` (the
"Blocksworld Universal Algorithm" n-shot CoT prompt), copied verbatim to `published_trace.txt`.

**Secondary source (problem-statement wording).** Valmeekam, Marquez, Olmo, Sreedharan &
Kambhampati, *"PlanBench"*, NeurIPS 2023 D&B, [arXiv:2206.10498](https://arxiv.org/abs/2206.10498).
Repo `karthikv792/LLMs-Planning`, `plan-bench/prompts/blocksworld/task_7_plan_execution.json` —
PlanBench's Task 7 (plan execution) is the published *tracking* task, and supplies the
`[STATEMENT] / [ACTION SEQUENCE] / [RESULTING STATE]` scaffold and the domain intro.

Action semantics are the standard 4-operator IPC Blocks World PDDL domain
(`pick-up` / `put-down` / `stack` / `unstack` over `clear` / `on-table` / `arm-empty` / `on`),
identical in both vendored repos.

### Licenses

- `vendor/LLMs-Planning/` — MIT (LICENSE present, © Valmeekam Karthik 2024).
- `vendor/cot-planning/` — **no license file**; treat as all-rights-reserved. Nothing from it is
  redistributed in `task.py` beyond (a) the domain intro prose, which is common to the
  MIT-licensed PlanBench repo, and (b) the six sentence templates of the step format, which are
  the format itself. `published_trace.txt` is a verbatim quotation for citation purposes.

## What was vendored vs written

**Vendored, used as a specification:** the PDDL domain, the `upb.txt` CoT exemplar, the
plan-execution prompt scaffold, and the domain intro.

**Written from scratch** (neither repo ships a stdlib-only seeded generator, an `Instance`
shape, or a per-step tracking trace — see `SOURCING.md`): the instance generator, two
independent world models, the renderers, the tolerant checker.

The two world models are the differential check required by the spec:

- `generate()` uses `_World`, a **stack-list** model (list of bottom-to-top lists plus a held
  block) and renders the trace from it.
- `solve()` ignores all of that, **re-parses `Instance.prompt` back into a PDDL predicate set**
  (`clear`/`ontable`/`on`/`handempty`/`holding`), applies textbook add/delete effects with
  precondition checking, and renders the answer from the predicate set. It shares only the
  sentence templates with the generator.

A third check runs in `--selftest`: the generator's renderer must reproduce the **published
trace verbatim**. `published_exemplar()` replays example 1 of `published_trace.txt` and the
selftest asserts all 6 step blocks are byte-identical to the file. (Example 2, 12 steps over 4
blocks, was verified the same way during development — all 18 published step blocks reproduce
exactly.) This pins the format to the paper rather than to my reading of it.

## Format decision

The gold trace reproduces the published `upb.txt` step block exactly — four lines per step,
three-space continuation indent, prose predicates:

```
1. Current State: Block B is clear, Block C is clear, the hand is empty, Block C is on top of Block A, Block A is on the table and Block B is on the table
   Action: unstack the Block C from on top of the Block A
   Reason: The above action is applicable in the current state because its preconditions; Block C is clear, the hand is empty and Block C is on top of Block A, are satisfied in the current state.
   Resulting State: Block A is clear, Block B is clear, the hand is currently holding Block C, Block A is on the table and Block B is on the table
```

Steps are separated by a blank line and `format_cot()` closes with the published closing
(`Final State: …` then `[PLAN END]`) followed by the harness-mandated `Answer: <answer>` line.
Predicate order inside a state is the published one: `clear` (alphabetical), then the hand,
then `on` (alphabetical by upper block), then `on the table` (alphabetical); items joined
`a, b, c and d`.

Three deviations from `upb.txt`, all forced by the task being tracking rather than planning:

1. **No goal.** `upb.txt` is a planning prompt, so its statement carries `My goal is to have
   that …` and its trace ends `The goal conditions are satisfied in the final state. Hence, the
   above plan is valid.` There is no goal in a tracking instance, so the goal line is dropped
   from the statement and that closing sentence from the trace. `Final State:` and `[PLAN END]`
   are kept verbatim.
2. **The statement is PlanBench Task 7's,** not `upb.txt`'s — `[ACTION SEQUENCE] … [ACTION
   SEQUENCE END] / [RESULTING STATE]` — because that is the published wording for *given*
   actions. Its stray leading space before `I have executed the following action sequence:` is
   reproduced verbatim, as is the dangling `[RESULTING STATE]` slot marker that ends the prompt.
3. **Block naming is `upb.txt`'s** (`Block A`, `Block B`, …), not PlanBench's colour names
   (`the red block`). The CoT format is the primary source, the naming follows it, and letters
   scale past the 26-colour vocabulary.

The `Reason:` line is kept even though it is pure derivable filler, because it is in the
published format and because it is exactly the kind of line the review is meant to probe
(is it load-bearing, or is it the "thoughtless" part of the chain?).

**Rules in the prompt.** AMENDMENT 3 says `prompt` is the problem statement only. The published
domain intro (the eleven action-restriction lines) is included at the top of `prompt` anyway,
under the `include_rules` knob (default `True`): it is domain background without which the
action semantics are underspecified, it is not a scratchpad instruction, and the harness does
not supply it. Set `include_rules=False` to strip it — useful if the harness prepends it once
rather than per exemplar.

## Depth semantics

**`depth` = the number of actions in the given plan** = `len(steps)` = `len(states)`. Each
action is one state update, so the published numbered blocks map one-to-one onto serial steps.

Instances are built by a random legal walk from a random initial configuration. The hand starts
empty; because every action is either an acquire (`pick up`/`unstack`) or a release (`put
down`/`stack`), the hand strictly alternates, so the final state has the hand empty at even
depth and holding a block at odd depth. By default (`allow_undo=False`) the walk will not
immediately reverse the previous action, which would make a step trivially recoverable from
the one before it.

`states[i]` is a compact canonical string, stacks bottom-to-top sorted, e.g. `AC|B|D/-`
(three stacks, hand empty) or `ACB|D/A` — for harness bookkeeping, not shown to the model.

### DEPTHS = [2, 4, 6, 10, 16, 24]

- **2** — one block moved. Near-ceiling; the floor of the sweep.
- **4, 6** — two to three block moves; 6 is the length of the paper's own first worked example.
- **10** — just under the paper's second worked example (12 steps). Expected to be where a 7B
  starts losing the thread without CoT.
- **16, 24** — beyond anything in the published prompt. ~160 tokens/step (measured in
  `desk.json`) puts depth 24 at ~3.9k CoT tokens, which is comfortable for a 32k-context 7B
  while being well past where the state has to be re-derived from the scratchpad rather than
  held.

Depths are not restricted to even numbers; odd depths simply end with a block in hand.

## ANSWER_FORMAT rationale

```
the resulting state, written as a list of conditions in the domain's wording, e.g.
`Block B is clear, the hand is empty, Block A is on top of Block C and Block C is on the table`
```

The answer is the **full resulting state**, which is the published answer slot
(`[RESULTING STATE]`, and PlanBench's own `ground_truth_plan` field is exactly such a predicate
list). The alternatives were rejected: "is the goal satisfied?" is binary (a coin-flip
baseline), and "what is Block D on top of?" has an answer space of only *n+1* and reveals
nothing about which step was dropped. A full state gives an answer space equal to the number of
Blocks World configurations — 125 reachable states at 4 blocks, 7057 at 6 — and a wrong answer
usually localises the divergence.

**How it is made exact-match checkable.** `check()` normalises both sides to the *core
configuration* — the set of `on(x, y)` pairs, the set of on-table blocks, and the held block —
and compares those for exact equality. So:

- clause order does not matter;
- `clear` and `hand is empty` are **ignored**, because they are derivable from the core
  configuration; a model is neither rewarded for listing them nor punished for omitting them;
- `Block A` / `A`, `the hand is currently holding` / `holding` / `I am holding`,
  `**Answer:**` / `answer:`, trailing periods, and `,` / `;` / newline separators all normalise
  away;
- everything about the actual configuration is strict — one misplaced or missing block fails.

Answer extraction follows the harness rule: last `Answer:` line (case-insensitive; if that line
is empty, the remainder of the completion), else the last non-empty line. An empty or
unparseable answer is `False`, never a vacuous match. A fixed accept/reject battery for all of
this runs in `--selftest`.

## Knobs

| knob | default | meaning |
| --- | --- | --- |
| `blocks` | `4` | number of blocks, 2–26. The paper's own knob (swept 3–20 in its Figure 2). 4 matches the trace's second worked example. Raising it widens the state space *and* lengthens every state line, so it trades against depth for token budget. |
| `include_rules` | `True` | prepend the published domain intro to the prompt. |
| `allow_undo` | `False` | let the walk immediately reverse the previous action. |

`depth` is not a knob.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4) returns the prompt with everything needed to *recompute*
the state after step `k` removed, and everything needed to *continue* from step `k+1` left in
place. For blocksworld that means the initial-conditions sentence (one `[…]` for the whole
span) and the first `k` lines of the `[ACTION SEQUENCE]` block (one `[…]` each). The domain
intro is a static rule table, not state, so it stays; so do actions `k+1..depth`, the
`[RESULTING STATE]` question and all the `[STATEMENT]` scaffolding. `redact_prompt(inst, 0)`
returns `inst.prompt` unchanged; `redact_prompt(inst, inst.depth)` leaves no `Block <letter>`
token anywhere in the prompt — the model can only continue from the trace it was handed.
`REDACTION_MEANINGFUL = True`.

Example, `generate(depth=6, seed=1000)` at `k = depth // 2 = 3` (domain intro elided):

```
[STATEMENT]
As initial conditions I have that, […].
 I have executed the following action sequence:

[ACTION SEQUENCE]
[…]
[…]
[…]
stack the Block B on top of the Block A
pick up the Block C
stack the Block C on top of the Block D
[ACTION SEQUENCE END]
[RESULTING STATE]
```

The unredacted prompt opens `As initial conditions I have that, Block A is clear, Block B is
clear, Block D is clear, the hand is empty, Block A is on top of Block C, Block B is on the
table, Block C is on the table and Block D is on the table.` and lists all six actions, the
first three being `unstack the Block A from on top of the Block C`, `put down the Block A`,
`pick up the Block B`.

Note that the action lines are unnumbered (see Caveats), so redaction shortens the visible list:
a model continuing from step `k+1` must take the placeholder count as its step offset. That is
inherent to the published `[ACTION SEQUENCE]` format — the alternative, dropping the lines
entirely, would hide the offset as well.

## Caveats

- **Action lines in the prompt are unnumbered**, per PlanBench's published `[ACTION SEQUENCE]`
  block, while the CoT steps are numbered `1.`, `2.`, … . The original SPEC wanted the step
  index given for free; AMENDMENT 3 withdrew that in favour of published fidelity, so at large
  depth the model also has to keep its place in an unnumbered list. If the harness wants that
  confound removed, numbering the action lines is a one-line change in `_build_prompt`.
- **The published accuracy numbers in `desk.json` are for the planning variant** of
  `blocksworld_state_tracking` (the model produces the actions), not for this tracking task.
  Tracking is strictly easier — there is no search — so those figures are a floor, not a
  prediction. No published tracking-only accuracy at varying depth was found.
- **Contamination risk is medium.** Blocks World is one of the most-discussed LLM benchmarks
  and the `upb.txt` prompt is public. Instances here are freshly generated, but the format and
  domain are certainly in pretraining data. (PlanBench's own "Mystery Blocksworld"
  obfuscation is the standard mitigation; it is a separate domain and was not implemented.)
- **Shallow instances have limited diversity.** At `blocks=4, depth=2` there are only ~48
  distinct action traces, so depth-2 accuracy is a ceiling measurement, not a fine-grained one.
  Raise `blocks` if more shallow variety is needed.
- `answer` is a sentence (~100–200 chars), not a token. That is the published answer slot;
  the normalisation in `check()` is what makes it exact-match comparable.
- The `Reason:` line inflates tokens per step roughly 2× versus a bare
  Current/Action/Resulting trace. It is published, so it stays; a knob to drop it was
  deliberately *not* added, since that would be a format the paper never ran.
- **`desk.json` was corrected in one field.** `state_bits` was 6.0, derived from OEIS A000522
  (⌊e·n!⌋ = 65 at n=4). That is the wrong sequence: a Blocks World configuration is an
  unordered *set* of ordered stacks, i.e. a "set of lists", OEIS A000262 (1, 1, 3, 13, 73, …),
  giving 73 hand-empty configurations at 4 blocks, plus 4 × A000262(3) = 52 states with a block
  in hand = **125 reachable states, log2 ≈ 6.97**. Confirmed by exhaustive BFS over the
  4-operator domain. `state_bits` is now 6.97 and `notes` records the correction; every other
  field is unchanged.

## Files

- `task.py` — the interface. `python3 task.py --selftest`, `python3 task.py --demo`.
- `published_trace.txt` — verbatim `upb.txt` (2 worked examples, 18 step blocks).
- `examples.txt` — `--demo` output: 3 instances at depth 2 and 3 at depth 24.
- `SOURCING.md` — phase-1 candidate survey.
- `desk.json` — desk metrics for the published format.
- `vendor/` — `LLMs-Planning` (PlanBench) and `cot-planning`, commits in `*.commit`.
