# s5_composition — word problem over S5 (permutation composition)

Compose a sequence of permutations of {1..5} and report the resulting arrangement. This is the
canonical non-solvable-group word problem: by Barrington (1989) it is NC1-complete under
projections, so (unlike its solvable siblings S3, Z_n) it has no known constant-depth shortcut and
a transformer must thread the state serially — exactly the property this review is testing.

## Source and citations

**Primary source (task):** Liu, Ash, Goel, Krishnamurthy & Zhang, *Transformers Learn Shortcuts to
Automata*, arXiv:2210.10749 (ICLR 2023). Studies transformers on automaton/group word problems
including S5. **No code release** (the linked GitHub repo, vendored at `vendor/shortcut_automata/`,
is the project website only) and **no textual trace**: their models consume the automaton as
categorical input for final-state classification.

**Format source (the published trace):** *(How) Do Language Models Track State?*, arXiv:2503.02854,
code at https://github.com/belindal/state-tracking, commit
`fc63e2db262265f42e9d2ac5c05888284f843b4c`, **MIT License**, vendored at `vendor/state-tracking/`.
`permutation_task.py` is a real S3/S5 generator that trains GPT-2/Pythia on this task **as text**;
its notation is what `published_trace.txt` and this task.py reproduce.

**Theory:** Barrington 1989 (NC1-completeness of non-solvable group word problems); Barrington &
Thérien 1988 (solvable groups fall into weaker classes); Merrill & Sabharwal, arXiv:2310.07923
(expressive power of transformers with CoT).

**Related-work only:** `vendor/neural_networks_chomsky_hierarchy/` (Delétang et al.,
arXiv:2207.02098, Apache-2.0) — its `cycle_navigation` task is the Z_5 (solvable) sibling; cited,
not used.

## Vendored vs written

- **Vendored:** the three repos above. Nothing is imported at runtime — `task.py` is pure stdlib.
- **Reused from the vendored code (transcribed, not imported):** the composition rule from
  `PermutationState.apply_action`, `new_perm[j] = state[action[j]-1]`, and the digit-string
  notation for both actions and states.
- **Written here:** everything else — `generate`, the independent `solve`, `check`, `format_cot`,
  `exemplars`, group knob (S_n / A_n / Z_n), selftest, demo. The pre-amendment draft `task.py`
  (invented named-generator table `P1..P4`, letters instead of digits, scratchpad instructions
  inside the prompt, no `exemplars`/`DEPTHS`/`ANSWER_FORMAT`) was discarded, not extended.

**Validation that the transcription is faithful:** `exemplars(k, seed)[0]` rebuilds the published
8-step story from `published_trace.txt` by running *our* composition forward, and asserts the
resulting state sequence equals the published `state_seq` digit-for-digit (`53124 15342 21543 35412
43215 43512 35124 12453`). The selftest also re-greps `published_trace.txt` to confirm the embedded
constants still match the file.

## Format decision

The published format is reproduced exactly for the action vocabulary, the notation, and the
problem statement; one documented addition is required for the gold trace.

- **Actions / prompt.** Published: a "story" is a space-separated sequence of actions, each an
  n-digit one-line-notation permutation (`53124 31254 51243 …`), starting from the identity
  `12345`. `Instance.prompt` is that story plus the minimum framing needed to pose it as a
  question — start state, "apply these in order", the composition rule, "What is the final
  arrangement?". No scratchpad instruction, no exemplars, no answer-format line (the harness adds
  those). The composition rule *is* spelled out: the published setup teaches the convention by
  fine-tuning, not by instruction, so without it the problem is genuinely ambiguous (left- vs
  right-action) for a zero/few-shot model. This is the only semantic content added to the
  published problem statement.
- **States.** *The published trace does not expose intermediate states.* The model reads only the
  action sequence; `state_seq` is recorded by the generator into a separate JSON field for
  supervision/probing (`desk.json.state_at_fixed_position = false` records this). A gold CoT that
  merely echoed the actions would carry no state, which would void the CoT/no-CoT comparison, so
  per AMENDMENT 3 each serial step is rendered as the published (action, state) **pair** that the
  generator itself records, in the published digit notation, with the `step k:` labelling used in
  `published_trace.txt`'s own state block:

      step 1: 53124 -> 53124
      step 2: 31254 -> 15342
      ...
      Answer: 12453

  Both sides of the arrow are published tokens; the pairing and the `step k:` label are ours.
  Cost: ~13 tokens/step versus the ~6 of the state-free published story (`desk.json`
  `tokens_per_step` describes the latter and remains correct as a description of the published
  format — it is not an error and was left unchanged).
- **Closing.** The published format has no closing sentence, so `format_cot` is the steps followed
  directly by the harness-imposed `Answer: <perm>` line.

## Ergonomic variant (AMENDMENT 5)

s5_composition is a `†` task: belindal/state-tracking taught its notation by **fine-tuning**, and
batch 1 confirmed the format does not transfer by prompting at any model size available. The
`format` knob adds an SFT-free alternative; `format="published"` remains the default so the
"as published" rows stay reproducible.

    format="ergonomic"

**What changes:** only the *rendering* of `prompt` and `steps`.

- **Prompt.** Plain language, and the convention is stated in one sentence — "a permutation is a
  list of positions" — followed by a one-line worked example in exactly the shape of the step
  lines. The example is generated for the instance's `n` and deliberately uses a non-identity
  arrangement, so it teaches the lookup rather than restating it.
- **Step.** The permutation, the position lookups, and the resulting arrangement, written out in
  full, with the arrangement first spaced digit-by-digit (the thing actually being read off) and
  then joined:

      step 3: apply 51243 to 21543 -> take positions 5,1,2,4,3 of 21543 -> 3 2 1 4 5 -> 32145

  This is the style DeepSeek-V4-Flash invented for itself when it succeeded at the published
  prompt. It is redundant on purpose: the operand state is restated before the lookup, so a model
  never has to carry the arrangement across the newline in activations.

**What does not change:** the instance distribution (the word drawn is a function of
`depth, seed, group, n` only — `format` is not fed to the RNG), `answer`, `states`, `depth`
semantics, `solve()` and `check()`. `--selftest` asserts all of this per instance, in both
directions (`check(published_inst, format_cot(ergonomic_inst))` and vice versa).

`exemplars(k, seed, **knobs)` forwards `format`, so the harness's few-shot block is written in the
same format as the instance being solved; `meta["format"]` records it. Under `"ergonomic"`,
`exemplars(k, seed, format="ergonomic")[0]` is still the published *story* (same actions, same
states, cross-checked against `published_trace.txt`), re-rendered in the ergonomic trace format.
`redact_prompt` is format-agnostic: like `solve()`, it re-reads the prompt text through one shared
parser that recognises both openings.

One rendered instance (`depth=6, seed=7`; the published-format rendering of the *same* instance is
beside it in `examples.txt`, and both answer `31524`):

    Start with the arrangement 12345.
    Apply these 6 permutations to it, one at a time, in the order given:
    31245 34512 14235 25341 42153 32145
    Each permutation is a list of positions: to apply one, read off the digits of the current
    arrangement at those positions, in that order, and that is the new arrangement. For example,
    applying 21345 to 54321 means taking positions 2,1,3,4,5 of 54321, which is 4 5 3 2 1, so the
    new arrangement is 45321.
    What is the final arrangement?

    step 1: apply 31245 to 12345 -> take positions 3,1,2,4,5 of 12345 -> 3 1 2 4 5 -> 31245
    step 2: apply 34512 to 31245 -> take positions 3,4,5,1,2 of 31245 -> 2 4 5 3 1 -> 24531
    step 3: apply 14235 to 24531 -> take positions 1,4,2,3,5 of 24531 -> 2 3 4 5 1 -> 23451
    step 4: apply 25341 to 23451 -> take positions 2,5,3,4,1 of 23451 -> 3 1 4 5 2 -> 31452
    step 5: apply 42153 to 31452 -> take positions 4,2,1,5,3 of 31452 -> 5 1 3 2 4 -> 51324
    step 6: apply 32145 to 51324 -> take positions 3,2,1,4,5 of 51324 -> 3 1 5 2 4 -> 31524
    Answer: 31524

**Token cost.** A step line is 88 chars ≈ 22 tokens at n=5, against 23 chars ≈ 6 tokens for the
published `step k: 53124 -> 15342` (and ~6 for the state-free published story in `desk.json`). At
DEPTHS[-1] = 32 that is ≈ 700 CoT tokens versus ≈ 190 — still far from any context limit, but the
CoT-token budget is ~3.7× the published format's and should be reported alongside accuracy.

## Depth semantics

`depth` = **number of permutations composed** = number of serial state updates =
`len(steps)` = `len(states)`. It is the repo's `--story-length`. Nothing else scales with depth:
the state is a fixed 5 digits and tokens/step are flat, so depth is a clean serial-compute knob.

**DEPTHS = [2, 4, 8, 12, 16, 24, 32].**
- 2 and 4 are near-trivial *with* CoT and sit at the plausible no-CoT ceiling — a 7B can hold one
  or two compositions in-activation, so this is where the CoT gap should open.
- 8 matches the published example trace length.
- 12–16 is where per-step error compounding should start eating CoT accuracy (a 1% per-step slip
  compounds to ~15% instance failure by 16).
- 24 and 32 should be at or past collapse for Qwen2.5-7B-Instruct even with CoT, giving the sweep
  a floor. 32 steps × ~13 tokens ≈ 420 CoT tokens, so context is never the binding constraint.

Chance is 1/120 = 0.8% at every depth (the composition of uniform S5 actions is uniform), so
accuracy is directly interpretable and does not drift across the sweep.

## ANSWER_FORMAT

    a permutation of the digits 1-5 written as a single 5-digit string, e.g. 53124

This is the published state notation verbatim, which makes CoT and no-CoT conditions answer in the
same alphabet, and it keeps the answer space at exactly 120 (`error_diagnostic` holds: a wrong
answer is a specific group element, and the step at which the model diverged is usually
recoverable by comparing its trace to the gold prefix). `check()` takes the last `Answer:` line
(falling back to the last non-empty line), keeps digits only, and exact-matches — so `1 2 4 5 3`,
`Answer: 12453.` and `the answer is 12453` all pass.

Non-default `n`: `ANSWER_FORMAT` is a module constant fixed to the default S5 wording; each
instance also carries `meta["answer_format"]` with the n-appropriate sentence, which a harness
should prefer when sweeping the group knob.

## Knobs

| knob | default | values | note |
|---|---|---|---|
| `group` | `"S5"` | `S5`, `S3`, `A5`, `S<n>`/`A<n>` (n≤9), `Zn` | action vocabulary |
| `n` | `None` | 2..9 | only meaningful for `Zn`; `S5`/`A5`/`S3` fix their own n |
| `format` | `"published"` | `published`, `ergonomic` | rendering only — see "Ergonomic variant" |

`S5` (default, non-solvable, NC1-complete, |G| = 120) is the experiment. `A5` is the other
non-solvable option (|G| = 60). `S3` (|G| = 6) and `Zn` (cyclic rotations, abelian, |G| = n) are
the solvable/TC0 contrast cases for a later ablation — Barrington & Thérien 1988 put them in
strictly weaker classes, and `Zn` is the same construction as DeepMind's `cycle_navigation`.
Depth is not a knob (it is the first positional argument).

## Redaction

`REDACTION_MEANINGFUL = True`. `redact_prompt(inst, k)` removes exactly what a model would need to
*recompute* the state after step k: the **start arrangement** (the initial state) and the **first k
permutations** of the story, each replaced by one literal `[…]`. What survives is everything needed
to *continue*: permutations k+1..depth in place, the composition rule (a static rule, not state) and
the question. `redact_prompt(inst, 0)` is the prompt verbatim; `redact_prompt(inst, inst.depth)`
leaves no digit string from which any state could be reconstructed. The permutation count in the
framing sentence is left alone — it describes how many permutations there are, not what they are —
so the story stays positionally aligned with the trace the harness supplies. Like `solve()`,
`redact_prompt` re-reads `inst.prompt` rather than `inst.meta`, through the one parser both formats
share, so it behaves identically under `format="ergonomic"` (`Start with the arrangement […].`).

The published exemplar (depth 8) at `k = depth // 2 = 4`:

    Start: […]
    Apply the following 8 permutations to the starting arrangement, in order:
    […] […] […] […] 31542 12543 23451 34521
    Each permutation is written in one-line notation over the positions 1-5: to apply a permutation
    a to the current arrangement s, the new arrangement's i-th digit is the digit that s has in
    position a_i.
    What is the final arrangement?

Paired with the gold trace for steps 1-4, whose last line is `step 4: 53421 -> 35412`, the state is
available only from the trace: the published step format restates the operator and the resulting
arrangement, so a model that is threading state can continue from step 5, while one that was
re-reading the prompt cannot.

## Corruption (AMENDMENT 6)

`corrupt_step(inst, k, seed) -> (step_text, corrupted_state)` rewrites step k so that the
arrangement it reports is a plausible wrong one: **the true state with two positions swapped**.
A transposition is the minimal same-shape lie here — the result is always a different string, always
a permutation of the same digits, and always reachable by one wrong lookup, which is exactly the
mistake a model makes on this task. The step's action text is untouched; only the reported result
moves. The pair of positions is drawn from a RNG keyed on `(start, actions, k, seed)`, so the
corruption is a deterministic function of `(inst, k, seed)` and is the *same wrong state* in both
formats (only its rendering differs).

`inst.meta["format"]` decides the rendering. Both are shown below for the published exemplar at
`k = 2`, `seed = 7`, whose true state is `15342`; the corrupted state returned is `15324` in each
case (positions 4 and 5 swapped):

    format="published"
      gold:      step 2: 31254 -> 15342
      corrupted: step 2: 31254 -> 15324

    format="ergonomic"
      gold:      step 2: apply 31254 to 53124 -> take positions 3,1,2,5,4 of 53124 -> 1 5 3 4 2 -> 15342
      corrupted: step 2: apply 31254 to 53124 -> take positions 3,1,2,5,4 of 53124 -> 1 5 3 2 4 -> 15324

In the ergonomic rendering **both** state fields are rewritten — the spaced digits *and* the joined
arrangement — because they are the same object written twice; leaving them inconsistent would read as
a typo the model can notice and repair, not as a state it must carry forward. The operand (`53124`),
the permutation (`31254`) and the `take positions` list are left exactly as the gold step has them:
the corrupted line still *claims* to be the result of the right lookup.

`--selftest` checks 20 instances × 2 formats × `k ∈ {1, depth//2, depth}`: the corrupted state differs
from `states[k-1]`, is an arrangement of the same digits differing in exactly two positions, the step
text differs from `steps[k-1]` but still matches that format's step-line regex with the action,
operand and position list intact, the two formats agree on the wrong state, and repeated calls are
identical.

## Caveats

1. **The trace is from a follow-up paper, not the primary source.** Liu et al. publish no textual
   trace at all; the notation here is belindal/state-tracking's. Flagged in `desk.json`,
   `published_trace.txt`, and above.
2. **States are ours to render.** See "Format decision" — the published format shows the model no
   intermediate state; anyone comparing token counts against `desk.json.tokens_per_step` must
   compare like with like.
3. **The composition convention is stated in the prompt** where the published setup taught it by
   fine-tuning. A model that adopts the other convention (right action) will fail systematically
   rather than randomly; worth checking in error analysis before reading a low score as
   state-tracking failure.
4. **`format_natural` is false.** No model emits this scratchpad unprompted; the harness must
   supply the format via `exemplars()`. This is what AMENDMENT 5's `format="ergonomic"` addresses;
   the published format is retained unchanged as the reference condition.
5. **Action vocabulary is all of G** (all 120 elements of S5), matching the published generator,
   not a small named generator set. Actions are therefore not memorizable and contamination risk
   is low, but each step is a full 5-digit lookup rather than a familiar named move.
6. `n > 9` is rejected: the published notation is one digit per item.
7. **A corrupted state can leave the subgroup.** Under `group="A5"` or `"Zn"` every true state is
   an even permutation / a rotation, but a transposition of it is neither. The corrupted state is
   always a legal *arrangement* of the same digits — which is what the step line asserts and what a
   model reading the trace can check — but a verifier that knows the subgroup could spot the lie
   without redoing the composition. Under the default S5 there is no such tell.
8. **The ergonomic prompt's worked example contains digit strings** (`21345`, `54321`, `45321` at
   n=5), so at `k = depth` the redacted prompt is not literally digit-free the way the published
   one is. Those digits are static — identical for every instance of a given `n` — and carry no
   information about the instance, and the selftest's leak assertion is scoped accordingly to the
   start line and the story line. Worth knowing if a harness greps prompts for digits.

## Files

- `task.py` — the AMENDMENT 3 interface, AMENDMENT 4 `redact_prompt` / `REDACTION_MEANINGFUL`, the
  AMENDMENT 5 `format` knob, and the AMENDMENT 6 `corrupt_step`. `python3 task.py --selftest`
  (200 instances across S5/S3/A5/Zn and depths 1–40, each checked in both formats; redaction and
  corruption each on 20 instances × 2 formats),
  `python3 task.py --demo`.
- `examples.txt` — `--demo` output: 3 instances at depth 2, 3 at depth 32, then the same instance
  rendered once per format.
- `published_trace.txt`, `desk.json`, `SOURCING.md` — phase-1 artifacts, unmodified.
- `vendor/` — the three cloned repos (commits and licenses above).
