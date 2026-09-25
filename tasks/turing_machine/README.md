# turing_machine — m-tag system simulation (TMBench)

## What machine model is implemented

**An m-tag system, not a head/state/tape Turing machine.**

This is worth stating loudly because the slug and the source benchmark's name both say "Turing
machine". The primary source recorded in `desk.json` is *Turing Machine Bench* (TMBench), and
TMBench's actual formalism — see `vendor/TMBench-repo/src/tag_generate.py` — is a **Post m-tag
system**: a string-rewriting machine with

- a single **queue** of symbols (no tape, no head position, no head-state register),
- a per-symbol **production rule** `symbol -> string`,
- a fixed **deletion number `m`**,

whose transition is: read the head symbol, append that symbol's production to the tail, delete `m`
symbols from the head. It halts when the queue is shorter than `m`.

Tag systems are Turing-universal (Cocke & Minsky 1964, *Universality of tag systems with P=2*,
J. ACM 11(1)), and running a deterministic machine for `T` steps is P-complete (Ladner 1975 via
CVP; Greenlaw/Hoover/Ruzzo 1995), so this still serves as the benchmark's "maximally serial"
anchor row. But it is **not** the `(head state, head position, tape)` simulator that `SOURCING.md`
originally sketched. SOURCING.md's search found no published LLM-facing trace for that literal
formalism anywhere (GitHub, HF, arXiv, CLRS, lm-evaluation-harness, BIG-bench). AMENDMENT 3
requires the gold trace to reproduce the published format of the primary source, and the
implementation brief for this task says to follow `desk.json`'s `primary_source` and its published
trace format — so the m-tag system is what got implemented. A genuine head/tape TM, if wanted,
should be a separate slug with an invented format and no published trace.

## Source, citation, license

- **Paper:** Wu, H., Han, Z., Zhou, J.T., Huang, H., Zhang, C., *Computational Reasoning of Large
  Language Models* (Turing Machine Bench / TMBench), arXiv:2504.20771 (2025).
  Repo cites itself as `@article{wu2025turing, title={Turing Machine Evaluation for Large Language
  Model}, ...}`.
- **Repo:** https://github.com/HaitaoWuTJU/Turing-Machine-Bench — shallow-cloned to
  `vendor/TMBench-repo/` at commit `9e04fcaec05664aa28fc86f20d9a2bb0e17551ea`.
- **Dataset mirror:** https://huggingface.co/datasets/Haitao999/Turing-Machine-Bench — vendored to
  `vendor/Turing-Machine-Bench/`, **Apache-2.0** (`vendor/Turing-Machine-Bench/LICENSE`). The
  GitHub repo ships no separate LICENSE file; the HF dataset card's Apache-2.0 is the license of
  record for the vendored material.
- **Published trace:** `published_trace.txt` — the verbatim `## Example:` block baked into
  TMBench's official evaluation prompt (`vendor/TMBench-repo/src/prompt.py`, `generate_prompt()`).

## Vendored vs written

| Piece | Origin |
|---|---|
| Machine semantics + step/halt rule | TMBench `src/tag_generate.py` (`mTagSystem`) |
| Published trace format (`### step N:` blocks) | TMBench `src/prompt.py`, reproduced byte-for-byte |
| Prompt wording (rules block, parameter block) | TMBench `src/prompt.py`, trimmed (see below) |
| Sampling shape + rejection rule | TMBench `random_rule` / `random_str` / `reject_sampling`, reimplemented on a seeded `random.Random` |
| `solve()` reference simulator | **wraps the vendored `mTagSystem` class**, loaded from `vendor/TMBench-repo/src/tag_generate.py` via `importlib` |
| `generate()` trajectory, `Instance`, `check`, `format_cot`, `exemplars`, selftest | written here |

The differential check is real: `generate()` computes the trajectory with its own inline
`_simulate()`, while `solve()` re-runs the instance through the paper authors' own `mTagSystem`
(which, incidentally, deletes-then-appends where the prompt text says appends-then-deletes — the
two agree exactly when `len(queue) >= m`, which the halt guard enforces, and the selftest confirms
this over 200 instances). If `vendor/` is missing, `solve()` falls back to a local transcription of
`mTagSystem.step()` and the check degrades to a self-consistency check; this is noted in-code.

Loading the vendor module executes its top-level `random.seed(1)`, so `_mtag_class()`
saves/restores the global `random` state around the import. `generate()` only ever uses a local
`random.Random(seed)`, so determinism is unaffected either way.

## Format decision

`format_cot()` reproduces TMBench's published trace format exactly, verified byte-for-byte: the
selftest renders the published exemplar (m=2, alphabet {A,B,C}, init [B C A]) and asserts the
result equals the `### step 0:` → end portion of `published_trace.txt` character for character.

```
### step 0:
   - Action: Init
   - Queue State: [B C A]

### step 1:
   - Head Symbol: B
   - Action: Append A to the end of the queue. Remove B C from the head.
   - Queue State: [A A]
...
Answer: [B]
```

Decisions inside that:

- **A "step" = one state update = one `### step N:` block, N >= 1.** `steps[i]` is the whole
  four-line block for step `i+1`; blocks are joined with a blank line, as published.
- **The `### step 0:` Init block is not a step.** It is the initial state, fully recomputable from
  the prompt, and counting it would make `len(steps) == depth + 1`. It is emitted by `format_cot()`
  as a prefix (so the published format is intact) but lives in `meta["init_block"]`, not in
  `steps`. `step_spans()` therefore starts after it.
- **`<halt>` marker** is appended to a step's `Queue State` line when the resulting queue is
  shorter than `m`, exactly as in the published example's final step. With the default knobs this
  essentially never fires on generated instances (see depth semantics), but the published exemplar
  does end in `<halt>`, so few-shot models still see the convention.
- **Closing line.** TMBench has no "So the answer is X." closing; the trace simply ends. Per
  AMENDMENT 3's one normalization, `format_cot()` appends `\n\nAnswer: <answer>` and nothing else.
- **`prompt` is the problem statement only.** It carries TMBench's `## Rules for Simulation:`
  block (that *is* the definition of the machine — without it the problem is unstated) and the
  `## The Only Problem to Solve:` parameter block, in the published wording. Dropped: the worked
  example (the harness supplies few-shot via `exemplars()`), the scratchpad instruction
  ("provide the queue's state at each step"), and the trailing `Simulation steps:` cue. The step
  budget is kept — "Stop upon reaching the halt condition or N steps" — because without it the
  question has no answer. One cosmetic normalization: the published template writes
  `Transition rules:` in the example block and `Transition Rules:` in the problem block; prompts
  here always use the problem block's `Transition Rules:`. The *trace* is untouched.

## Depth semantics

**`depth` = number of m-tag rewrite steps executed**, i.e. the number of state updates, i.e. the
number of `### step N:` blocks with `N >= 1`. `len(steps) == len(states) == depth` exactly.

To make that identity hold, `generate()` **rejects instances that reach the halt condition before
`depth` steps** and resamples (mean 1.0–1.2 attempts across all of `DEPTHS`; growth is the norm
because mean production length 3 exceeds `m = 2`). TMBench's own `reject_sampling` — no two
consecutive queue states identical — is applied as well.

The state is **unbounded**: net growth is `mean(rule length) - m = +1` symbol per step, so the
final queue runs ~8 symbols at depth 2 and ~41 (max ~89) at depth 30 with default knobs. That is
the published task's own behaviour, not something introduced here, but it means tokens per step
grow linearly in depth — see the desk.json correction below.

## DEPTHS

```python
DEPTHS = [2, 4, 8, 12, 20, 30]
```

- **30** is TMBench's own published ceiling: `tag_generate.py` uses `max_step = 31`, and
  `prompt.py` instructs the model to stop at `max_step - 1 = 30` transitions. Going past it would
  leave the published regime.
- **2** is trivial — two rewrites on a short queue, solvable without a scratchpad.
- The middle values are roughly geometric (4, 8, 12, 20), which is where TMBench's own step-wise
  accuracy curves bend for mid-size open models; a 7B should be comfortable at 2–4, degrading
  through 8–12, and near-floor by 20–30, where the exact-match target is a 30–40 symbol queue and a
  single dropped symbol anywhere in 30 serial rewrites is fatal.

## ANSWER_FORMAT

```
"the final queue state as a bracketed, space-separated symbol list, e.g. [A B C]"
```

TMBench never asks a separate question — it scores the per-step `Queue State` lines directly. The
natural single short answer in the published notation is therefore the **final queue state**, in
the same `[A B C]` bracket notation the trace uses on every line. `check()` extracts the last
`Answer:` line (falling back to the last non-empty line), pulls the bracketed content if present,
and strips spaces/commas/quotes before exact-matching — the same normalization TMBench's own
`src/acc.py` applies to model output (`content.replace(" ", "").replace(",", "")`). So
`[A B C]`, `ABC`, and `[A, B, C]` all match; anything else does not.

## Knobs

| knob | default | TMBench source |
|---|---|---|
| `m` | 2 | `delete_count = 2` |
| `alphabet_size` | 5 | `symbol_set = ['A','B','C','D','E']` |
| `rule_min_len` | 1 | `rule_min_length = 1` |
| `rule_max_len` | 5 | `rule_max_length = 5` |
| `init_min_len` | 2 | `str_min_length = delete_count = 2` (clamped up to `m`) |
| `init_max_len` | 9 | `str_max_length = delete_count + 7 = 9` |

Depth is not a knob. The selftest sweeps `m ∈ {1,2,3} × alphabet_size ∈ {2,3,5,8}`.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4, prompt blinding) replaces the **initial queue and nothing
else**. In an m-tag system the only per-instance state in the prompt is `Init:`; the transition
rules, the alphabet, `m` and the step budget are *static material* consumed by every step, including
the ones the model still has to do, so they stay. There is no per-step operator list to trim either
— the "operator" at step `n` is whichever rule the current head symbol selects, which is not known
from the prompt. So redaction here is **all-or-nothing**: `k = 0` returns `inst.prompt` byte-for-byte,
and every `k >= 1` yields the same string, with the queue replaced by one `[…]` placeholder. `k`
matters only through the trace prefix the harness supplies alongside the prompt — which is exactly
the intended manipulation: with the `### step 0:` Init block and steps `1..k` present, the model can
continue; with the initial queue gone from the prompt, it cannot re-derive step `k`'s queue any other
way, because each queue is a function of the whole history.

Rendered example, `generate(8, 42)` at `k = depth // 2 = 4` (the `## Rules for Simulation:` block
above is unchanged and elided):

```
## The Only Problem to Solve:
m: 2
Alphabet: {A, B, C, D, E}
Init: […]                 <- was: Init: [D B A E]
Transition Rules:
A : A
B : B A A
C : A C A B C
D : C
E : C C B E D
```

The selftest checks 20 instances at `k ∈ {0, 1, depth//2, depth}`: `k=0` is identity; every `k > 0`
differs from the prompt, contains `[…]`, no longer contains the rendered initial queue, and still
contains every transition rule, `m`, and the step budget; at `k = depth` no bracketed queue survives
anywhere in the prompt; and `k < 0` raises.

## Corruption

`corrupt_step(inst, k, seed) -> (step_text, corrupted_state)` (AMENDMENT 6, mistake propagation)
rewrites step `k`'s reported queue into a plausible wrong one. The whole machine state is the queue,
and the published block reports it on exactly one line, so **only the `   - Queue State:` line
changes**: the `### step N:` header, `Head Symbol:` and `Action:` lines — which restate the operator
that step applied — are byte-identical to `inst.steps[k-1]`.

The wrong queue is one **minimal edit** away from the true one: either one symbol replaced by a
different alphabet symbol, or two adjacent symbols transposed (the two slips a model actually makes
when copying a queue forward). Both preserve length and stay inside the alphabet, so the result is
indistinguishable in shape from a real queue — including its halt status, so a trailing ` <halt>`
marker is carried over verbatim. `corrupted_state` comes back in the same canonical form as
`inst.states`: the bare symbol string (`"BACA"`), not the bracketed rendering used by `answer`.

Determinism is keyed on the instance's defining parameters (`m`, alphabet, `init`, rules), `k` and
`seed` — not on object identity or `hash()` — so the same `(inst, k, seed)` yields the same pair
across processes. There is no `format` knob here (TMBench's published layout is the only format),
so AMENDMENT 6 (e) is satisfied trivially.

Rendered example, `generate(8, 42)` at `k = depth // 2 = 4`, `seed = 7` — an adjacent transposition
of the head two symbols:

```
### step 4:                                            ### step 4:
   - Head Symbol: A                                       - Head Symbol: A
   - Action: Append A to the end of the queue.            - Action: Append A to the end of the queue.
     Remove A C from the head.                              Remove A C from the head.
   - Queue State: [A B C A]        ->                    - Queue State: [B A C A]
```

(returned `corrupted_state` is `"BACA"`; the true `states[3]` is `"ABCA"`. The `Action:` line is one
line in the file; it is wrapped here only to fit the column.)

`k` outside `1..depth` raises `ValueError`. So does `alphabet_size=1`, where every same-length string
over the alphabet *is* the true state and no legal-looking wrong queue exists; that knob setting is
outside the published regime and is the only case with no corruption available.

## desk.json corrections

`desk.json` was written during the sourcing phase, when the intended implementation was still the
head/position/tape TM with a 6-cell binary tape and 3 states. Four fields were computed from that
abandoned formalism and are wrong for the task actually implemented; they have been corrected
(with a dated `CORRECTION` sentence appended to `notes`). Everything else, including all tokenizer
measurements, describes the published TMBench trace and is untouched.

| field | was | now | why |
|---|---|---|---|
| `state_bounded` | `true` | `false` | the queue grows ~1 symbol/step |
| `state_bits` | `10.17` | `null` | depth-dependent, ≈ `log2(5) * (|init| + depth)`; no single value |
| `answer_space` | `64` | `null` | unbounded — the answer is a growing queue |
| `knob.tokens_flat_across_knob` | `true` | `false` | each step restates the whole queue |
| `knob.granularity` | tape/state prose | tag-system knobs | described the abandoned formalism |

## Caveats

1. **Name vs formalism.** See the top section. Anyone reading a collated table should not assume
   this row is a head/tape Turing machine.
2. **State is unbounded**, unlike most other rows in this benchmark. Token cost per step grows
   linearly with depth, which confounds a pure depth sweep: a depth-30 trace is ~4× the tokens per
   step of a depth-2 trace. If the review needs a flat-token serial anchor, cap growth by setting
   `rule_max_len=2` (mean production 1.5 < m=2 shrinks the queue — expect heavy rejection) or
   `rule_min_len=rule_max_len=2` (exactly length-preserving: the queue stays at its initial length
   forever, which is a bounded-state regime and arguably a different task).
3. **Long exact-match answers.** At depth 30 the answer is a ~41-symbol string. `acc(CoT)` will be
   dominated by transcription slips near the end, not only by reasoning failures. TMBench's own
   headline metric is per-step accuracy for exactly this reason; the per-step trace is fully
   delimitable here (`step_spans()`), so a per-step metric can be recovered later without
   regenerating anything.
4. **Halting is suppressed by construction** so `len(steps) == depth`. Generated instances
   therefore never exercise the `<halt>` path (the published exemplar does). If a halting regime is
   wanted, it should be a separate variant.
5. **Contamination.** Fresh instances are generated per seed, so the *instances* are not in any
   public set, but the prompt template and the 100-sample TMBench bank are public on GitHub and
   HuggingFace. `desk.json` says `contamination_risk: "low"`; "low for instances, higher for
   format familiarity" is the more precise reading.
6. **`prompt.py` is not importable as a wrapper** — `generate_prompt()` reads a fixed JSON sample
   bank and returns whole prompt strings for all 100 samples. The prompt text here is reproduced
   from it rather than called into it, because the vendored function cannot render an arbitrary
   generated instance.

## Running

```
python3 task.py --selftest     # exit 0 on pass
python3 task.py --demo > examples.txt
```

`--demo` prints 3 instances at `DEPTHS[0]` (2) and 3 at `DEPTHS[-1]` (30), each as prompt +
`format_cot`. Pure python stdlib; no network.
