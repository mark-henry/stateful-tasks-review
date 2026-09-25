# dyck — close a Dyck-k prefix

Given a prefix of a Dyck-k word (balanced sequences over up to four bracket types), emit the
sequence of closing brackets that completes it. The scratchpad walks the input one symbol at a
time and prints the stack configuration after each symbol.

## Source and citation

Primary source (format and wording): **BIG-Bench-Hard `dyck_languages`**.

> Suzgun, Scales, Schärli, Gehrmann, Tay, Chung, Chowdhery, Le, Chi, Zhou, Wei.
> *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.* arXiv:2210.09261 (2022).
> Repo: github.com/suzgunmirac/BIG-Bench-Hard, commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d`, **MIT**.

Upstream dataset: **BIG-bench `dyck_languages`** (google/BIG-bench, commit
`092b196c1f8f14a54bbc62f24759d43bde46dd3b`, **Apache 2.0**) — 1000 fixed Dyck-4 examples, no
generator shipped (the PCFG that produced them lived in a `/legacy` folder that no longer exists
on `main`).

Theory reference for the (k, m)-bounded framing used by the `max_nesting` knob:

> Yao, Peng, Papadimitriou, Narasimhan. *Self-Attention Networks Can Process Bounded Hierarchical
> Languages.* ACL 2021, arXiv:2105.11115. Code: github.com/princeton-nlp/dyck-transformer,
> commit `5d21fcfff22a2f1446e9a62d80ca3ee08ff0e419`. **No LICENSE file in the repo** — treat as
> all-rights-reserved; nothing from it is copied here (see below).

## What was vendored vs written

Vendored (phase 1, unchanged):

- `vendor/BIG-Bench-Hard` — symlink to the shared BBH checkout. `bbh/dyck_languages.json`
  (250 canary-tagged fixed examples) and `cot-prompts/dyck_languages.txt` (the 3-shot CoT prompt
  whose scratchpad format this task reproduces). **Read at runtime** by `bbh_instances()`.
- `vendor/BIG-bench-google` — the upstream 1000-example `task.json`. Not read by `task.py`.
- `vendor/dyck-transformer` — Yao et al.'s Dyck-(k, m) generator and corpora. **Not read and not
  ported.** SOURCING.md recommended porting its `DyckPDFA` off torch, but that class generates
  *complete balanced* strings that then have to be truncated, and its sampler is a torch
  `Categorical` over a hand-built transition matrix. The prefix-with-target-final-stack sampler
  written here (a reachability-constrained ±1 stack walk, ~15 lines of stdlib) is both simpler
  and a better fit for the task, and it sidesteps the missing-license question entirely. Only the
  *idea* of the (k, m) knobs is taken, and that is from the paper, not the code.

Written from scratch in `task.py`: the generator, the independent solver, the scratchpad emitter,
the tolerant checker, the fixed-set loader, and the tests.

## Format decision

The published format is reproduced exactly. `published_trace.txt` holds the first few-shot
exemplar of `cot-prompts/dyck_languages.txt` verbatim, and `--selftest` asserts byte-for-byte that
`format_cot(exemplars(1, 0)[0])`, minus the appended `Answer:` line, equals it. A gold trace is:

```
Let's think step by step.
We should process each input one by one and keep track of the stack configuration.
0: empty stack
1: [ ; stack: [
2: { ; stack: [ {
3: [ ; stack: [ { [
Now, we have reached the end. The final stack is "[ { [".
We will need to pop out "[", "{", "[" one by one in that order.
So, we need "]", "}", "]". So the answer is ] } ].
Answer: ] } ]
```

- `Instance.steps` is the numbered symbol lines only — one element per input symbol, i.e. per
  stack update. The three-line preamble (the initial `0: empty stack` state) and the three-line
  coda belong to no step and are emitted by `format_cot()`.
- An empty stack renders as the word `empty` mid-trace, exactly as in the published exemplars
  (`2: > ; stack: empty`).
- Per AMENDMENT 3, `prompt` is the problem statement only, in the published BBH wording
  (`Complete the rest of the sequence, making sure that the parentheses are closed properly.
  Input: ...`), with no scratchpad instruction and no exemplars; the harness assembles those.
  The only addition to the published format is the trailing `Answer: X` line.
- `states[i]` is the stack bottom-to-top with no spaces (`[{[`), or `-` when empty — a compact
  form of the same state the step line shows in prose.

## Depth semantics

**`depth` = number of input symbols in the sequence to be completed = number of stack updates =
`len(steps)` = `len(states)`.** Each symbol is exactly one push or one pop, so depth is the serial
step count with no fudge.

`DEPTHS = [4, 8, 16, 24, 32, 48]`. BBH's own fixed set runs 2–99 symbols (median 13, mean 24), so
this sweep spans it: 4 is trivially solvable without a scratchpad, 8–16 sits at BBH's median where
Codex scored 56.8% with CoT, and 32–48 is out in the tail where a 7B should fail with CoT and fail
badly without it. Six values, all even (see the parity caveat).

## ANSWER_FORMAT

`"a sequence of closing brackets separated by single spaces, e.g. ] } )"`

The answer is the closing sequence itself, space-separated, as BBH's `target` field writes it
(`"] } ]"`). The example deliberately uses three different bracket types so the no-CoT instruction
cannot be read as implying one type or a fixed length. `check()` is tolerant: it takes the last
`Answer:` line (falling back to the last non-empty line, which catches a bare `So the answer is
] } ].`), then compares only the bracket characters, so `]}]`, `] } ]` and `So the answer is ] } ].`
all match and spacing/punctuation never decides a grade.

## Knobs

| knob | default | meaning |
|---|---|---|
| `bracket_types` | 4 | distinct bracket types available, 1–4, taken in order `()`, `[]`, `{}`, `<>`. BBH uses all 4. |
| `max_nesting` | 8 | cap on stack depth anywhere in the sequence (the *m* of Dyck-(k, m)). Clamped to `depth`. |
| `max_answer_len` | 4 | cap on the final stack size, i.e. on the answer length. |

`max_nesting` is deliberately held **fixed** across the depth sweep so sequence length is the only
varying factor. Default 8 was chosen to track BBH's own mean max-nesting per length bucket (BBH:
2.9 at len≤4, 6.4 at len≤16, 10.5 at len≤48, uncapped; this generator at m=8: 3.1, 4.5, 6.8).
Setting `max_nesting` small (2–3) puts the task in the regime Yao et al. prove a bounded-depth
transformer can do without a scratchpad; setting it large makes the state genuinely unbounded.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4) returns the problem statement with the first `k` input
symbols — the operators consumed by steps 1..k — replaced by a single `[…]` span, keeping the
symbols for steps k+1..depth, the question wording, and nothing else changed. There is no initial
state to remove: the stack starts empty and the prompt never states it, so for this task the
operator list *is* the whole redactable surface. The symbols are separated only by spaces and are
not individually delimited, so one placeholder stands for the whole removed run rather than one per
symbol. `redact_prompt(inst, 0)` returns `inst.prompt` unchanged; `redact_prompt(inst, inst.depth)`
leaves only `Input: […]`, so a model can only continue from the stack carried in the trace
prefix the harness supplies. `REDACTION_MEANINGFUL = True`. Out-of-range `k` is clamped to `depth`;
negative `k` raises `ValueError`.

Rendered example, `generate(8, 4)` at k = depth//2 = 4 (answer `) ]`):

```
prompt:              Complete the rest of the sequence, making sure that the parentheses are
                     closed properly. Input: ( ) [ ( < > [ ]
redact_prompt(_, 4): Complete the rest of the sequence, making sure that the parentheses are
                     closed properly. Input: […] < > [ ]
```

Paired with the gold trace for steps 1..4, whose last line is `4: ( ; stack: [ (`, this is
solvable: the stack after step 4 is in the trace, and `< > [ ]` is still in the prompt. Without
that line the removed run is unrecoverable — note that `( )` at steps 1–2 cancels, so the surviving
suffix carries no information about what the stack was.

Caveat specific to this task: because the walk often returns to an empty stack, a redacted instance
whose stack happens to be empty at step k (roughly 10% of steps, and all three of
`generate(8, 1..3)`) is *easier* than the unredacted one to continue — the trace prefix tells the
model the state is `empty`, which is also the state it would have assumed with no prefix at all.
Redaction-sensitivity numbers should therefore be read against `states[k-1] != "-"`.

## Corruption

`corrupt_step(inst, k, seed)` (AMENDMENT 6) returns `(step_text, corrupted_state)`: step `k`
rewritten so the stack it reports is a plausible wrong one, plus that stack in the same canonical
form as `inst.states` (`-` for empty). The action half of the line — `k: <symbol>` — is left
byte-for-byte alone; only the text after `; stack: ` changes. The wrong stack is a minimal edit of
the true one: **replace** the top symbol with a different opening bracket, **drop** the top symbol,
or **push** one extra opening bracket (roughly uniform over the three). When the true stack is
empty there is nothing to edit down, so it is corrupted upward to a one-symbol stack. The
replacement/pushed brackets are drawn from the opening brackets that instance actually uses, so the
line stays plausible; the result is deterministic in `(inst, k, seed)` and never equal to
`states[k-1]`.

Rendered example, `generate(8, 4)` (sequence `( ) [ ( < > [ ]`, true step 4 is `4: ( ; stack: [ (`):

```
corrupt_step(_, 4, 0) -> ('4: ( ; stack: [',      '[')     # dropped the top symbol
corrupt_step(_, 4, 1) -> ('4: ( ; stack: [ ( (',  '[((')   # pushed a spurious symbol
corrupt_step(_, 4, 7) -> ('4: ( ; stack: [ ( <',  '[(<')   # replaced the top symbol
```

There is no `format` knob here (dyck was not one of the AMENDMENT 5 ergonomic tasks), so a single
step-line shape covers every trace this task emits.

Two task-specific things to expect when reading mistake-propagation results:

- A **drop** or **push** changes the stack's parity, so a model that propagates the mistake
  faithfully returns an answer of the wrong *length*, not merely the wrong brackets — an easy
  signal to separate "carried the error forward" from "silently recomputed".
- A **replace** or **drop** can make a later step of the gold sequence impossible (a closing
  bracket that no longer matches the reported top, or a pop from an empty stack). The harness
  does not repair the rest of the trace; the model meets that contradiction itself, and what it
  does with it — recompute from the prompt, or invent a pop — is the measurement.

## Caveats

- **Parity.** The stack moves ±1 per symbol, so the final stack size — and hence the answer length
  — always has the same parity as `depth`. With all-even `DEPTHS` and `max_answer_len=4`, answers
  are 2 or 4 brackets long. This is inherent to the task, not to this implementation (BBH's set has
  the same property), but it does mean a model gets the answer's parity for free. `max_answer_len`
  is the lever if a wider spread is wanted; an odd `DEPTHS` entry gives 1- and 3-bracket answers.
- **Answer length is capped, sequence length is not.** BBH's final stacks are only ever 1, 2 or 3
  deep however long the input; that is kept (cap 4), so difficulty scales with the length of the
  state-tracking run, not with the length of the string to be emitted. Without this the answer
  would grow with depth and confound the two.
- **Local redundancy is high and the error is not always fatal.** Because the walk returns to an
  empty stack roughly 10% of steps, a model that loses the stack mid-sequence can be rescued by a
  later reset. This inflates accuracy at large depth relative to a task with no absorbing resets
  (`s5_composition`, `cup_shuffling`). Worth remembering when comparing CoT gaps across tasks.
- **Contamination: medium.** The 250 vendored BBH examples are canary-tagged but widely mirrored,
  and the CoT prompt file itself is all over the web — the format may well be memorized even where
  the instances are not. `generate()` exists for exactly this reason; `bbh_instances()` is provided
  for reproducing published numbers, not for the sweep.
- `exemplars(k, seed)[0]` is always the published `[ { [` exemplar (depth 3), reconstructed and
  verified against `published_trace.txt`. The remaining exemplars are generated at depths 10/16/12/…
  with default knobs, seeded from `seed`. BBH's other two published exemplars (depths 15 and 29)
  are not reconstructed because only the first was saved verbatim in phase 1.
- `solve()` is a different algorithm from `generate()` (iterated deletion of adjacent matched pairs
  to a fixed point, reading the sequence back out of `prompt`), and is checked against all 250
  published BBH targets in `--selftest` as well as against the generator.
- `desk.json` was not modified; no errors found in it. One clarification worth noting: it records
  `steps_in_trace: 3` and per-step token counts from the *first* published exemplar only, which is
  a depth-3 instance; `tokens_per_step` therefore excludes the preamble/coda amortization that a
  long instance gets, and it undercounts the stack field, which lengthens with nesting
  (`tokens_flat_across_knob: false` already records this).

## Commands

```
python3 task.py --selftest     # 200 random instances + all 250 BBH examples + verbatim trace
                               # check + 20 redaction instances at k in {0, 1, depth//2, depth}
                               # + 20 corruption instances at k in {1, depth//2, depth}
python3 task.py --demo         # 3 instances at depth 4 and 3 at depth 48 -> examples.txt
```
