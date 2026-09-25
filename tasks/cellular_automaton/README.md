# cellular_automaton

Elementary 1-D cellular automaton on a finite ring. The model is given a rule table, an initial
row of `width` cells, and a number of generations; it must report the row after that many
generations. Default rule 110, default width 8.

## Source and citation

Primary source (theory anchor, per `desk.json`):

- Neary, T. & Woods, D. (2006). *P-completeness of Cellular Automaton Rule 110*. ICALP 2006,
  LNCS 4051, 132–143. doi:10.1007/11786986_13.
- Cook, M. (2004). *Universality in Elementary Cellular Automata*. Complex Systems 15(1), 1–40.
  (Rule 110's Turing-completeness, which Neary & Woods sharpen to P-completeness.)
- Wolfram, S. (1984). *Universality and complexity in cellular automata*. Physica D 10, 1–35.
  (Rule numbering, the 111…000 rule-table presentation, and the additive/linear rules such as 90
  and 150.)

Both primary citations are pure complexity-theory papers: **there is no published example trace,
no LLM benchmark, and no dataset for this task.** `desk.json` records `trace_available: false`
and nulls every trace-dependent metric. An extensive phase-1 search (GitHub, HuggingFace,
BIG-bench, lm-evaluation-harness, arXiv) found nothing; see `SOURCING.md`.

Note the correction already recorded in `desk.json`: Liu et al., *Transformers Learn Shortcuts to
Automata* (arXiv:2210.10749), sometimes cited for this task, contains **no** cellular-automaton
content (its automata are mod-counters, gridworlds, permutation groups, parity, Dyck). It is not
the source here.

## What was vendored vs written

Vendored (kept for provenance only; **nothing is read at runtime**):

- `vendor/shortcut_automata/` — ClaraBing/shortcut_automata @ `60e8c76`, the Liu et al. project
  *website* (HTML/CSS/figures). No code, no data, no LICENSE file; treat as all-rights-reserved,
  citation value only.
- `vendor/rule110_span_long/` — 5 KB sample of the `N8Programs/rule110_span_long` HuggingFace
  corpus, CC0-1.0. Raw unlabelled bitstrings with no width, boundary condition, seed or step
  count recorded, so it cannot be turned into instances with ground truth.

Written from scratch: everything in `task.py` (generator, two independent solvers, prompt and
trace format, checker, exemplars). The phase-1 verdict was "nothing usable — must write", and the
transition rule is one line of bit arithmetic, so there was no procurement value in wrapping
either vendored artifact.

License of this directory's own code: same as the rest of the repo; no vendored code is imported.

## Format decision

No published CoT/scratchpad trace exists for this task, so per AMENDMENT 3 the format was our
own choice. It has now been revised **twice**, each time against measured failures, and each
revision is recorded below because the revisions are the evidence. Both the current format and
the original are reachable through the `format` knob; the original is `format="rows"`.

### Revision 1 — row-only (batch 1) failed

The batch-1 trace gave one line per generation and no per-cell work:

```
generation 1: 11100101
generation 2: 00101111
```

That asks the model to perform `width` simultaneous neighbourhood lookups *inside a single
forward pass* and emit the result as one token run. Models did not do it. Concrete failure:
DeepSeek at depth 4, generation 0 = `01010110`, wrote `generation 1: 11111101`; the correct row
is `11111110`. The error is in cell 8, whose right neighbour wraps around to cell 1 — the model
lost the periodic boundary at exactly the cell where the ring closes. A row-only trace has no
place to show the wrap-around neighbourhood, so nothing in the format forces the model to
construct it, and nothing in the trace reveals which cell went wrong.

### Revision 2 — bare per-cell lookups: better, still two systematic failure modes

Revision 2 gave one line per cell with the neighbourhood and the lookup, then the assembled row:

```
generation 1:
  cell 8: 110 -> 1
  row: 11111110
```

Second-chance results on that format: **DeepSeek 0.78 / 0.54 / 0.36 and Qwen-9B 0.59 / 0.44 /
0.12 at depths 1 / 2 / 4.** Better than row-only, but the diffs showed two failures that are
both artefacts of the format rather than of the task:

1. **Periodic wrap-around, still.** A model wrote `cell 8: 110 -> 1` where the gold neighbourhood
   is `111 -> 0`: it read the wrong right neighbour for the last cell (and, symmetrically, the
   wrong left neighbour for cell 1). Writing the neighbourhood as a bare triple `110` does not
   say *which* cells it came from, so nothing in the line is checkable and the ring's closure is
   still implicit.
2. **Row assembly.** All eight cell lines correct, then `row: 10000011` where the gold row is
   `10000101`. Gathering eight bits from eight preceding lines and transcribing them in order is
   its own error-prone operation, unrelated to the automaton.

### Revision 3 — the current default (`format="cells"`)

One step is still one generation, rendered as a multi-line block (explicitly allowed by
AMENDMENT 3): a header, one line per cell, and a restatement of the row.

```
generation 1:
  cell 1: left c8=0, self c1=1, right c2=0 -> 010 -> 1  row so far: 1_______
  cell 2: left c1=1, self c2=0, right c3=1 -> 101 -> 1  row so far: 11______
  ...
  cell 8: left c7=0, self c8=0, right c1=1 -> 001 -> 1  row so far: 11100101
  row: 11100101
```

Each part answers one of the two failure modes:

- **Neighbours are named by index *and* value** (`left c8=0`), so the wrap-around is written out
  as a cell reference the model has to commit to: cell 1's left neighbour is literally `c8` and
  cell `width`'s right neighbour is literally `c1`. Failure mode 1 becomes a visible, checkable
  claim instead of an invisible one, and the neighbourhood triple that follows is now a
  transcription of three values already on the same line, not a fresh act of indexing.
- **The row grows one bit per line** (`row so far:`), with `_` for cells not yet computed. The
  running row extends the previous line's by exactly one bit — the bit computed on that same
  line — so the model never gathers bits from earlier lines. Failure mode 2 becomes a copy of
  the immediately preceding line plus one character.
- **The final `row:` line is a pure restatement** of the completed running row on the last cell
  line (asserted in `--selftest`). It carries no new work; it exists so the load-bearing state
  still sits at one fixed, greppable position once per generation, and so the next generation's
  cell lines read from a clean row.

Other decisions, unchanged across revisions:

- **`len(steps) == len(states) == depth` holds.** `steps[k-1]` is the whole block for generation
  `k`; `states[k-1]` is the row after generation `k`. Depth is still the generation count, so
  depth sweeps stay comparable across formats modulo the format itself.
- **Cells numbered 1..width, matching the prompt** ("the cells are numbered 1 to 8"). Indices are
  given for free; the model never counts cells or generations to know where it is.
- **Machine-delimitable.** A block starts at column 0 with `generation k:`; every other line is
  indented two spaces. `^generation ` recovers the steps, `^  row: ` recovers the states (the
  `row so far:` lines use a different label on purpose, so this stays unambiguous), and
  `^  cell (\d+): left c(\d+)=([01]), self c\d+=([01]), right c(\d+)=([01]) -> (\d{3}) -> ([01])  row so far: ([01_]+)$`
  recovers every neighbour reference, every lookup and every running-row prefix — so a diff can
  now attribute an error to *which* of the two failure modes it was.
- **Bits unspaced.** Measured (tokenizers 0.21.1, local tokenizer.json for gemma-2-2b /
  Meta-Llama-3-8B / Qwen2-7B): gemma-2 and Qwen2 split a bit run into one token per digit, so one
  cell = one token for them; Llama-3 merges digits into runs of up to three, so it is *not*
  atomic there.
- **Periodic boundary and the rule table stay in the prompt**, stated explicitly ("the left
  neighbour of cell 1 is cell `width`"), all eight mappings in the published 111-down-to-000
  order. The prompt is the problem statement only — no scratchpad instruction, no exemplars, no
  answer-format sentence; the harness assembles those from `ANSWER_FORMAT` and `exemplars()`.
- `format_cot()` ends with the harness-mandated `Answer: <bits>` line. There is no published
  closing sentence to precede it.

**Cost.** ~358 tokens per generation at width 8 (gemma-2; ~356 Qwen2, ~314 Llama-3), against ~123
for revision 2 and ~15 for row-only. Ergonomics are being bought with tokens, which is why
`DEPTHS` came down again.

### The legacy format (`format="rows"`)

```
generation 1: 11100101
generation 2: 00101111
```

Kept so the formats can be compared as an experimental condition — "does per-cell work close the
CoT gap?" is a question this task can answer directly. `format` only changes the rendering: for
a fixed `(depth, seed, rule, width, avoid_absorbing)` both formats give the same prompt, the same
`states` and the same answer (asserted in `--selftest`). `examples.txt` ends with one rows-format
instance for side-by-side comparison.

## Depth semantics

`depth` = number of generations simulated = number of serial state updates = `len(steps)`. The
prompt asks for generation `depth`; generation 0 is given. Depth is not a knob (`KNOBS` holds
`rule`, `width`, `format`, `avoid_absorbing`). Under the default per-cell format one step is a
multi-line block of `width + 2` lines, which AMENDMENT 3 explicitly permits; the step count is
still the generation count.

## DEPTHS

`DEPTHS = [1, 2, 3, 4, 6, 10]`. (Batch 1 used `[2, 4, 8, 12, 16, 24]`; revision 2 used
`[1, 2, 4, 6, 10, 14]`.)

- **Measured, not guessed.** On revision 2 of the format, DeepSeek scored 0.78 / 0.54 / 0.36 and
  Qwen-9B 0.59 / 0.44 / 0.12 at depths 1 / 2 / 4. The knee is between depth 2 and depth 6, so the
  grid is dense there: 1, 2, 3, 4 sample the descent and 6, 10 anchor the floor. Revision 3
  should shift the curve right by removing two format-induced error sources, but not by an order
  of magnitude.
- **1, 2 and 4 are retained deliberately** so the grid remains directly comparable with the
  second-chance run above — the format changed, the depths should not have to.
- **Work is `depth × width` lookups**, all written out: 8 at depth 1, 80 at depth 10. Per-instance
  accuracy is roughly `(1-p)^(depth·width)` in the per-cell error rate `p`.
- **Token cost caps the top.** At ~358 tokens per generation a depth-10 gold trace is ~3.6 k
  tokens; the previous top of 14 was ~5 k, for a depth that both models would floor anyway.
- **Never a multiple of 16 at width 8.** 216 of the 256 width-8 rule-110 orbits have period 16, so
  at depth 16 generation 16 equals generation 0 for ~20 % of instances (41/200 measured) — a
  visible give-away. At 10 that is back at chance.

## ANSWER_FORMAT

`"a bit string with one character per cell, each 0 or 1, no spaces (e.g. 01101001)"`

The answer is the final row, i.e. the same object as every intermediate state — so the no-CoT
condition asks for exactly what the CoT condition would have written last, and there is no
format mismatch between conditions. It is deliberately width-agnostic (the prompt already states
the ring size) so the constant does not go stale when the `width` knob moves. Answer space is
2^width (256 at the default), which keeps the metric finite and chance-level at 1/256 — far from
the binary-answer failure mode the spec warns about.

`check()` takes the last `Answer:` line, falls back to the last non-empty line, strips a leading
`label:` if present, drops a trailing period, and keeps only `0`/`1` characters. So
`Answer: 0 1 1 0 1 0 0 1`, `Answer: 01101001.` and a bare final `generation 12: 01101001` all
parse; a wrong-length string does not match. Under the per-cell format the fallback also parses a
bare final `  row: 01101001`; a completion that stops on a `cell k: ... -> b` line does not
parse to a valid row, which is the desired behaviour (a truncated generation is not an answer).

## Knobs

| knob | default | note |
|---|---|---|
| `rule` | 110 | any Wolfram rule 0–255. 110 is P-complete (Neary & Woods 2006). 90 and 150 are additive over GF(2) and shortcuttable by repeated squaring in O(log depth) — the intended contrast condition. 30, 45, 54, 184 are the other commonly studied rules. |
| `width` | 8 | ring size; answer space 2^width, `state_bits = width`. Kept at 8 to match `desk.json` (`state_bits: 8.0`, `answer_space: 256`). Raising it raises both — and, under the per-cell format, raises tokens per generation linearly. |
| `format` | `"cells"` | trace format. `"cells"` = per-cell named neighbours + lookup + running row, then a restated row (the default; revision 3, see Format decision). `"rows"` = one line per generation, the batch-1 format, kept as a comparison condition. Does not change the instance, only its rendering. |
| `avoid_absorbing` | True | resample the initial row if the orbit hits a fixed point at or before the final generation. |

## Caveats

- **Orbits cycle.** A width-`w` ring has only 2^w states, so every trajectory is eventually
  periodic. Exhaustive enumeration of rule 110 at width 8: transients are 0–7 generations and
  periods are 1, 2, 8 or 16, with period 16 covering 216 of the 256 starts. So at depths that are
  multiples of 16 the answer frequently equals generation 0 (41/200 default instances at depth
  16), which is why `DEPTHS` tops out at 14. This is unavoidable given the brief's requirement of
  a finite answer space, and it is a *bounded* shortcut: a model can only exploit it by having
  already simulated a full transient-plus-period serially and by holding those rows in its
  context, which is the state-carrying behaviour the benchmark is measuring. It does mean depth
  past ~16 buys little additional serial difficulty at width 8; raise `width` (12 roughly doubles
  the transient, and under the per-cell format raises tokens per generation with it) before
  raising depth much further.
- **`avoid_absorbing` is best-effort, not guaranteed.** The all-zero row is a fixed point for any
  rule with `000 -> 0`, and some rule/width combinations (e.g. rule 90 at width 8) die out from
  *every* start, so the resampler is capped at 64 attempts and then accepts what it has. At the
  default knobs this never fires: 0/300 sampled default instances contained a fixed point.
- **Depth-difficulty is not uniform across the rule knob.** Rule 90's answer is a GF(2) matrix
  power and rule 184's is a particle/traffic system with conserved density; both admit genuine
  shortcuts. Only the default rule 110 carries the P-completeness claim, so a CoT-gap number
  reported against another rule is a different experiment and should be labelled as such.
- **Error attribution needs the trace, not the answer.** A wrong answer says the model went
  wrong; the first mismatching `row:` says at which generation, and — new with the per-cell
  format — the first mismatching `cell k:` line says at which cell and whether the model built the
  wrong neighbourhood (a state/wrap-around error) or looked up the right neighbourhood wrongly (a
  copy error). `error_diagnostic` is true against the trace, not against the answer alone.
- **The per-cell format is ~24× the tokens of the row format** (~358 vs ~15 per generation at
  width 8). Any comparison with batch-1 or revision-2 numbers is a comparison of different
  formats *and* different depth grids, and should be labelled as such rather than plotted on one
  curve. Depths 1, 2 and 4 are the only points that appear in every grid.
- **The format is our invention, three revisions deep.** There was no published trace to copy,
  and the default has been changed twice on the basis of observed failures. It is an
  evidence-driven format, not a validated one — and each revision trades tokens for ergonomics,
  so at some point the trace is doing the task's work for the model. The two remaining pieces of
  real work under revision 3 are (a) reading the previous row's three bits at the named indices
  and (b) the rule-table lookup; everything else is copying. If accuracy at depth 1 is still not
  near ceiling after this, the problem is not the format.
- **`_` as the placeholder in `row so far:`** is a format detail the checker does not police in
  model output; `check()` only ever reads the final `Answer:` line (or the last non-empty line),
  and strips non-`0`/`1` characters, so a stray `_` in a model's last line would be dropped
  silently rather than flagged.
- No published baseline numbers exist for this task in any form — the `published_data` entries in
  `desk.json` are complexity results, not accuracies. Anything this bench produces is the first
  measurement, with no external cross-check.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4) removes what generations 1..k consumed and keeps what
generations k+1..depth need. For this task the only per-instance state in the prompt is the
**generation-0 row**; the rule table, the ring size, the periodic boundary and the requested
generation count are static material that *every* remaining generation consumes, and there is no
per-step operator list to trim. So the redaction is all-or-nothing, exactly as for
`turing_machine`'s tag system: `k == 0` returns the prompt unchanged, any `k >= 1` replaces the
generation-0 row with a single `[…]` placeholder and changes nothing else. `k` matters only
through the length of the trace prefix the harness supplies alongside the redacted prompt.
`REDACTION_MEANINGFUL = True`.

This is a clean blinding for this task: with generation 0 gone, the only way to reach generation
`depth` is to read the last `row:` out of the supplied trace prefix and keep going. A model that
had been re-deriving the row from the prompt each time has nothing left to re-derive from.

Rendered example, `generate(depth=6, seed=1000)` at `k = depth // 2 = 3`:

```
An elementary cellular automaton runs rule 110 on a ring of 8 cells. The cells are numbered 1 to 8 from left to right, and the ring is circular: the left neighbour of cell 1 is cell 8, and the right neighbour of cell 8 is cell 1.
In each generation, every cell is updated at the same time. A cell's new value depends on the three cells (left neighbour, itself, right neighbour) in the current generation, according to the rule table:

111 -> 0   110 -> 1   101 -> 1   100 -> 0
011 -> 1   010 -> 1   001 -> 1   000 -> 0

Generation 0 is:

[…]

Run the automaton for 6 generations. What is generation 6?
```

## Corruption

`corrupt_step(inst, k, seed)` (AMENDMENT 6) returns `(step_text, corrupted_state)`: generation `k`
rewritten so that the row it reports is a plausible wrong row, and that row as a bit string in the
same canonical form as `inst.states`. The corruption is **one flipped cell** — the smallest mistake
available here, and exactly the mistake the format is designed to expose (a single misread rule
lookup). The flipped cell is chosen by a `random.Random` seeded with `(seed, k, the instance's
seed/depth/rule/width/generation-0 row)`, so it is a deterministic function of `(inst, k, seed)`.
The wrong row is always a legal row: same width, same alphabet, and it always differs from
`inst.states[k-1]`.

Both formats are handled, because the step text differs per format:

- `format="rows"` — the single line becomes `generation k: <corrupted row>`.
- `format="cells"` — the row is reported in three places and all three move together, or the
  corruption reads as a typo instead of a mistake worth propagating: the flipped cell's `-> b`
  **lookup result**, every `row so far:` prefix from that cell onward, and the closing `row:` line.
  Cell lines *before* the flip are byte-identical to the gold trace.

What does **not** change is the action: the `left c8=1, self c1=0, right c2=0 -> 100` part of every
cell line keeps the true neighbour indices, the true neighbour values, and the neighbourhood string
they spell. Only the reported output bit and the rows are wrong — so the corrupted step is a model
that looked up `111` in the rule table and wrote down the wrong answer, not a model that hallucinated
its neighbours. (The consequence, deliberately, is that a corrupted `cells` step is *locally
falsifiable*: the rule table is right there in the prompt, so one flipped lookup contradicts it. In
the `rows` format there is nothing to check it against. That asymmetry is the interesting measurement
— whether a model notices, and whether the mistake propagates either way.)

Rendered example, `generate(depth=6, seed=1000)`, `corrupt_step(inst, k=3, seed=1)`. The true
generation 3 is `00110001`; the returned `corrupted_state` is `00111001` (cell 5 flipped), and the
returned step text is:

```
generation 3:
  cell 1: left c8=1, self c1=0, right c2=0 -> 100 -> 0  row so far: 0_______
  cell 2: left c1=0, self c2=0, right c3=0 -> 000 -> 0  row so far: 00______
  cell 3: left c2=0, self c3=0, right c4=1 -> 001 -> 1  row so far: 001_____
  cell 4: left c3=0, self c4=1, right c5=1 -> 011 -> 1  row so far: 0011____
  cell 5: left c4=1, self c5=1, right c6=1 -> 111 -> 1  row so far: 00111___
  cell 6: left c5=1, self c6=1, right c7=1 -> 111 -> 0  row so far: 001110__
  cell 7: left c6=1, self c7=1, right c8=1 -> 111 -> 0  row so far: 0011100_
  cell 8: left c7=1, self c8=1, right c1=0 -> 110 -> 1  row so far: 00111001
  row: 00111001
```

Cell 5 alone reports `111 -> 1` where the rule table says `111 -> 0`; cells 1–4 are untouched, and
cells 6–8 plus the `row:` line carry the corrupted row forward. The same instance in
`format="rows"` gives `("generation 3: 00111001", "00111001")`. `k` is 1-based and out-of-range `k`
raises `ValueError`.

## Verification

- `python3 task.py --selftest` — 200 random (depth, seed) pairs across 8 rules, widths 5–12 and
  both trace formats: `solve() == answer`, `check(gold)` true, `check(wrong)` false, `generate`
  deterministic, `len(steps) == len(states) == depth`, the two formats agree on prompt/states/
  answer, every step is labelled `generation k`, and — for the per-cell format — every one of the
  `width` cell lines is re-derived from the previous row by an independent rule-bit computation
  and cross-checked against the assembled row, with the running row asserted to extend the
  previous line's by exactly one bit and the final `row:` line asserted to be a restatement of
  the last running row. Then the AMENDMENT 4 redaction checks on 20
  instances at `k in {0, 1, depth//2, depth}`: `k=0` is identity, `k>0` differs and carries
  exactly one `[…]`, the generation-0 row is gone, the rule table / boundary sentence /
  question survive, the redaction is all-or-nothing across `k`, and `k < 0` raises. Then the AMENDMENT 6
  corruption checks on 20 instances x both formats x `k in {1, depth//2, depth}`: the corrupted
  state is a legal row exactly one bit from the true one, the step text differs from the gold
  step, the call is deterministic, and - for the per-cell format - every cell line is rebuilt
  from the true previous row so the neighbour indices/values/neighbourhood string are unchanged,
  exactly one reported lookup result differs from the gold trace, the lines before the flip are
  byte-identical to it, every line from the flip onward differs, and the `row:` line restates
  the corrupted row; out-of-range `k` raises. Plus tolerant-parser spot checks and
  `exemplars(3, 0)`.
- `solve()` is written independently of `generate()`: `generate()` evolves the row cell by cell
  with a per-cell table lookup; `solve()` packs the ring into one integer and evolves it with
  bit-parallel rotate/complement/AND/OR masks. A third implementation (string slicing over a dict
  table) was cross-checked against both during development over 300 random instances: 0
  mismatches.
- `python3 task.py --demo > examples.txt` — 3 instances at depth 1 and 3 at depth 10 in the
  default per-cell format, plus one depth-10 instance rendered in the legacy `rows` format for
  comparison.
- Token counts in this README were measured with `tokenizers` 0.21.1 against the local
  `tokenizer.json` for `unsloth/gemma-2-2b`, `NousResearch/Meta-Llama-3-8B` and `Qwen/Qwen2-7B`
  (50 instances at depth 14, width 8, rule 110). `task.py` itself imports nothing beyond the
  stdlib.
- Second-chance accuracies quoted in "Format decision" and "DEPTHS" were measured by the harness
  on format revision 2, not by anything in this directory.
