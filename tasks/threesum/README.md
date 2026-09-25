# threesum

3SUM over digit tuples mod 10, with the *serial* ("instance-adaptive") chain of thought
from Pfau et al.'s filler-token paper.

> Given `n` tuples of `dimension` digits mod `mod`, decide whether some three of them sum to
> the all-zero vector coordinate-wise. Answer: `True` or `False`.

## Source

- **Paper (primary source):** Jacob Pfau, William Merrill, Samuel R. Bowman (2024),
  *"Let's Think Dot by Dot: Hidden Computation in Transformer Language Models."*
  arXiv:2404.15758. <https://arxiv.org/abs/2404.15758>
- **Repo:** <https://github.com/JacobPfau/fillerTokens>, vendored shallow at commit
  `cb39af6458b7476ba07f25e89a9c8fd339c1e229` in `vendor/fillerTokens/`.
- **Published trace:** `published_trace.txt` — one `serial_cot()` rendering at
  `dimension=3, mod=10, length=6`, regenerated from the repo's own algorithm (see that file's
  header for exactly how). The paper prints no serial-CoT trace in its body text.

### License

The repo ships **no LICENSE file** and GitHub's license API returns 404 — treat it as
"all rights reserved, publicly viewable for research reference." Nothing in `task.py` is
copied from it. `src/match3.py` was read as an algorithm specification and reimplemented from
scratch in pure python; the vendored tree is kept only so the reimplementation is auditable.

### Vendored vs. written

| | |
|---|---|
| **Vendored** | `vendor/fillerTokens` (reference only — never imported; it pulls in numpy/torch/transformers at module load). |
| **Written here** | Everything in `task.py`: instance generator, candidate enumeration, serial-CoT renderer, the ergonomic renderer and prompt, independent solver, tolerant answer parser, depth→`n` calibration, redaction, step corruption, exemplars. |

There is no fixed published dataset for this task (`data/` and `output_dir/` in the repo are
empty stubs), so `generate()` is the only instance source; `exemplars(k, seed)[0]` supplies the
one published instance.

## Format decision

The gold trace reproduces `Match3.serial_solve()` + `serial_cot()` exactly. A rendered
instance in the published format is a single space-separated line:

```
901 457 892 282 752 384 P 0- 1- 4- 1 7 2 0 0 A True
```

- Everything up to and including `P` is the problem statement (`Instance.prompt`).
- Everything between `P` and `A` is the serial CoT.
- `A <True|False>` is the published closing.

The CoT enumerates **candidate triples**: position triples `d < e < f` whose *coordinate-0*
digits already sum to 0 mod `mod` (finding these is 1-D 3SUM). Each candidate is emitted as

```
<d>- <e>- <f>-   <x> <y> <z>   <s1> [<s2> ...]
```

1. three position tokens, each a digit followed by a literal `-`;
2. one raw input digit copied out of a **single randomly chosen coordinate** of each of the
   three tuples (the paper's way of forcing the model to copy and project the D-dimensional
   input, not just index it);
3. the coordinate-wise sums mod `mod` for coordinates `1, 2, ...`, **stopping at the first
   nonzero sum** — a non-match is proved lazily, so a step is 7 tokens when it dies on
   coordinate 1 and 8 when it survives to coordinate 2 (at `dimension=3`).

Enumeration stops at the first candidate that is zero in every coordinate.

**One step = one candidate-triple check** (points 1–3 together), i.e. `steps[k]` is one such
group — 7 or 8 whitespace tokens at `dimension=3`, `6 + (1 … dimension-1)` in general.
`format_cot()` joins them with single spaces exactly as the paper does, so
the published portion is byte-for-byte the published format on one line.

### Deviations from the published string (all deliberate, all small)

1. **`\nAnswer: <answer>` is appended** after `A <answer>`, per AMENDMENT 3's single
   harness-imposed normalization. The published closing `A True` is kept ahead of it.
2. **The leading space is dropped.** `serial_cot()` builds `' ' + ' '.join(rows)`; a chat
   user-message that begins with a space is asking for trouble with tokenizers and API
   whitespace handling, so `prompt` starts at the first digit. Nothing else changes.
3. **`prompt` carries no natural-language framing**, because the published task has none:
   Pfau et al. train a 34M-parameter model from scratch on exactly these symbol strings. The
   contract says "problem statement only, published wording," and the published wording is the
   symbol string. All task semantics therefore reach the model through the harness's few-shot
   exemplars — see caveats.
4. **Instances are not bit-identical to the paper's.** `task.py` is stdlib-only, so it uses
   `random.Random` rather than `numpy.random.default_rng`; the construction is transcribed
   faithfully (planted triple `a + b + (-a-b)`, geometric corruption capped at 3 rows with
   numpy's broadcast-assignment semantics reproduced, noise padding, shuffle, reject-if-true)
   but the stream differs.

### `desk.json` step-granularity discrepancy (not an error, just a different unit)

`desk.json` reports `steps_in_trace: 8` and `tokens_per_step: 2.25` for the published trace,
counting **each whitespace token** as a step. Under the definition above that same trace is
**one** step of 18 model tokens. Both descriptions are of the same string; the desk numbers are
left as they are, since a single `0-` token is not a state update and `tokens_per_step` is only
meaningful relative to the unit chosen. To convert: multiply desk `tokens_per_step` by 7–9.

## Ergonomic variant (`format="ergonomic"`, AMENDMENT 5)

Batch 1 showed the published symbol format does not transfer by prompting: it was taught to a
34M model by training, and a prompted instruct model cannot infer it from a few-shot block. The
`format` knob adds an SFT-free rendering of **the same task**. `format="published"` is still the
default, so every "as published" row stays reproducible.

What is identical across the two formats, asserted in `--selftest` for all 200 instances:
the instance itself (`meta["rows"]`), the candidate enumeration order, `depth`, `states`,
`answer`, `solve()`, `check()`. Only `prompt` and `steps`/`format_cot()` differ — the prompt is
plain language and each step writes its whole computation out.

`generate(4, 3, format="ergonomic")`:

```
Here are 7 3-digit numbers (read each one as 3 separate digits), numbered 0 to 6:

0: 805
1: 575
2: 835
3: 008
4: 247
5: 038
6: 111

Is there a triple of positions i < j < k whose digit-wise sums are all 0 mod 10 -- that is, the
first digits of the three numbers sum to 0 mod 10, the second digits sum to 0 mod 10, and so on
for all 3 digit positions? Answer True or False.
```

`format_cot()`:

```
(0,3,4): 805 + 008 + 247 -> (8+0+2, 0+0+4, 5+8+7) = (10, 4, 20) -> (0, 4, 0) mod 10 -> no
(0,4,5): 805 + 247 + 038 -> (8+2+0, 0+4+3, 5+7+8) = (10, 7, 20) -> (0, 7, 0) mod 10 -> no
(2,3,4): 835 + 008 + 247 -> (8+0+2, 3+0+4, 5+8+7) = (10, 7, 20) -> (0, 7, 0) mod 10 -> no
(2,4,5): 835 + 247 + 038 -> (8+2+0, 3+4+3, 5+7+8) = (10, 10, 20) -> (0, 0, 0) mod 10 -> yes
The triple (2,4,5) has every digit-wise sum 0 mod 10. So the answer is True.
Answer: True
```

Design notes, all deliberate:

- **One line per candidate check, same order, same count.** `steps` is one step per candidate,
  exactly as in the published format, so `depth` means the same thing and the depth sweep is
  comparable across formats.
- **Both lazy behaviours of the published trace are kept where they carry the task**: the
  enumeration still stops at the first hit (the last line of a positive instance is the only one
  ending `-> yes`). What is *dropped* is the published within-line laziness (Pfau emits
  coordinate sums only up to the first nonzero); the ergonomic line computes every coordinate,
  because the point of this variant is that the state is written out in full and the line is
  self-checking.
- **The random `cot_dim` copied digits are gone.** They exist in the published format to force a
  trained model to project the input; for a prompted model they are three unexplainable digits.
  The ergonomic line copies all three numbers instead, which is the same forcing function and is
  readable.
- **Positions are named `(i,j,k)`** rather than `0- 1- 4-`, and the numbers are listed one per
  line with their index, so "position 4" is checkable by eye.
- **Closing.** `The triple (i,j,k) has every digit-wise sum 0 mod 10. So the answer is True.` /
  `No triple of positions has every digit-wise sum 0 mod 10. So the answer is False.` — the
  BBH-style "So the answer is X." closing, then the mandated `Answer: X` line.
- **`exemplars(k, seed, format="ergonomic")`** generates all `k` exemplars in that format. It does
  *not* reconstruct the published instance: the one published trace is a symbol string, which is
  not an exemplar of this format. In the published format `exemplars()[0]` is unchanged.

## Depth semantics

**`depth` = the number of enumerated candidate-triple checks = `len(steps)` = `len(states)`,
exactly.**

The knob the paper actually sweeps is the list length `n` (Figure 2 sweeps 6–14 at
`dimension=3`). Candidate checks scale with it: a uniformly random list has about
`C(n,3)/mod` candidates. `generate()` therefore picks

```
n = argmin over n >= 6 of | C(n,3)/mod - depth |     (ties -> smaller n)
```

and then rejection-samples an instance at that `n` whose candidate count is exactly `depth`.
At the default `mod=10` this makes the recommended sweep coincide with the paper's own:

| depth | 2 | 4 | 8 | 12 | 22 | 36 |
|---|---|---|---|---|---|---|
| n | 6 | 7 | 9 | 10 | 12 | 14 |
| triples scanned (`C(n,3)`) | 20 | 35 | 84 | 120 | 220 | 364 |

`n` is also exposed directly as a knob: `generate(depth, seed, n=12)` fixes the length and
rejection-samples for `depth` candidates within it, which is how to sweep the paper's axis
while holding trace length constant (or vice versa).

### Two sampling conditions, and why

Both labels are conditioned on having **exactly `depth` candidate triples**, and a positive
instance is additionally required to match on the **last** of them. Consequences:

- The answer is **not inferable from the candidate count**. Without the second condition a
  positive stops early, so it would typically have *more* candidates than it emitted checks,
  while a negative always has exactly as many — a model that counted candidates could read off
  the label. The condition closes that channel.
- **Every emitted step is load-bearing.** No prefix of the trace determines the answer; the
  decision is made at step `depth` for both labels. That is the property this suite is
  measuring, so it is worth the distributional distortion.

The distortion is real and is the main caveat below: conditioned positives are "the true triple
is the last candidate," which is atypical of Pfau's unconditioned 50/50 sampler. Set
`n=<length>` and a large `max_attempts` if you want a distribution closer to the paper's.

### `DEPTHS = [2, 4, 8, 12, 22, 36]`

Trivial to hard along the paper's own length axis. `depth=2` (`n=6`) is two candidate checks
over 20 triples — a careful reader can do it by hand. `depth=12` (`n=10`) is where a
from-scratch 34M model in the paper still needs intermediate tokens; `depth=22` (`n=12`) is the
paper's headline setting, where its no-intermediate-token model sits at ≈66% against a 50%
floor; `depth=36` (`n=14`) is the top of the paper's sweep, 364 position triples to scan and a
~300-token trace, which is well past where a prompted 7B holds together. Six values, cost at
the deep end ≈25 ms per instance to sample.

## ANSWER_FORMAT

```
"either True or False"
```

Literally the published answer vocabulary: `serial_solve()` ends with `str(three_sum)`, so the
token is Python's `True`/`False`. The harness interpolates it as "…where `<answer>` is either
True or False." `check()` takes the last `Answer:` line (falling back to the last non-empty
line, which in an unprompted completion is the `… A True` trace line), then the last
`true|false|yes|no` word in it, mapping `yes`→`True` and `no`→`False`.

## Redaction (AMENDMENT 4)

`REDACTION_MEANINGFUL = True`. `redact_prompt(inst, k)` works in both formats.

The rows are the inputs and the state a step updates is one bit ("has a full match been seen
yet") plus the enumeration cursor. What steps 1..k consume is the *verifying* digits
(coordinates 1..) of the rows they name. A row's **coordinate-0 digit is not state**: it is what
makes a triple a candidate at all, so it is the enumeration itself and always stays, as does
every row still named by a candidate after step k — removing either would stop the model
continuing for reasons that have nothing to do with state. So the rule is:

> rows used by steps 1..k and by no step after k lose their verifying digits to one `[…]` span
> each; everything else is untouched.

`k=0` returns the prompt unchanged. `k=depth` blanks the verifying digits of every row the trace
ever touches — exactly the digits needed to decide the answer — so the answer becomes
unrecoverable from the problem statement while the enumeration stays readable.

`generate(4, 3)` at `k = depth//2 = 2` (published; rows 0's checks are behind it, everything else
is still ahead):

```
805 575 835 008 247 038 111 P          # inst.prompt
8[…] 575 835 008 247 038 111 P         # redact_prompt(inst, 2)
```

and the same instance in the ergonomic format redacts the same row, in place:

```
0: 8[…]
1: 575
...
```

Because the enumeration is lexicographic, low-index rows drop out of it early and the redaction
is non-empty in practice — no no-op was observed at `k=depth//2` at any depth in `DEPTHS`. A row
that recurred in every later candidate would legitimately produce one, and `redact_prompt` then
returns the prompt unchanged rather than removing something the continuation needs.

## Corruption (AMENDMENT 6)

`corrupt_step(inst, k, seed) -> (step_text, corrupted_state)` works in both formats.

The state a step reports is `<d>-<e>-<f>/<T|F>`: the candidate cursor plus one bit, "has a full
match been seen yet". The cursor is the enumeration, not something the model computes, so the
only thing there is to get wrong is **the verdict of candidate check `k`** — which is also the
task's own error mode (Pfau's negatives are near-misses, one coordinate off) and the minimal
edit AMENDMENT 6 asks for. So the corruption flips that verdict and, with it, the flag:

- **`no` → `yes`**: the offending digit-wise sums are rewritten to the nearest plausible value
  that is `0 mod m`, so the tuple reads `(0, 0, 0) mod 10` and the check reports `yes`.
- **`yes` → `no`**: exactly one verifying coordinate is fabricated nonzero.

Coordinate 0 is never touched: it sums to zero *by candidacy*, so corrupting it would contradict
the enumeration rather than the check. The positions, the copied-out operands and the `a+b+c`
term list are byte-identical to the true step — only the reported sums and the verdict move.

`generate(4, 3)` (answer `True`, so step 4 is the hit), `seed=0`:

```
# k=2, a `no` fabricated into a match
(0,4,5): 805 + 247 + 038 -> (8+2+0, 0+4+3, 5+7+8) = (10, 7, 20) -> (0, 7, 0) mod 10 -> no    0-4-5/F
(0,4,5): 805 + 247 + 038 -> (8+2+0, 0+4+3, 5+7+8) = (10, 10, 20) -> (0, 0, 0) mod 10 -> yes  0-4-5/T

# k=4, the real hit fabricated into a miss
(2,4,5): 835 + 247 + 038 -> (8+2+0, 3+4+3, 5+7+8) = (10, 10, 20) -> (0, 0, 0) mod 10 -> yes  2-4-5/T
(2,4,5): 835 + 247 + 038 -> (8+2+0, 3+4+3, 5+7+8) = (10, 6, 20) -> (0, 6, 0) mod 10 -> no    2-4-5/F
```

and the same two corruptions in the published format, where the same lie has to be told through
the lazy emission (sums are printed only up to the first nonzero, so a fabricated match grows the
run of zeros back to full length and a fabricated miss truncates it):

```
0- 4- 5- 5 7 8 7      ->  0- 4- 5- 5 7 8 0 0     # no -> yes
2- 4- 5- 5 7 8 0 0    ->  2- 4- 5- 5 7 8 6       # yes -> no, coordinate 1 breaks on residue 6
```

The rng is keyed on format-independent data (rows, true state, `k`, `seed`), so both formats
break the *same* coordinate on the *same* residue — asserted in `--selftest`.

**Lazy-stop semantics.** A fabricated `yes` at `k < depth` is a state on which the gold trace
would have ended (the enumeration stops at the first hit). That is intended and not repaired:
the model continues however it likes from the corrupted prefix, and only the final answer is
scored. The seed only has anything to choose when flipping a `yes` (which coordinate, which
residue); flipping a `no` is forced, since the one consistent lie is the all-zero match.

## Knobs

| knob | default | meaning |
|---|---|---|
| `dimension` | 3 | tuple width. Coordinate 0 selects candidates, coordinates 1.. verify them. Must be ≥ 2. |
| `mod` | 10 | modulus; one digit per coordinate, so 2 ≤ `mod` ≤ 10. |
| `n` | `None` | list length; `None` derives it from `depth` as above. |
| `label` | `None` | force the answer (`True`/`False`); `None` draws 50/50 from the seed. |
| `corruption_rate` | 4/3 | Pfau's negative-instance knob: a planted zero-sum triple is corrupted in `min(Geom(1/corruption_rate), 3)` rows, so negatives are near-misses, not random lists. |
| `max_attempts` | 400000 | rejection budget before `generate()` raises. |
| `format` | `"published"` | `"published"` = Pfau's symbol string; `"ergonomic"` = plain-language prompt and one full-state line per candidate check (see above). Same instances, same order, same `depth`/`states`/`answer`/`solve`. |

Paper's defaults (`scripts/data_match3.py`): `dimension=3, mod=10, length=10,
true_instance_rate=0.5, corruption_rate=4/3` — all matched.

## Caveats

- **Binary answer ⇒ 50% chance floor.** The canonical task is a decision problem, so
  `answer_space = 2`, a wrong answer identifies nothing about which step was dropped
  (`error_diagnostic: false`), and acc(no-CoT) has a high floor. Implemented as published, per
  SPEC ("implement the canonical version AND note the fix"). The cheap richer variant, if one
  is ever wanted, is to ask for the matching triple's indices instead of the bit — the trace
  already contains them (`states[-1]` is the cursor) — under a separate slug.
- **`dimension=1` is rejected.** With one coordinate the selector coordinate is also the only
  verifier, so every candidate is a full match and the serial trace collapses to a single step.
  The paper uses `dimension=1` only with the *parallel* (pairwise) CoT, which is a different
  format and not implemented here.
- **The prompt is opaque symbols.** A prompted instruct model sees `899 366 ... P` and must
  infer the task from the few-shot block. That is faithful — this task was published as a
  from-scratch training task, not a prompting benchmark — but it means low absolute accuracy in
  both conditions is expected and is *not* by itself evidence about CoT. Read the CoT gap, not
  the levels, and check `leaked_reasoning` / format-compliance metadata before trusting either.
  This is what `format="ergonomic"` exists to lift; the published format stays the default so the
  two can be compared on identical instances.
- **Published numbers are not comparable to ours.** Pfau et al.'s 95.1% / 93.6% / 78.7%
  (2SUM-Transform) and ~66% / ~100% (3SUM, `n=12`) are from a 34M `LlamaForCausalLM` *trained*
  on this format with dense supervision. They are a ceiling reference, not a baseline.
  `desk.json` flags the Figure-2 and Figure-5 numbers as medium confidence (extracted from
  HTML, not read off the PDF figures) — re-verify before using them to grade anything.
- **Sampling conditions distort the instance distribution** (see "Two sampling conditions").
- **`exemplars(k, seed)[0]` violates the generator's own match-last condition**, because it is
  the published instance reproduced verbatim: its triple `0,1,4` matches on the first of its two
  candidates, so its trace is one step long at `n=6`. That is published data, left alone.
- **The `cot_dim` coordinate is random per step**, as in `serial_solve()`. The copied digits are
  therefore not predictable from the position tokens alone, which is deliberate in the paper
  (it forces projection) but means a model cannot be graded on them step-by-step without
  knowing the draw.
- **Deep instances get slow.** Rejection sampling is ~25 ms at `depth=36` and grows roughly as
  `depth·sqrt(depth)·n^3`; beyond `depth≈60` prefer passing `n` explicitly.

## Running

```
python3 task.py --selftest    # 200 random instances over depth/dimension/mod, each rendered in
                              # BOTH formats, plus redaction and corruption checks; exit 0 on pass
python3 task.py --demo > examples.txt   # 3 instances at each of DEPTHS[0]/DEPTHS[-1], then the
                                        # same instance rendered once per format
```
