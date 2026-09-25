# addition

Multi-digit column addition with the Nye et al. (2021) carry scratchpad: one digit column per
scratchpad line, right to left, with the running carry written as a literal `C: 0` / `C: 1`
token at the end of every line.

A second rendering of the same instances, `format="ergonomic"` (SPEC.md AMENDMENT 5), is
available and documented below; the published format stays the default.

Implemented against SPEC.md AMENDMENT 3 (batch-1 contract). The pre-amendment `task.py` draft,
which used an invented one-line-per-step template (`step 1: 7+8+carry0=15 -> digit 5 carry 1`),
has been discarded — that format is not in the literature.

## Source and citation

Primary source (the published format this module reproduces):

> Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David
> Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus
> Odena. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models."**
> arXiv:2112.00114, 2021. Figure 2 (Section 2, "Method").

Cited for context, not for format (see `desk.json` `published_data` and `SOURCING.md`):

- Lanham et al. 2023, "Measuring Faithfulness in Chain-of-Thought Reasoning" (arXiv:2307.13702),
  Section 3.2 "Addition Tasks" — free-form associative-regrouping CoT on synthetic addition, and
  the faithfulness-measurement motivation this whole suite serves. Not a carry scratchpad.
- Dziri et al. 2023, "Faith and Fate" (github.com/nouhadziri/faith-and-fate, MIT, commit
  `1e90edb54b4ed0fa150259a72c994b0fee90d388`) — columnar compositional framing of arithmetic;
  its repo implements multiplication only, no addition.

## License / vendored vs written

Nothing was vendored; `vendor/` does not exist and is not needed. Nye et al. released no code
or dataset for the addition task (see `SOURCING.md` — verdict "nothing usable, must write"),
so `task.py` is written from scratch. The one piece of third-party material in this directory is
`published_trace.txt`, a verbatim quotation of Figure 2 of arXiv:2112.00114 (a few lines of a
figure listing, decoded from the base64 plaintext block the paper's own HTML embeds) used here
for identification and verification. No runtime network access; pure python + stdlib.

## Format decision

The published format is reproduced exactly. From `published_trace.txt`:

```
Input:
2 9 + 5 7

Target:
<scratch>
2 9 + 5 7 ,  C: 0
2 + 5 , 6 C: 1  # added 9 + 7 = 6 carry 1
, 8 6 C: 0  # added 2 + 5 + 1 = 8 carry 0
0 8 6
</scratch>
8 6
```

Line template: `<unprocessed A digits> + <unprocessed B digits> , <result digits so far> C: <carry>`,
all digits space separated. Two published spellings are reproduced faithfully:

- the setup line has an empty result field, which yields the figure's **double space** before
  `C:` (`2 9 + 5 7 ,  C: 0`);
- once both operands are exhausted the whole `a + b` part disappears, comma first (`, 8 6 C: 0`).

`format_cot()` emits, in order: `<scratch>`, the setup line, the `depth` column lines
(`Instance.steps`), the consolidation line (the final carry prepended to the result, written out
even when it is `0` — `0 8 6`), `</scratch>`, the published answer line with the leading zero
stripped (`8 6`), and then the one harness-imposed normalization required by AMENDMENT 3: a
final `Answer: 86` line.

`published_exemplar(comments=True)` reproduces the figure **byte for byte**, and
`--selftest` asserts that against both a hardcoded copy and the block parsed out of
`published_trace.txt`. The `#` comments are off by default because the figure's caption says
they "are added for clarity and are not part of the target"; the `comments` knob turns them on.

`Instance.prompt` is the problem statement only, in published wording, keeping the paper's
`Input:` label and spaced digits (`Input:\n2 9 + 5 7`). The paper's `Target:` label is the
harness's business, not the task's, so it is not part of `format_cot()`.

### What counts as a step

`steps` holds only the **column lines** — the lines that update state. The `<scratch>` tag, the
setup line, the consolidation line, `</scratch>` and the answer line are scaffolding and live in
`meta` (`setup_line`, `final_line`, `answer_line`); `format_cot()` reassembles them, and
`step_spans()` still locates each step inside the rendered trace. This differs from
`desk.json`'s `steps_in_trace: 2` only in framing (that field also counts 2 for the 29+57
example — the two genuine digit-column additions — so no desk.json correction was needed).

## Ergonomic variant (`format="ergonomic"`)

AMENDMENT 5: batch 1 showed that the Nye et al. notation — a format the paper *fine-tuned* into
the model — does not transfer by prompting at the model sizes available. `format="ergonomic"` is
an SFT-free rendering of the *same instances*: plain-language prompt, schoolbook long addition
written out right to left with the state (the carry) restated on every line.

```
What is 828306 + 181525?
```

```
ones: 6 + 5 + 0 = 11 -> write 1, carry 1
tens: 0 + 2 + 1 = 3 -> write 3, carry 0
hundreds: 3 + 5 + 0 = 8 -> write 8, carry 0
thousands: 8 + 1 + 0 = 9 -> write 9, carry 0
column 5: 2 + 8 + 0 = 10 -> write 0, carry 1
column 6: 8 + 1 + 1 = 10 -> write 0, carry 1
carry out: 1
sum: 1009831
Answer: 1009831
```

Line template: `<column>: <digit A> + <digit B> + <carry in> = <total> -> write <digit>, carry
<carry out>`. Decisions:

- **Operands are plain integers** (`828306`), not the published spaced digits — the prompt is
  the sentence a person would write.
- **The carry-in term is always written, even when it is 0.** Every line then has the same
  shape, and the carry is never something the model has to remember *implicitly*; it is copied
  forward from the previous line in plain sight. This is the redundancy the format is for.
- **Column names**: `ones`, `tens`, `hundreds`, `thousands`, and past that the 1-based column
  index counted from the right — `column 5`, `column 6`, ... Place-value words (`ten thousands`,
  `hundred thousands`, ...) get long, ambiguous to tokenize and hard to get right at depth 24;
  `10^k` was the alternative and was rejected as less natural to a chat model than the small
  place-value words it does know. So the first four columns read like a school textbook and the
  rest read like an index.
- **`carry out:` is always emitted**, 0 or 1, mirroring the published format's consolidation
  line which also writes the final carry unconditionally. `sum:` restates the answer in full,
  and then the harness-imposed `Answer:` line closes.

### What is identical across formats

`generate(depth, seed, **knobs)` samples the operands *before* any rendering, from an RNG keyed
only on `(depth, seed)`, so **the instance distribution is byte-identical across formats** — the
same `(depth, seed)` gives the same two operands. `answer`, `depth`, `len(steps) == depth`,
`states`, `solve()` and `check()` are all format-agnostic (`solve()` re-parses the operands out
of whichever prompt it is given, with one regex that reads both notations). `--selftest` asserts
all of this on 60 paired instances, including that `check()` returns the same verdict for the
same completion under both formats.

What differs: `prompt` wording, the step lines, and the scaffolding in `meta` (`setup_line` /
`final_line` / `answer_line` for the published format; `carry_line` / `sum_line` for the
ergonomic one). The `comments` knob is published-only — the ergonomic column line already spells
out the addition — and is ignored when `format="ergonomic"`.

`exemplars(k, seed, **knobs)` forwards `format` (and the other knobs) to `generate`, so the
few-shot prefix is always in the same notation as the instance being solved. `exemplars(...)[0]`
is still the Figure 2 instance, 29 + 57, rendered in the requested format:

```
What is 29 + 57?
ones: 9 + 7 + 0 = 16 -> write 6, carry 1
tens: 2 + 5 + 1 = 8 -> write 8, carry 0
carry out: 0
sum: 86
Answer: 86
```

Only under `format="published"` does that exemplar carry `meta["published"] = True`, since only
then is it the published *trace*.

## Depth semantics

**depth = number of digit columns = number of digits in each operand, and `len(steps) == depth`
exactly: one step per digit column.** Both operands are generated with exactly `depth` digits and
no leading zero, which is what makes the identity exact — with ragged operand lengths the number
of columns would decouple from any single operand's width, and the published format gives no
spelling for a line where one operand is exhausted and the other is not.

A carry out of the most significant column does **not** add a step: the published format handles
it in the consolidation line, which prepends the final carry unconditionally (`0 8 6` when it is
zero, `1 9 8` for `99+99`). So `99 + 99` is depth 2 with 2 steps and a 3-digit answer. (The
discarded draft emitted an extra "carry flush" step for this case, making `len(steps)` depend on
the operands; that is gone.)

`states[i]` is the full state after step i, `"<result digits so far>|<carry out>"` — e.g.
`["6|1", "86|0"]` for 29+57. Per-step new information is one result digit plus one carry bit,
log2(20) = 4.32 bits, matching `desk.json`'s `state_bits`.

## DEPTHS

`DEPTHS = [2, 4, 6, 8, 12, 16, 24]` (7 values).

- **2** — floor anchor. Both conditions should be at ceiling; if no-CoT is not ~100% here the
  measurement is broken, not the model.
- **4, 6, 8** — where the no-CoT arm is expected to fall apart. Nye et al. trained
  in-distribution on 1-8 digits precisely because that is the interesting band, and direct-answer
  accuracy on multi-digit addition is known to degrade sharply past ~5 digits for models of this
  scale. The CoT gap should open here.
- **12, 16** — CoT arm still expected to work; longer carry chains, more chances to drop the
  carry bit.
- **24** — intended to break the CoT arm too. It is also a practical ceiling: the published
  notation restates the entire remaining problem *and* the entire accumulated result on every
  line, so trace length is **quadratic** in depth (~87 chars at depth 2, ~2.2k at depth 24, i.e.
  roughly 1.8k tokens at the ~1.17 chars/token this spaced-digit notation measured at in
  `desk.json`). Depth 32 would roughly double that again for little extra signal.

## ANSWER_FORMAT

`"a single integer written with no spaces, commas, or other separators (e.g. 86)"`.

The published trace's own last line is space-separated (`8 6`), but `answer` is the plain
integer `86`: it has to be exact-match comparable and it is also what the harness shows the
model as the assistant turn in the **no-CoT** condition (`Answer: 86`), where the spaced form
would be a weird thing to demand. `check()` forgives the difference in both directions — it
strips spaces, commas and underscores — so a model answering in either notation is scored the
same.

`check()` follows the AMENDMENT 3 rule: take the text after the last `Answer:` marker; if there
is no such line, fall back to the last non-empty line (which, for a model that emits the
published trace but forgets the final line, is the `8 6` answer line — still correct). Then
normalize: strip a trailing period, strip whitespace/commas/underscores, drop a leading `+` and
leading zeros; if what remains is not a bare integer, fall back to the trailing digit run
(so "the sum is 86" scores). Under `format="ergonomic"` that fallback lands on the `sum: 1009831`
line, which also scores.

## Knobs (`depth` is not a knob)

| knob | default | effect |
|---|---|---|
| `format` | `"published"` | `"published"` = the Nye et al. `<scratch>` notation (default, so the "as published" rows stay reproducible); `"ergonomic"` = the SFT-free variant above. Everything the harness scores is unchanged. |
| `comments` | `False` | append the figure's `  # added 9 + 7 = 6 carry 1` comments to each column line. Off by default per the caption. |
| `carry_rate` | `None` | `None` = digits uniform (a column carries 45% of the time with carry-in 0, 55% with carry-in 1). A float in [0,1] forces each column to carry with that probability: `1.0` is a maximal carry chain (state matters most), `0.0` is carry-free (state matters least — a useful control). |
| `force_carry_out` | `False` | force the top column to carry, so the answer has `depth+1` digits and the consolidation line's leading-carry case is exercised. |

`exemplars(k, seed, **knobs)[0]` is always the Figure 2 instance (29+57, comment-free target), rendered in the requested `format`.
The remaining k-1 are generated at depth 3-4, with `exemplars(..)[1]` forced to carry out of the
top column so the few-shot prompt demonstrates the leading-carry consolidation line.

## Caveats

- **Addition is in TC0.** Carry-lookahead computes all carries in constant depth, so the serial
  right-to-left carry chain is the natural human algorithm, not a computational necessity. A
  no-CoT failure here is evidence about what the model learned, not proof that the task *requires*
  serial state. This is the main reason `addition` is a weak member of this suite; see
  `desk.json` `theory_class`.
- **`local_redundancy` is high.** Step k needs only step k-1's carry bit plus two operand digits
  that are visible in the prompt. Filler-token and suffix-knockout ablations can plausibly beat
  this task by local recomputation.
- **Contamination risk is high** for the exemplar specifically: 29+57 is one of the most
  reproduced illustrations of the scratchpad technique. Evaluation instances are freshly seeded,
  so the risk is confined to the few-shot prefix.
- **State representation is unbounded in the published notation.** Each line rewrites the whole
  remaining problem and the whole accumulated result, so line length grows with depth even though
  the genuinely new state per step is 4.32 bits (`desk.json` `state_bounded: false`). Tokens per
  step are therefore *not* flat across the knob — token budget is quadratic in depth.
- **Trace length is quadratic in depth in the published format, linear in the ergonomic one.**
  The ergonomic column line is a constant ~40 characters regardless of depth, so at depth 24 it is
  roughly a third of the published trace. Token budget is therefore not comparable across formats;
  a CoT-gap comparison between them is confounded by length as well as by notation.
- `answer` is unbounded in size (`answer_space: null`), so there is no chance floor to subtract;
  a no-CoT model cannot guess.
- `solve()` is independent of the generator: it re-parses the two operands out of the rendered
  `prompt` and adds them with python's `int` addition, touching neither `_build()`'s column loop
  nor the digit lists that produced the gold trace. It therefore also validates prompt rendering.

## Files

- `task.py` — the module. `python3 task.py --selftest` (exit 0 on pass), `python3 task.py --demo`.
- `examples.txt` — `--demo` output: 3 instances at depth 2, 3 at depth 24, and the same
  depth-6 instance rendered once per format.
- `published_trace.txt` — Nye et al. Figure 2, verbatim, with extraction provenance.
- `desk.json` — phase-2 desk metrics against the published implementation (unmodified; no errors
  found while implementing).
- `SOURCING.md` — phase-1 search record and the "nothing usable, must write" verdict.
