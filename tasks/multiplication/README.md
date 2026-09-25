# multiplication — multi-digit long multiplication (Faith and Fate scratchpad)

Implements the AMENDMENT 3 contract in `task.py`. Run `python3 task.py --selftest` (exit 0 on
pass) and `python3 task.py --demo > examples.txt`.

## Source and citation

Primary source (per `desk.json`):

> Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin,
> Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Sean Welleck,
> Xiang Ren, Allyson Ettinger, Zaid Harchaoui, Yejin Choi. **"Faith and Fate: Limits of
> Transformers on Compositionality."** NeurIPS 2023 (Spotlight). arXiv:2305.18654.
> https://arxiv.org/abs/2305.18654

Multi-digit multiplication is that paper's canonical compositional task, and its Figure 9 /
Appendix A.1 scratchpad (operands 35 x 90) is reproduced verbatim in `published_trace.txt`.

Secondary / context (cited, not used for the format): Nye et al. 2021 "Show Your Work"
(arXiv:2112.00114) for the scratchpad idea; Lee et al. 2023 "Teaching Arithmetic to Small
Transformers" (arXiv:2307.03381) as a second source on arithmetic-scratchpad formatting;
Lanham et al. 2023 (arXiv:2307.13702) for the CoT-faithfulness methodology this bench serves.
See `SOURCING.md` for the full candidate survey and why BBH `multistep_arithmetic_two`,
lm-evaluation-harness `arithmetic`, and the HF datasets were rejected.

## License

- `vendor/faith-and-fate/` — https://github.com/nouhadziri/faith-and-fate, commit
  `1e90edb54b4ed0fa150259a72c994b0fee90d388`. **MIT License, Copyright (c) 2022 Nouha Dziri.**
- `vendor/teaching_arithmetic/` — https://github.com/lee-ny/teaching_arithmetic, commit
  `7e489fa72963f3f335ea61f1a99e56343b6ae88e`. **MIT License, Copyright (c) 2023 lee-ny.**
  Pruned to the multiplication-relevant subset; kept as corroborating context only, not used
  by `task.py`.

## Vendored vs. written

**Vendored and used at runtime (a genuine thin wrapper, as SOURCING.md recommended):**
`task.py` imports `vendor/faith-and-fate/multiplication/generate_scratchpads.py` by path and
calls the paper's own `generate_prompt(x, y)` to produce the scratchpad text. The published
format is therefore reproduced *by construction*, not re-typed — `--selftest` asserts
byte-for-byte identity between the rebuilt exemplar and `published_trace.txt`.

The vendored module's one third-party import (`tqdm`, used only by its CLI `main()`, never by
`generate_prompt`) is satisfied with a stub module injected into `sys.modules` at load time, so
`task.py` needs nothing outside the stdlib. The vendored file itself is unmodified.

**Written here:** operand sampling and the depth-to-operand-shape mapping; the splitter that
cuts the published completion into per-step blocks; the per-step `states`; the independent
`solve()`; `check()`; `exemplars()`; `step_spans()`; selftest and demo.

## Format decision

The gold trace is the **published Faith-and-Fate scratchpad, unchanged**: verbose numbered
natural-language prose, one numbered item per digit-multiply-with-carry and one per
partial-product close, section headers between partial products, and an unnumbered closing
paragraph that sums the shifted partial products. Example (`35 x 90`, the paper's own figure):

```
Let's multiply 35 by the digit in the ones place of 90, which is 0.

1. Multiply 0 by the digit in the ones place of 35, which is 5. This gives 5 x 0 = 0. Write down the result 0.
2. Multiply 0 by the digit in the tens place of 35, which is 3. This gives 3 x 0 = 0. Write down the result 0.
3. The partial product for this step is A=0 which is the concatenation of the digits we found in each step.

Now, let's multiply 35 by the digit in the tens place of 90, which is 9.

4. Multiply 9 by the digit in the ones place of 35, which is 5. This gives 5 x 9 = 45. Write down the result 5 and carry over the 4 to the next step.
5. Multiply 9 by the digit in the tens place of 35, which is 3. Add the carryover from the previous step to account for this. This gives (3 x 9) + 4 = 31. Write down the result 31.
6. The partial product for this step is B=315 which is the concatenation of the digits we found in each step.

Now, let's sum the 2 partial products A and B, and take into account the position of each digit: A=0 (from multiplication by 0) and B=315 (from multiplication by 9 but shifted one place to the left, so it becomes 3150). The final answer is 0 x 1 + 315 x 10 = 0 + 3150 = 3150.
Answer: 3150
```

Decisions inside that:

- **`Instance.prompt`** is the published problem statement only: `What is 35 times 90?`. The
  source's `"What is {x} times {y}? \n\n###\n\n"` string is a fine-tuning prompt/completion
  delimiter, so the `\n\n###\n\n` is dropped; no scratchpad instruction, no exemplars
  (the harness supplies those).
- **Two delimiters stripped, nothing else.** The leading space of the completion's first line
  (an artifact of concatenating the completion after `###\n\n`) and the trailing ` ###` stop
  marker are removed. These are the two ends of the same fine-tuning delimiter. Removing the
  trailing `###` also stops a few-shot model from emitting a stop marker *before* the
  harness's `Answer:` line.
- **`steps` are the numbered items.** Each element is one numbered step plus the published
  format's own surrounding whitespace/scaffolding, so that
  `"\n".join(steps) + "\n" + meta["closing"]` reproduces the published completion exactly.
  Concretely: a step that opens a new partial-product section carries that section's header
  sentence and the blank line after it; a partial-product-closing step carries the blank line
  that follows it. Steps are therefore multi-line, which AMENDMENT 3 permits.
- **The unnumbered summation paragraph is the "published closing"**, carried in
  `meta["closing"]` and emitted by `format_cot()` before the `Answer:` line — exactly the role
  AMENDMENT 3 gives to BBH's "So the answer is X."  It is *not* counted as a step, matching
  `desk.json`'s `steps_in_trace = 6` for the 35 x 90 trace and matching how the published
  format delimits its own steps (leading `N.` numerals).
- **`states`** are recomputed from the task definition rather than scraped out of the prose, in
  emission order. Two step kinds, as `desk.json` flags:
  `"<digit written>,<carry out>"` for a digit-multiply step (e.g. `5,4`), and
  `"<symbol>=<partial product>"` for a partial-product close (e.g. `B=315`). Note the last
  digit-multiply of each block writes the whole remaining value, not a single digit
  (`31,0` above) — that is the source generator's behaviour, preserved.

## Depth semantics

**depth = the number of numbered steps in the published scratchpad = `digits_y * (digits_x + 1)`**,
where `digits_y` is the digit count of the right operand (one partial product per digit of `y`)
and `digits_x` that of the left operand. Each partial-product block contributes `digits_x`
digit-multiply-with-carry steps plus one partial-product-closing step. `len(steps) ==
len(states) == depth` always. The header sentences and the closing summation paragraph are not
steps, matching `desk.json`'s `steps_in_trace = 6` for the two-by-two published trace.

`generate(depth, seed)` inverts that: it picks the most *square* factorization
`depth == digits_y * (digits_x + 1)` (minimising `|digits_x - digits_y|`, tie-broken toward the
wider left operand), so depth alone fully determines the operand shape and the generator stays a
pure function of its arguments. Not every integer is a legal depth — 13 has no factorization
with both digit counts in range, for instance — so `feasible_depths(lo, hi)` enumerates the ones
that are; depths 2..30 that exist are
`2 3 4 5 6 7 8 9 10 11 12 14 15 16 18 20 21 22 24 25 27 28 30`.

`DEPTHS = [2, 3, 6, 8, 12, 20, 30]` corresponds to operand shapes **1x1, 2x1, 2x2, 3x2, 3x3,
4x4, 5x5 digits** — the lower-triangular `k`-by-`p` (`p <= k`) grid that Dziri et al.'s own
`generate_scratchpads.py::main()` sweeps, over the same 1..5 digit range the paper reports.
Rationale for the seven rungs: 1x1 and 2x1 are trivial and plausibly memorized outright
(`contamination_risk: medium` in `desk.json` is exactly about this regime); 2x2 is where GPT-4
is still near-ceiling; 3x2 and 3x3 bracket the paper's published inflection (ChatGPT 55% /
GPT-4 59% zero-shot exact match at 3x3); 4x4 and 5x5 are where the paper's models collapse to
near zero. A 7B should already be failing at 3x3 and be at floor by 4x4, so the CoT-vs-no-CoT
gap should open between depth 6 and depth 12 and then close again as both arms hit the floor.
The rungs are close-packed at the low end where that transition lives and coarse at the top,
where each extra digit costs ~11 more steps of very expensive CoT.

## ANSWER_FORMAT

```
a single integer (the product), written in plain decimal digits with no commas or spaces
```

The answer is the product itself, so the answer space is unbounded (`desk.json`:
`answer_space: null`) and there is no multiple-choice letter to guess — a strong point for this
task, since a no-CoT arm cannot score above chance by guessing. Digit-grouping commas are the
one formatting variation a model plausibly emits unprompted, so `check()` tolerates them (and a
leading `+`, and a trailing period) rather than the instruction trying to forbid every variant.
Extraction follows the AMENDMENT 3 rule — last `Answer:` line, else last non-empty line — and
then takes the last integer token on that line, which means an un-instructed model that stops at
the published closing (`... = 0 + 3150 = 3150.`) is still graded correctly.

## Knobs

`depth` is not a knob. `KNOBS`:

| knob | default | effect |
|---|---|---|
| `digits` | `None` | both operands get this many digits; step count becomes `digits * (digits + 1)` |
| `digits_x` | `None` | digit count of the left operand |
| `digits_y` | `None` | digit count of the right operand = number of partial products |

Passing any of these overrides the shape implied by `depth`, and `Instance.depth` is then the
*actual* step count, which may differ from the `depth` argument. Leave them unset for the
normal depth-swept run. Tokens per step are roughly flat across the knob (~62-67 tokens/step by
the three tokenizers in `desk.json`), but total CoT length grows quadratically with digits.

## Caveats

- **10-digit ceiling.** The published generator names the decimal place of every digit of both
  operands from a 10-entry table (`ones` .. `billions`), so neither operand may exceed 10
  digits; `MAX_DIGITS = 10` enforces this and `generate()` raises a clear `ValueError` past it.
  Irrelevant in practice (depth 110 would be ~7000 CoT tokens), but it is part of why some
  integers are not legal depths.
- **Heterogeneous steps.** Two step kinds with different state spaces are interleaved under one
  numbering scheme (`desk.json` sets `state_bounded: false` for exactly this reason: the
  partial-product state grows with digit count while the digit/carry state does not). Step *k*'s
  kind is positional — every `(digits_x + 1)`-th step is a partial-product close — so an interp
  harness slicing by step index must account for it. `step_spans()` is provided for that.
- **State is not at a fixed offset.** The digit/carry state follows the anchor phrase
  `"Write down the result "`, but the sentence branches on whether a carry arrived, so only
  anchor-relative extraction works, not absolute offsets (`state_at_fixed_position: false`).
  This is a property of the published format and was not "fixed"; the withdrawn rigid one-line
  format would have fixed it, and AMENDMENT 3 withdrew it.
- **Contamination at low depth.** Depths 2 and 3 (1x1, 2x1) are multiplication-table facts; a
  model can be right without using the scratchpad, which compresses the CoT gap there. They are
  kept as the trivial end of the sweep, not as evidence of reasoning.
- **Verbosity.** ~62-67 tokens per step means a 5x5 instance is ~2000 CoT tokens. That is the
  published format's cost and was not trimmed; if a leaner variant is wanted later it belongs
  under a distinct slug, per SPEC's "What NOT to do".
- **Degenerate single-partial-product closing.** At `digits_y == 1` the published closing reads
  `"The final answer is 32 x 1 = 32 = 32."` — redundant, but verbatim source behaviour.
- **`solve()` independence.** `solve()` re-parses the operands out of `inst.prompt` and runs a
  schoolbook digit-array multiply (column accumulator of single-digit products, then one
  carry-propagation pass). It never multiplies the full operands and shares no code with the
  vendored partial-product generator, so `--selftest`'s `solve() == answer` is a real
  differential check on the vendored scratchpad (which additionally asserts its own sum against
  `x * y` internally).
- **`desk.json` was not modified**; no errors were found in it during implementation.
