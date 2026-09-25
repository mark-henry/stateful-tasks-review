# SOURCING — last_letter_concat

Assigned: Wei et al. 2022 last-letter-concatenation task. Generator: list of random names/words,
scratchpad accumulates the string one line per word. Knob: number of words. Growing lexical state.

## Candidates found and vendored

### 1. `vendor/kojima-zero_shot_cot/` — Kojima et al. 2022 official repo (NeurIPS 2022, "Large Language
Models are Zero-Shot Reasoners"), `kojima-takeshi188/zero_shot_cot`, commit
`5ef330fcdeec0cd26aee27943504f91f8ec1c33c` (2023-03-28). Same repo used for the parity_coinflip task; see
that task's SOURCING.md for full repo notes. Vendored here:

- `dataset/last_letters/last_letters.json`: **fixed 500-example test set**, `{"question": "Take the last
  letters of each words in \"<Name1> <Name2> <Name3> <Name4>\" and concatenate them.", "answer":
  "<concatenated last letters>"}`. Fixed depth: always 4 words per instance. This, again, is a
  re-synthesis in the style of Wei et al. 2022 (who never released their own last-letter-concatenation
  test data) rather than the original itself.
- `create_dataset_for_symbolic_reasoning.py`: the shared generator (also used for coin_flip; branches on
  `--dataset last_letters`). Depends on the `names-dataset` PyPI package (not stdlib) for a US census
  name pool, takes `--dataset_size` and `--names_in_sample` (= number of words = our depth knob) as CLI
  args. Builds the answer by concatenating `name[-1]` for each drawn name in the same loop that builds
  the question string — i.e. generate-and-label are fused, no independent solver, and no per-word
  scratchpad/running-string state is emitted (only the final question+final answer).
- No LICENSE file found anywhere in the repo (checked root + subdirs). Vendored for citation/reference
  only; flagged with `NOTICE_no_license.txt`.

### 2. `vendor/DataGenLM-last-letter-concatenation/` — `atfortes/DataGenLM`,
`last-letter-concatenation/last_letter_concatenation.py`, MIT license (LICENSE file present, copyright
"Google 2021", inherited from the BIG-bench license per this generator's lineage). A cleaned-up
reimplementation of the same Kojima logic: `--names_in_sample` (depth), `--dataset_size`,
`--random_seed`. Still depends on `names-dataset`, still no scratchpad/running-state trace, still no
independent solver (label computed in the same loop as the question text).

- `uclaml/Rephrase-and-Respond`'s `last_letter_concat.py` is a byte-for-byte copy of this same
  DataGenLM script (confirmed by diffing; its own header comment credits
  `atfortes/DataGenLM/.../last_letter_concatenation.py`) — not separately vendored since it adds nothing.
- A Hugging Face dataset mirror, `yoonholee/last-letter-concatenation`, also exists but is (per its
  viewer) just a re-hosting of one of these same generated JSON sets; not vendored since it would require
  network access at runtime and adds no code/format not already covered by the two repos above.

## Searched, not found

- **Wei et al. 2022 official repo**: as with coin-flip, the CoT paper never published an official
  code/data release. All last-letter-concatenation data in circulation traces back to this
  Kojima -> DataGenLM -> (Rephrase-and-Respond, HF mirror) lineage, not to the original paper.
- **BIG-bench / BIG-Bench-Hard**: no `last_letters` / `last_letter_concat` task in either suite.
- **lm-evaluation-harness**: no hits for `last_letters`. This task, like coin-flip, lives only in the
  small CoT-replication-repo ecosystem, never absorbed into the major eval harnesses.
- No repo anywhere emits a **per-word running-scratchpad** trace (word 1 -> partial string, word 2 ->
  partial string, ...); every implementation found only stores the final question/answer pair.

## Assessment against common interface

Every candidate found (a) fuses generation and labeling into one pass with no independently-written
solver, (b) has no `steps`/`states` scratch trace (just a final Q/A pair) despite this task being
naturally suited to one (the running concatenated string IS the state, and it's trivial to emit
per-word), and (c) depends on the non-stdlib `names-dataset` package for the word/name pool.

**Recommendation: nothing usable as-is; usable only as phrasing/format reference; must write.** Borrow
the exact prompt phrasing ("Take the last letters of each word in "..." and concatenate them.") and the
depth-as-word-count knob convention from Kojima/DataGenLM, but swap `names-dataset` for a small
stdlib-only word/name pool (`random.choice` over a hardcoded list, or synthesize short pronounceable
tokens) to stay pure-stdlib per spec, and write `generate`/`solve`/the per-word running-scratchpad from
scratch.
