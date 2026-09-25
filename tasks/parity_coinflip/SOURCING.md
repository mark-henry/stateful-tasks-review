# SOURCING — parity_coinflip

Assigned: Wei et al. 2022 coin-flip task + raw parity variant. Cite Anil et al. 2022 (length
generalization) and Bhattamishra et al. 2020 (counter languages / TC0). Depth knob. Note in desk.json
that parity is in TC0 (Chiang & Cholak 2022 built an explicit constant-depth transformer for it, so it
is NOT provably serial) and the 50% guessing floor.

## Candidates found and vendored

### 1. `vendor/kojima-zero_shot_cot/` — Kojima et al. 2022, "Large Language Models are Zero-Shot
Reasoners" (NeurIPS 2022), official repo `kojima-takeshi188/zero_shot_cot`, commit
`5ef330fcdeec0cd26aee27943504f91f8ec1c33c` (2023-03-28).

- Contains `dataset/coin_flip/coin_flip.json`: the **fixed 500-example coin-flip test set** actually
  used in a widely-cited CoT follow-up paper. Each example: `{"question": "A coin is heads up. <Name>
  flips/does not flip the coin. ... Is the coin still heads up? Note that \"flip\" here means
  \"reverse\".", "answer": "yes"/"no"}`. Fixed depth: always 4 names/flips per instance (not a knob).
  This is very close to, but not verbatim, Wei et al. 2022's own (unpublished) coin-flip set — Wei et
  al. never released their exact test data, so this Kojima re-synthesis (same surface form, same
  phrasing family) is the closest thing to a "canonical" fixed dataset in circulation.
- Also contains `create_dataset_for_symbolic_reasoning.py`, the **generator** that produced that json.
  It depends on the `names-dataset` PyPI package (not stdlib) for a US census name list, takes
  `--dataset_size` and `--names_in_sample` (= depth) as CLI args, and for `coin_flip` computes the
  heads-up/tails-up state incrementally per name — i.e. the per-step state (needed for our `states`
  field) is computable from this code's inner loop but is **not emitted**; the script only writes the
  final question/answer pair, no scratchpad, no per-step trace.
- Also contains the byte-identical logic branch for `last_letters` (see last_letter_concat/SOURCING.md).
- **No LICENSE file anywhere in the repo** (checked root and all subdirs) — copyright default applies.
  Vendored only for citation/reference; a `NOTICE_no_license.txt` is included as a flag. Do not
  redistribute the vendored `coin_flip.json` beyond this research use without checking with the
  authors.
- No independent `solve()`: the generator computes the label as it builds the question, so using this
  code's own logic to also "solve" would violate the spec's differential-check requirement (solve()
  must be written independently of generate()).

### 2. `vendor/DataGenLM-coin-flip/` — `atfortes/DataGenLM`, `coin-flip/coin_flip.py`, MIT license
(LICENSE file present, copyright "Google 2021" — inherited from BIG-bench's license, oddly, since this
generator is adapted from Kojima's original per its sibling repo's README acknowledgement).

- A cleaned-up variant of the same Kojima generator: `--names_in_sample` (depth), `--dataset_size`,
  `--random_seed` (seeded, so nominally deterministic given a fixed name-source snapshot). Still depends
  on the `names-dataset` package, still no scratchpad/state trace, still bundles generate+label in one
  pass (no independent solver).
- Same author also publishes `atfortes/LLMSymbolicReasoningBench` with the identical `coin_flip.py`,
  confirming this is the actively-maintained canonical version of this small generator lineage
  (Kojima's original -> DataGenLM -> LLMSymbolicReasoningBench, all by the same or citing authors).
  Not separately vendored since it is byte-identical to DataGenLM's copy.

## Searched, not found

- **Wei et al. 2022 official repo**: the CoT paper itself never published an official code/data repo;
  confirmed via general web search and arXiv listing — no GitHub link in the paper. All coin-flip and
  last-letter datasets in circulation (Kojima, atfortes/DataGenLM, uclaml/Rephrase-and-Respond, etc.) are
  independent re-synthesis, not the original.
- **Anil et al. 2022** ("Exploring Length Generalization in Large Language Models", NeurIPS 2022,
  arXiv:2207.04901), the raw-parity + scratchpad length-generalization paper explicitly cited in our
  assignment: no public code repository found. Checked the paper's own pages, semantic scholar, NeurIPS
  proceedings page, and searched `google-research/google-research` on GitHub for "parity"/
  "length_generalization" — nothing task-shaped turned up (only unrelated statistical-parity /
  quantum-sampling code). This paper's dataset and scratchpad format appear to never have been
  open-sourced.
- **Bhattamishra et al. 2020** ("On the Ability of Self-Attention Networks to Recognize Counter
  Languages", EMNLP 2020): no code repo found. Cited for the TC0/counter-language theory framing only,
  not as a code source.
- **Chiang & Cholak 2022** ("Overcoming a Theoretical Limitation of Self-Attention", ACL 2022): code IS
  public at `github.com/ndnlp/parity` (PyTorch, tested Py3.9/PyTorch1.9). Deliberately **not vendored**:
  it is a model-construction repo (an explicit constant-depth transformer that computes PARITY exactly),
  not a task/dataset generator — nothing in it matches the `Instance`/`generate`/`solve` interface. Cited
  in desk.json's `theory_class` field only, per the assignment's explicit instruction to note that
  parity is in TC0 and is NOT provably serial.
- **BIG-bench / BIG-Bench-Hard**: searched both `google/BIG-bench` and the vendored BBH repo (see
  `web_of_lies/vendor/bbh`) for "coin_flip" and "parity" tasks — neither exists in either suite. Coin-flip
  and last-letter-concat are CoT-paper-only tasks, never absorbed into the big benchmark collections.
- **lm-evaluation-harness**: searched for "coin_flip"/"parity"/"last_letters" — no hits. Confirms the
  same: these tasks live only in the small CoT-replication-repo ecosystem, not in the major eval harnesses.
- No repo anywhere implements the **raw parity** variant (bit string, count parity, no coin-flip framing)
  with a scratchpad/state-per-step format matching our interface.

## Assessment against common interface

Nothing found is usable as-is or even as a thin wrapper: every candidate (a) bundles label computation
into generation with no independent solver, (b) emits a single Q/A pair with no `steps`/`states` scratch
trace, and (c) depends on a non-stdlib package (`names-dataset`) for the name list. The raw-parity variant
has no existing implementation at all.

**Recommendation: nothing usable, must write.** Reuse only the *surface phrasing* from Kojima/DataGenLM
("A coin is heads up. `<Name>` flips/does not flip the coin. ... Is the coin still heads up?") as the
faithful prompt template, swap `names-dataset` for a small stdlib-only hardcoded name list (or `random`
+ a short fixed pool) to keep pure-stdlib, and write `generate`/`solve`/scratchpad from scratch for both
the coin-flip and raw-parity variants. Cite Anil et al. 2022 and Bhattamishra et al. 2020 for framing
(no code to borrow); cite Chiang & Cholak 2022 for the TC0 constant-depth-transformer construction in
desk.json's `theory_class`.
