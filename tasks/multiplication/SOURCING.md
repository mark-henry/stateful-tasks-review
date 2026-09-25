# Sourcing notes — `multiplication`

Phase 1 (per SPEC.md AMENDMENT): sourcing only. No task.py / generator code was written.
This document surveys existing implementations for a multi-digit long-multiplication task
with a scratchpad, and recommends how much of the common interface (`Instance`, `generate`,
`solve`, `check`, `format_cot`, `step_spans`, `KNOBS`) can be procured vs. must be written.

## Candidates found

### 1. Dziri et al. 2023, "Faith and Fate: Limits of Transformers on Compositionality"
- Paper: arXiv:2305.18654 (NeurIPS 2023, Spotlight)
- Repo: https://github.com/nouhadziri/faith-and-fate — vendored at
  `vendor/faith-and-fate/`, commit `1e90edb54b4ed0fa150259a72c994b0fee90d388` (shallow
  clone, fetched 2026-09-11/12). License: MIT, Copyright (c) 2022 Nouha Dziri.
- Contains a `multiplication/` subfolder with:
  - `build_data.py` — generates raw `x*y=` pairs, no scratchpad, digit-count / count params.
  - `generate_scratchpads.py` — the core generator. For each `(x, y)`, it forms one partial
    product per digit of `y` (digit-by-digit multiply-with-carry over `x`), symbol-labels
    each partial product (A, B, C, ...), then sums the shifted partial products to the final
    product, asserting `sum(pp * 10**p) == x*y` against ground truth (i.e. it ships its own
    differential check). The algorithmic structure (per-digit-pair multiply+carry per
    partial product, then a weighted sum of shifted partial products) is exactly the
    reference algorithm we want.
  - **Format mismatch**: its scratchpad is verbose natural-language prose, e.g. "1. Multiply
    4 by the digit in the ones place of 37, which is 7. This gives 7 x 4 = 28. We write down
    8 and carry over 2." This is not the SPEC's rigid one-line-per-step symbolic format
    (`step k: <op> -> <state>`), and steps are not delimited at fixed character offsets — a
    `step_spans()` implementation over this text would need fragile regex parsing of prose,
    not clean line splitting. So the *text* is not usable as `format_cot()` output as-is.
  - `generate_graph_from_scratchpad.py`, `build_graph.py` — parse a scratchpad into a
    networkx computational graph (nodes = partial-product digits / carries / final-sum
    digits, edges = dependency). This is directly relevant prior art for reasoning-graph
    depth/width thinking, though we did not reuse it numerically.
  - `graph_error_analysis.py`, `graph_pattern_analysis.py` — analyze, per node in the
    dependency graph, whether a transformer's error at that step is (a) explainable by an
    error propagated from a parent node (compositional error propagation) or (b) explainable
    by surface pattern-matching against training examples (memorization) rather than
    algorithm execution. This is the paper's central empirical method for arguing transformers
    reduce compositional tasks to "linearized subgraph matching."
  - No published fixed dataset artifacts kept in this vendor copy (the `data/` dir in the
    original repo is a password-protected sample and was not pulled in favor of keeping
    the vendor copy lean and using the generator).
- **Fit to common interface**: `generate_scratchpads.py`'s number/carry logic could be
  reimplemented into `generate()`, but the string format must be rewritten from scratch to
  satisfy the rigid-format and `step_spans()` requirements. Nothing here plugs into
  `Instance`/`solve`/`check` directly. **Verdict for this candidate: usable with a thin
  wrapper for the algorithm's *logic* only; the CoT format itself must be written from
  scratch.**

### 2. Lee et al. 2023 (Nogueira/Lee-ny), "Teaching Arithmetic to Small Transformers"
- Paper: arXiv:2307.03381
- Repo: https://github.com/lee-ny/teaching_arithmetic — vendored at
  `vendor/teaching_arithmetic/`, commit `7e489fa72963f3f335ea61f1a99e56343b6ae88e` (shallow
  clone, fetched 2026-09-11/12). License: MIT, Copyright (c) 2023 lee-ny.
  Pruned after recording the commit hash: removed `.git`, `plots/`, `matrix_completion/`,
  `prompts/`, `run_gpt2/`, `config_gpt2/`, and all non-multiplication bulk data dirs under
  `data/` (large addition/multi-digit training corpora, tens of MB, irrelevant to this task)
  to keep the vendor copy lean (385M -> 1.5M). Kept: `data/multiplication/` (plain-format
  `x*y=` examples, no scratchpad — see below), `data/create_multiplication.ipynb`,
  `config/multiplication/`, `config2/multiplication/`, `run/run_multiplication.sh`,
  top-level `README.md`/`LICENSE`.
- This is training infrastructure for small GPT-2-scale transformers on arithmetic, testing
  formatting tricks (digit-reversal, `$`-padding) and a "detailed scratchpad" vs "simplified
  scratchpad" ablation (`algo_reason` / `simple` flags in their training configs).
- **Data actually present for multiplication is "plain" format only**: `x*y=` with no
  scratchpad (confirmed by inspecting `data/multiplication/plain/*.txt`). The
  `algo_reasoning`-format scratchpad *data* that exists in this repo is for **addition only**
  (`data/algo_reasoning/add_examples_algorithmic*.txt`); the multiplication
  `algorithmic_reasoning` training config (`config2/multiplication/algorithmic_reasoning/`)
  references a `train_multiplication_3000.txt` file that is not checked into the repo — it
  is presumably produced on-the-fly by `data/create_multiplication.ipynb`, which we opened
  and found contains only generic tokenization/data-splitting helper cells (`get_abc`,
  `make_binary_file_shuffle`, digit-reversal helpers), not a distinct scratchpad-string
  builder for multiplication that we could locate in this snapshot.
- **Verdict for this candidate: not directly usable for a multiplication scratchpad format**
  — no complete, present-in-repo generator producing multiplication CoT text. Useful only as
  corroborating evidence that "detailed scratchpad" formatting is the standard framing used
  in this literature, and as a second, independent MIT-licensed source confirming the
  digit-reversal / padding tricks are a separate concern from the scratchpad-content question
  we care about.

### 3. Nye et al. 2021, "Show Your Work: Scratchpads for Intermediate Computation"
- Paper: arXiv:2112.00114 (Google Research).
- No public code repository was found (search results and paper page turn up no GitHub link;
  this matches the earlier finding). The paper's addition/multiplication scratchpad examples
  are described in the paper text/appendix but there is no runnable generator to vendor.
- **Verdict: not vendorable.** Cite for methodology and motivation only.

### 4. Lanham et al. 2023 (Anthropic), "Measuring Faithfulness in Chain-of-Thought Reasoning"
- Paper: arXiv:2307.13702.
- No standalone public code repository for their task generators was found via search.
- **Verdict: not vendorable.** Cite for the faithfulness-measurement methodology (truncation,
  corrupting a step, paraphrasing, filler/ellipsis substitution) that motivates why this desk
  cares about clean `step_spans()` and well-defined per-step state in the first place.

### 5. BIG-bench / BIG-Bench-Hard `multistep_arithmetic` / `multistep_arithmetic_two`
- Repos: https://github.com/google/BIG-bench (task at
  `bigbench/benchmark_tasks/multistep_arithmetic`) and
  https://github.com/suzgunmirac/BIG-Bench-Hard (`bbh/multistep_arithmetic_two.json`).
- Considered but **not vendored**: this task evaluates nested arithmetic *expressions*
  (parenthesized combinations of `+ - *`), not multi-digit long multiplication of two
  numbers with a digit-by-digit scratchpad. It is a different task shape (expression-tree
  evaluation depth vs. digit-grid long multiplication) and does not match the assignment.
  Noted here for completeness since it was the most plausible BIG-bench hit for
  "multiplication"-adjacent multi-step arithmetic.

### 6. lm-evaluation-harness `arithmetic` task suite (EleutherAI)
- Repo: https://github.com/EleutherAI/lm-evaluation-harness,
  `lm_eval/tasks/arithmetic/`.
- Considered but **not vendored**: this suite is bare "What is X plus/times Y?" -> single
  final-answer QA (2-digit multiply is one of its ten subtasks), with no scratchpad/CoT
  component at all. Not useful as a scratchpad-format source; only relevant as a possible
  future "no-CoT baseline" prompt-phrasing reference, which the SPEC says the harness can
  derive directly from our own `generate(..., cot=False)`-style prompt anyway.

### 7. HuggingFace dataset search (GSM8K, DeepMind `math_dataset`, grade-school-math, etc.)
- Searched broadly; nothing found that is specifically "long multiplication with a rigid
  digit-by-digit scratchpad." GSM8K and similar are word-problem datasets, not algorithmic
  scratchpad generators; DeepMind `math_dataset` generates many question types but not the
  specific long-multiplication-with-carry-scratchpad structure we need. Not vendored.

## Recommendation

**Nothing usable as-is; usable-with-a-thin-wrapper at best, most likely "must write" for the
rigid text format.** Concretely:

- The reference *algorithm* (per-digit-pair multiply-with-carry to build each partial
  product, symbol-label partial products, then sum shifted partial products) is well
  established in Dziri et al.'s `generate_scratchpads.py` and can be reimplemented compactly
  and independently in a future `generate()` — this saves algorithm-design risk but not
  implementation effort, since none of the vendored *string formatting* meets the SPEC's
  rigid one-line-per-step / atomic-token / fixed-offset requirements.
- No candidate provides a ready `Instance`/`solve`/`check`/`step_spans` implementation, a
  rigid symbolic scratchpad format, or a drop-in `KNOBS` dict.
- Best framing: **"usable with a thin wrapper" for the arithmetic logic (partial-product +
  carry decomposition, verified against `x*y` the way Dziri et al.'s own generator asserts),
  but "must write" for everything format-and-interface-related** (the rigid step template,
  state encoding, `step_spans` offsets, differential `solve()`, and the demo depth-targeting
  logic). A human should decide at implementation time whether to import any of
  `generate_scratchpads.py`'s number-formatting helpers (e.g. `digits()`) verbatim, or to
  write everything from scratch given how small the reusable surface actually is.

## Vendored commit hashes (for the record)

| repo | path | commit | license |
|---|---|---|---|
| nouhadziri/faith-and-fate | `vendor/faith-and-fate/` | `1e90edb54b4ed0fa150259a72c994b0fee90d388` | MIT, (c) 2022 Nouha Dziri |
| lee-ny/teaching_arithmetic | `vendor/teaching_arithmetic/` (pruned to multiplication-relevant subset) | `7e489fa72963f3f335ea61f1a99e56343b6ae88e` | MIT, (c) 2023 lee-ny |
