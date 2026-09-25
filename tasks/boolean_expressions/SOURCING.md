# SOURCING — boolean_expressions

Phase 1 (sourcing-only, per SPEC.md AMENDMENT). No task.py / generator code written.

## Candidates vendored

### 1. `vendor/BIG-Bench-Hard` (shared symlink to `../grid_navigation/vendor/BIG-Bench-Hard`)
- Source: github.com/suzgunmirac/BIG-Bench-Hard, commit `9ee07bd481feebf959a6b59d61ea57bdcf30964d` (2022-10-22), MIT license.
- Paper: Suzgun et al. 2022, "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them," arXiv:2210.09261.
- Contains: `bbh/boolean_expressions.json` — 250 canary-tagged fixed examples (input expression string, target True/False). `cot-prompts/boolean_expressions.txt` — 3-shot CoT prompt. Its scratchpad decomposes the expression into named sub-expressions (Z, A, B, ...) and evaluates innermost sub-expression first, one line per named piece — this is the target scratchpad *style* (matches the assignment's "evaluate innermost-first, one line per reduction") but is prose, not the assignment's rigid `step k: <expr>` single-line-per-step template.
- Also contains raw Codex (code-davinci-002) outputs + eval metrics for direct and CoT prompting on all 250 examples — used below for published numbers (computed directly from the vendored eval_metrics.jsonl, not from the paper's typeset table).
- Assessment: fixed dataset only, no generator, no reference solver, no knob. Answer space binary (True/False) — canonical BBH limitation, per SPEC's "What NOT to do" this is expected and should just be implemented as-is (a richer non-binary variant would need a separate slug, out of scope here).

### 2. `vendor/BIG-bench-google` (sparse checkout of google/BIG-bench)
- Source: github.com/google/BIG-bench, path `bigbench/benchmark_tasks/boolean_expressions/`, commit `092b196c1f8f14a54bbc62f24759d43bde46dd3b` (2024-01-18), Apache 2.0 license.
- Paper: Srivastava et al. 2022, "Beyond the Imitation Game," arXiv:2206.04615.
- Contains `task.py` — a REAL programmatic generator (`BooleanExpressionsTask`), not a fixed file. Takes `seed`, `expression_lengths` (tuple, 1-14, exponential complexity per length — this is effectively the depth/complexity knob the assignment asks for), `constants=("True","False")`, `binary_ops=("or","and")`, `unary_ops=("not",)`, `allow_parentheses=True`. It enumerates/samples random boolean expressions of a given token length and evaluates them via Python's own `eval()`/AST (need to read further into the file to confirm the exact evaluation mechanism — first ~100 lines show expression construction; the reference-answer computation logic is later in the 240-line file and should be read carefully before reuse, since SPEC.md requires `solve()` to be written *independently* from `generate()`, so this file's built-in evaluator should NOT be reused as `solve()` even if `generate()` borrows its expression-construction logic).
- Also has `README.md` (task documentation, human-readable, notes 428 multiple-choice queries against a dummy model) and a `results/` dir with a dummy-model transcript, not real LM results.
- No CoT scratchpad format at all — this is a plain input/target task, no rigid step-by-step template. The scratchpad (innermost-first reduction) would still need to be written from scratch on top of this generator's expression trees.
- License note: Apache 2.0, compatible with reuse; would need attribution if code from `task.py` is adapted.
- Assessment: closest thing to reusable generator logic (seeded, has a length/complexity knob), but ships no scratchpad, no reference solver usable as an independent `solve()`, and no common-interface wrapper. **Usable with a thin wrapper at best for the expression-generation half only** — the scratchpad-rendering and independent solver still have to be written.

### 3. `vendor/lm-evaluation-harness` (sparse checkout, `lm_eval/tasks/bbh/`)
- Source: github.com/EleutherAI/lm-evaluation-harness, commit `ad8737ae7fad24cf64e50fc7fc31397bff586b9e` (2026-09-10), MIT-style license (harness repo license included).
- Contains YAML task configs for `bbh_cot_fewshot_boolean_expressions` etc. across zeroshot/fewshot/cot_zeroshot/cot_fewshot variants. `dataset_path: SaylorTwift/bbh` — this is just a HuggingFace re-hosting of the identical fixed Suzgun et al. BBH data (confirmed: same 3-shot example text as vendored BIG-Bench-Hard cot-prompts, verbatim). `output_type: generate_until`, regex answer-extraction filter `"(?<=the answer is )(.*)"`, `exact_match` metric.
- Assessment: pure harness wrapper around the same fixed dataset, no new generator, no new scratchpad structure, no new numbers. Useful only as a second confirmation of the "the answer is X" answer-extraction convention and as documentation of how a standard harness parses BBH CoT outputs. Not independently useful.

### Other candidates searched but not vendored
- HuggingFace datasets `lukaemon/bbh`, `maveriq/bigbenchhard`, `Joschka/big_bench_hard`, `SaylorTwift/bbh`, `RUCAIBox/bbh`, `GPTTT/bigbenchhard`: all mirror the identical fixed Suzgun et al. BBH json data (confirmed same 250-example structure / same 3 CoT exemplars verbatim), no generator, no added value over the vendored source repo. Not cloned to avoid redundant network/storage.
- **guyuntian/CoT_benchmark** (companion code to Feng et al. 2023, "Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective," arXiv:2305.15408): checked the repo's file tree directly (`ED/`, `LIS/`, `arithmetic/`, `equation/`, `model.py`, `train.py`, `test.py`, MIT license). It does **not** include a boolean-expression/boolean-formula task at all — the paper's empirical tasks are edit-distance, longest-increasing-subsequence, multi-digit arithmetic, and linear-equation-solving. The paper's *theory* (bounded-depth Transformers can't solve circuit-value-problem-shaped tasks without CoT; CoT-augmented constant-size autoregressive Transformers can) is directly on-point for boolean-formula evaluation, which is exactly a circuit/formula-value problem, but there is no code to vendor here. Not vendored.
- Searched for CoT/NC1-complexity papers with released boolean-expression-evaluation code (Buss 1987 citing works, "Compositional Reasoning with Transformers, RNNs, and Chain of Thought" arXiv:2503.01544, "Lower Bounds for Chain-of-Thought Reasoning in Hard-Attention Transformers" arXiv:2502.02393, maxpool.dev CoT-serial-computation write-up, Nye et al. 2021 "Show Your Work" arXiv:2112.00114): these are theory papers (or, for Nye et al., a scratchpad paper covering Python-execution/arithmetic scratchpads) about circuit complexity and CoT expressivity, useful for the desk.json `theory_class` citation, but none ships a task-shaped boolean-expression generator with a rigid scratchpad — they use their own synthetic formula-evaluation setups for circuit-complexity proofs, not benchmark-style released code. Not vendored (no code artifact to vendor); cited in desk.json notes instead.

## Full per-model published numbers (Suzgun et al. 2022, Table 3, p.8)

The vendored BBH `code-davinci-002-outputs` eval_metrics gave the exact Codex row already recorded in
desk.json (88.4 direct / 92.8 CoT). Reading the paper's typeset Table 3 directly (arxiv.org/pdf/2210.09261,
page 8) gives the full row across all three model families the paper evaluated, confirming the Codex numbers
and adding InstructGPT and PaLM 540B:

| Model | Answer-only (no-CoT) | CoT |
|---|---|---|
| Random baseline | 50.0 | — |
| Prior BIG-Bench SOTA (Srivastava et al. 2022) | 68.5 | — |
| Human average / max | 79.4 / 100 | — |
| InstructGPT (text-davinci-002) | 90.0 | 87.6 |
| Codex (code-davinci-002) | 88.4 | **92.8** (best in table) |
| PaLM 540B | 83.2 | 80.0 |

Boolean Expressions is marked with the paper's λ superscript (one of the "algorithmic" BBH tasks solvable by
a rule-based algorithm without any NLP). Notably it's one of the few BBH tasks where CoT prompting **does
not help, or even hurts**, two of the three model families (InstructGPT 90.0→87.6, PaLM 540B 83.2→80.0);
only Codex improves. All three families are already well above the 79.4% human average and far above the
50% random baseline with answer-only prompting alone — i.e. near-ceiling difficulty at the *canonical* BBH
token-length distribution. This matters for benchmark design: to actually force load-bearing CoT state (the
stated purpose of this task library), instances will need materially more nesting/operators than the
canonical BBH examples, which is exactly what the google/BIG-bench `expression_lengths` knob (1–14, but the
canonical BBH set skews short) would let a from-scratch generator dial up.

From the (pre-CoT, GPT-2-only) BIG-bench task README itself: GPT-2-XL scores 61.25% @
`num_shots=3, expression_lengths=3` vs. 83.33% human average (6 subjects: 2 with technical backgrounds hit
100%, 4 without scored 70–80%); GPT-2-XL drops to 52.75% (~chance) at
`expression_lengths=8, allow_parentheses=True`. These are not directly comparable to the Suzgun CoT numbers
(different era, no reasoning prompting) but are a useful cross-check that difficulty scales with
`expression_lengths` as expected.

## Recommendation

**Nothing usable as-is; usable with a thin wrapper for expression generation only, but the scratchpad and independent solver must be written from scratch.**

The google/BIG-bench `task.py` (candidate 2) is the one piece of real generator logic found anywhere: it is seeded, has a natural length/complexity knob, and produces exactly the domain (nested True/False/and/or/not expressions) the assignment needs. But per SPEC.md's requirement that `solve()` be written independently of `generate()`, its bundled evaluator cannot simply be repurposed as the reference solver, and it has zero scratchpad/CoT structure — the assignment's "evaluate innermost-first, one line per reduction, showing the current expression" format does not exist in any vendored candidate and will need to be authored from scratch in a later phase, most likely reusing only the expression-tree-construction idea (not the code) from candidate 2, informed by the reduction style already modeled in the BBH CoT prompt (candidate 1).
