# SOURCING — random_lookup_table

Phase 1 (sourcing-only) per SPEC.md AMENDMENT. No `task.py` was written for this task (correctly —
the scope-change message arrived before any code for this task existed).

Task recap: state is a single symbol in [N]; at each step k a *fresh* random function
f_k: [N] -> [N] (not necessarily a bijection — must contain S_N, so NC1-hard for N>=5) is given in
the prompt as a full lookup table, and the model applies it to the current state.

## Candidates found and vendored

### 1. `vendor/compositional_capabilities/` — Ramesh et al., "Compositional Capabilities of
Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks" (arXiv:2311.12997, ICML
2024). Repo: https://github.com/rahul13ramesh/compositional_capabilities, commit
`54256841ef5f3af3a39d47288d071f7d04b6da5e`. **MIT License.**

**What it contains:** `synthetic/functions.py` (`CreateFunctions`) generates, per instance, a
sequence of `depth` **bijective** lookup tables over an alphabet of size `n_alphabets`
(`np.random.permutation(n_alphabets)`, one fresh table per depth level — this is structurally
*exactly* our task's "fresh table per step" design) and composes them
(`reduce_functions`: `cur_fn = fn_list[i][cur_fn]`, applied depth times). `synthetic/generator.py`
(`SyntheticData`) renders these into token sequences and, per `config/gen/conf.yaml`, can emit
either a **direct** (input -> output) or a **step-by-step** prompt format — i.e. it already has a
notion of a chain-of-thought scratchpad variant, closer to our interface than anything else found.

Caveats vs. our spec:
- Functions are restricted to **bijections** (`np.random.permutation`); our task explicitly wants
  arbitrary functions [N]->[N] (not necessarily injective/surjective) — this is the whole point
  (contains S_N as a sub-case but is strictly harder/more general). Relaxing this is a one-line
  change (`np.random.randint(0, n, n)` instead of `np.random.permutation(n)`) but changes the
  semantics (no longer guaranteed invertible, no longer literally "S_N" per step) and is not
  present in the repo as shipped.
  - Bug (already in code, not ours): `reduce_functions` writes `cur_fn = fn_list[i][cur_fn]`
    where `cur_fn` is initialized as the identity array `np.arange(n_alphabets)` and each
    `fn_list[i]` is itself a length-n array — i.e. this composes n_alphabets AT-ONCE (batched over
    all starting points, vectorized), not a single tracked scalar state; would need to be adapted
    to a single-symbol trace to match our `states: list[str]` (one state per step) semantics.
  - Rendering is for LM pretraining corpora (arbitrary alphabet tokens), not our
    `step k: ... -> ...` rigid line format, and there's no independent `solve()` (ground truth is
    whatever the generator itself computed) or `check()`/`step_spans()`.

**Verdict:** usable with a **non-trivial (more-than-thin) wrapper**: the "fresh table per depth,
then compose serially" structure and the direct/step-by-step prompt duality are directly on
target, but the bijection restriction must be relaxed, the composition must be pulled out of its
batched-array form into a single traced state, and all of the interface plumbing
(`Instance`/`solve`/`check`/`step_spans`) is absent and must be written.

### 2. `vendor/zoology/` — Arora, Eyuboglu et al., "Zoology: Measuring and Improving Recall in
Efficient Language Models" (arXiv:2312.04927, ICLR 2024 workshop / COLM). Repo:
https://github.com/HazyResearch/zoology, commit `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
**Apache-2.0.**

**What it contains:** `zoology/data/multiquery_ar.py` (multi-query associative recall, MQAR) and
`zoology/data/compositional_mqar.py` (compound-key variant). Both build ONE static key-value
lookup table per example and issue multiple *independent, parallel* queries against it (the model
must retrieve `value` given `key`, for several unrelated keys in one sequence) — this is a
*parallelizable* retrieval task, not a *serial* iterated-composition task: there is no notion of
"apply f_1, then feed the result into f_2, then f_3, ..." The compositional variant uses *compound
keys* (K1,K2)->V, still a single flat table, still answered independently per query — not chained
hops through a sequence of distinct tables.

**Fit to common interface:** essentially none for the *composition* structure we need (chained
serial application through k distinct fresh tables); it is the natural fit for a *different*
task shape (parallel key-value lookup) and is more relevant background/contrast for desk.json's
`local_redundancy` / parallelizability discussion than as reusable generator code.
**Verdict:** not usable for this task's core mechanic; cite as related work (associative recall /
lookup-table literature) and as a citable contrast for "why chaining through fresh tables, rather
than one flat table with parallel queries, is the serial-state-forcing design."

## Recommendation

**Nothing is usable as-is.** `compositional_capabilities`'s `synthetic/functions.py` is the
closest match found — same "fresh table per depth level, then compose" idea, MIT-licensed, and
already distinguishes direct vs. step-by-step prompting — but needs real surgery (drop the
bijection restriction to get true [N]->[N] functions containing S_N as required by the spec,
un-batch its vectorized composition into a single traced scalar state, and add the
`Instance`/`solve`/`check`/`step_spans` plumbing), so call it **usable with a substantial wrapper**
rather than a thin one. Zoology's MQAR family is a good citation for the "associative
recall" / lookup-table benchmark lineage but solves a structurally different (parallel-query)
problem and contributes no reusable generator code here. On balance a from-scratch
`task.py`, informed by `compositional_capabilities`'s design, is likely the fastest correct path.
