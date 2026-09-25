# SOURCING — s5_composition

Phase 1 (sourcing-only) per SPEC.md AMENDMENT. **`task.py` in this directory is a DRAFT written
before the amendment landed; it is NOT vetted, NOT to be trusted, and should not be extended.**
It was a first-pass from-scratch implementation of the interface (random per-instance named
generator permutations P1..Pn composed step by step) written before the scope change to
sourcing-only was received. Keeping it only so the coordinator can see what a from-scratch
attempt looked like; a human should decide whether to build on it or discard it once candidate
review below is done.

## Candidates found and vendored

### 1. `vendor/shortcut_automata/` — Liu, Ash, Goel, Krishnamurthy, Zhang, "Transformers Learn
Shortcuts to Automata" (arXiv:2210.10749, NeurIPS 2023). Repo:
https://github.com/ClaraBing/shortcut_automata, commit `60e8c76b2dda0341d3c69e180643174fcb2b5f57`
(HEAD at clone time). No LICENSE file found.

**What it contains:** This is the paper's *project website* only — `index.html`, `css/`,
`resources/` (figures/assets for the page). There is no generator, no dataset, no training or
eval code. The paper itself defines the S5/S_n word-problem task (and other automata: parity,
modular counting, etc.) but the authors did not release the code that produced their instances.

**Fit to common interface:** none — nothing to wrap.
**Verdict:** not usable. Cite the paper for motivation/theory only.

### 2. `vendor/state-tracking/` — Kim & Schuster (or similar; repo attributes to arXiv:2503.02854),
"(How) Do Language Models Track State?" Repo: https://github.com/belindal/state-tracking,
commit `fc63e2db262265f42e9d2ac5c05888284f843b4c`. **MIT License.**

**What it contains:** `permutation_task.py` — a genuine, from-scratch S3/S5 permutation-composition
generator. Key class `PermutationTask(num_items=3|5)`:
- `PermutationState` wraps a tuple permutation of `range(1, n+1)`, with `apply_action(action)`
  composing via `new_perm = tuple(self.permutation[action[j]-1] for j in range(n))` — i.e. exactly
  the S_n word-problem composition our task needs.
- `_init_actions()` uses **all n! permutations** as the action vocabulary (not a small named
  generator set of size ~4 as our spec calls for) — every step's action is itself an arbitrary
  permutation spelled out in full (as a string of n digits), not a name drawn from a small
  per-instance table.
- `simulate()` produces "stories": a sequence of (action, resulting-state) pairs rendered as plain
  digit strings, written out to train/test files, intended for GPT-2/Pythia fine-tuning, not a
  single-call `generate(depth, seed) -> Instance`.
- No `solve()` independent of generation (the state is exactly the accumulated ground truth from
  simulation — there's no separate differential check).
- No CoT-formatted prompt/scratchpad string matching `step k: <op> -> <state>`; it's tokenized
  digit sequences for LM training, not natural-language chain-of-thought text.

**Fit to common interface:** the core permutation composition math (`apply_action`) is directly
reusable and is the cleanest "reference" logic found for this task. Everything else (prompt
rendering, named-generator-table framing per spec's `step k: P3 -> b a e c d` format, `solve()`
independence, `check()`, `step_spans()`) is missing and would need to be written.
**Verdict:** usable with a thin wrapper for the core group-composition arithmetic; the
generator-table / rigid-scratchpad / interface plumbing must be written new.

### 3. `vendor/neural_networks_chomsky_hierarchy/` — Delétang et al., "Neural Networks and the
Chomsky Hierarchy" (arXiv:2207.02098, ICLR 2023). Repo:
https://github.com/google-deepmind/neural_networks_chomsky_hierarchy, commit
`2b8eb4bee872acf5b637fcc672f01c06939dc4ef`. **Apache-2.0.**

**What it contains:** `tasks/regular/cycle_navigation.py` — walks on a length-5 **cycle** (Z_5,
i.e. an *abelian/solvable* group under addition mod 5) using actions in {-1,0,1}; JAX,
one-hot/batched, classification-style (final position only, no scratchpad, no per-step string).
This is the same *family* of task (finite-state/group word problem) but deliberately the
*solvable* (TC0) contrast case, not S5. Also present: `even_pairs.py`, `parity_check.py`,
`modular_arithmetic.py` (all regular/TC0-or-NC1-adjacent finite automata), useful as citations
for the theory_class contrast but none is S5 composition.

**Fit to common interface:** no direct code reuse for S5 itself, but useful as a documented,
citable "this is the solvable/TC0 sibling task" comparison point for desk.json / README, and its
existence corroborates that this exact task family (finite automaton / group word problem length
generalization) is an established benchmark line.
**Verdict:** not usable as source for S5 code; valuable as related-work citation only.

## Recommendation

**Nothing is usable as-is.** `belindal/state-tracking`'s `permutation_task.py` is **usable with a
thin wrapper** for the core S_n composition arithmetic (verified correct, MIT-licensed, exactly
the Barrington/S5 word-problem construction); but the specific interface this project needs —
small *named* per-instance generator table (P1..P4) shown in the prompt, rigid
`step k: Pname -> arrangement` lines, `Instance` dataclass, independent `solve()`, `check()`,
`step_spans()`, `--selftest`/`--demo` — is not present anywhere found and **must be written**.
Liu et al.'s repo is a dead end (website only); the Chomsky-hierarchy repo is background/contrast
only. A human should decide whether to (a) wrap `permutation_task.py`'s composition logic, or
(b) keep the draft `task.py` already in this directory (written independently, same math, already
speaks the target interface) — the draft is not vetted but may be closer to done than a wrapper
would be.
