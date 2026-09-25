# SOURCING — modn_counter (cyclic group Z_N counter)

Phase 1 (sourcing-only, per SPEC.md AMENDMENT). No task.py, generator, or wrapper code was written in
this pass. This document records what was searched for, what was found, and a recommendation for the
later implementation phase.

## Target paper

**Liu, Bingbin; Ash, Jordan T.; Goel, Surbhi; Krishnamurthy, Akshay; Zhang, Cyril.**
"Transformers Learn Shortcuts to Automata." arXiv:2210.10749 (v1: 2022-10-19, v2: 2022-05-02),
published as an oral presentation at ICLR 2023. Verified live via web search (arXiv abstract page,
OpenReview forum id `De4FYqjFueZ`, ICLR 2023 slide deck). Note: the last author's surname is **Zhang**,
not "Zhu" as speculatively suggested in the task brief — corrected after checking the arXiv listing and
Microsoft Research's publication page. Affiliations: CMU, Microsoft Research NYC, UPenn.

Relevant claim (from the paper's public abstract/slides, confirmed via search — not re-derived here):
Transformers can simulate finite-state automata using far fewer layers (o(T)) than the number of
recurrent steps T, via hierarchical/shortcut reparameterizations. Solvable groups (their word problems
are in TC0) admit such shortcuts; the paper uses group word problems, including small cyclic groups, as
solvable/"easy" test cases, and non-solvable groups like S5 (whose word problem is NC1-complete/hard) as
the contrasting "should need depth" case. This framing is exactly the abelian-vs-S5 contrast the task
brief asked to cite.

### Repo search for the paper itself — NOT FOUND (code never released)

- The paper's own project page is `https://clarabing.github.io/shortcut_automata/` (Bingbin Liu / GitHub
  handle `ClaraBing`). Fetched live: the page's "Github" link is a placeholder — literally "Github: coming
  soon!" next to a joke loading gif (`meow_code.gif`). No repository URL is given.
- Confirmed via GitHub search API (`api.github.com/search/repositories?q=shortcuts+automata`, and again
  filtered `in:name,description`): the only hit is `ClaraBing/shortcut_automata`, whose description is
  "Website for Transformers Learn Shortcuts to Automata" — it is the GitHub Pages source for the
  placeholder site above, not an experiment/generator repo.
- Checked GitHub profiles of all five listed authors for any automata/shortcut/group-word-problem repo:
  `ClaraBing` (Bingbin Liu, 50 repos — nothing automata-related beyond the website), `jordantash` (0
  repos), `SurbhiGoel` (2 repos, unrelated), `akshaykr` (5 repos, unrelated — RL/bandit tooling),
  `cyrilzhang` (9 repos, unrelated, mostly old personal projects). No alternate handle guess turned up a
  repo either.
- **Conclusion: the paper's code was never publicly released.** Nothing to vendor from the primary source.

### Adjacent/follow-up work — ONE usable candidate found and vendored

Searched the follow-up line of work on state tracking that explicitly uses cyclic + S5-family group word
problems as LLM/SSM benchmarks (this is the natural place a reusable generator would live even if the
2022 paper's own code is gone):

- **Merrill, Petty, Sabharwal**, "The Illusion of State in State-Space Models," ICML 2024
  (arXiv:2404.08819). Confirmed via search (arXiv abstract, ACM DL entry, ICML proceedings page). This
  paper is the direct intellectual descendant: it shows SSMs (S4, Mamba) are also TC0-limited and fail at
  permutation-composition state tracking, again using group word problems (cyclic + symmetric families)
  as the test bed. It does not appear to be a "not solvable" test — the paper's own project/code pointer
  (`http://jpetty.org/ssm-illusion`, surfaced via search) was not independently fetched in this pass, but
  a derived dataset generator built on this line of work WAS found and IS vendored:

- **VENDORED**: `BeeGass/Group-Dataset-Generator` — GitHub repo, shallow-cloned into
  `vendor/Group-Dataset-Generator/` at commit `28d0772530679284a561a08125970bfd2b948750`
  (last commit 2025-07-13). Its `.git` directory was removed after recording the hash (only the tree is
  kept, per SPEC's "shallow clone ok, note commit hash" allowance — the hash above is the authoritative
  record).
  - **License**: no `LICENSE` file is present in the repo tree at this commit, but `README.md` (line 215)
    and `docs/HF_README.md` (front-matter `license: mit`, and line 780) both explicitly state MIT. Treat
    as MIT but flag the missing LICENSE file as a minor provenance gap if this is used later.
  - **What it contains**: a permutation-group dataset generator (`gdg/` package) with one generator module
    per group family — `gdg/generators/cyclic.py`, plus `symmetric.py`, `alternating.py`, `dihedral.py`,
    `klein.py`, `quaternion.py`, `elementary_abelian.py`, `frobenius.py`, `mathieu.py`, `psl.py`. This
    backs the companion Hugging Face dataset `BeeGass/Group-Theory-Collection` (94 permutation-group
    datasets across 10 families, split into a TC0 bucket — Symmetric/Alternating/Cyclic/Dihedral/Klein/
    Quaternion/Elementary-Abelian/Frobenius/PSL(2,p) — and an NC1 bucket — S5-S9, A5-A9, PSL, Mathieu —
    which is precisely the solvable-vs-nonsolvable contrast this task's README will cite).
  - **Cyclic-group generator specifics** (`gdg/generators/cyclic.py`): represents C_n as permutations —
    element k is the k-th power of the base n-cycle, i.e. `perm_k = [(i+k) % n for i in range(n)]`. Valid
    `n` values used in their released data: {3,4,5,6,7,8,9,10,12,15,20,25,30}. A "word" is a sequence of
    permutation-IDs (i.e. a sequence of amounts k_i drawn from 0..n-1); the label/target is the index of
    the composed permutation, i.e. exactly `sum(k_i) mod n` — this is arithmetically identical to the
    modn_counter task as specified (counter c, moves k_i, c -> (c+k_i) mod N), just phrased through a
    permutation-composition encoding rather than directly as modular addition.
  - **Data format**: `input_sequence` (space-separated permutation IDs, length 3–1024) -> `target`
    (single permutation ID). This is a supervised-learning sample, NOT a natural-language prompt and NOT
    a CoT/scratchpad. There is no per-step rendered text, no step template, no prompt wording, and no
    reference-solver text output — `_compute_composition` in `base_generator.py` just does array indexing.
  - **Fit to the common interface (SPEC.md)**: none of `Instance`, `generate`, `solve`, `check`,
    `format_cot`, `step_spans`, or `KNOBS` exist or are implied by this repo's shape. What IS directly
    reusable, if this is picked up in the implementation phase, is narrow: (a) the confirmation that
    representing cyclic-group elements/moves as `k mod N` with target `sum(k_i) mod N` is the standard
    choice in this literature, (b) the standard N values used (mostly ≤30, matching the task brief's
    "keep N small, ≤~36" instruction almost exactly), and (c) it depends on `numpy`, `datasets`,
    `huggingface_hub`, `pandas`, `pyarrow`, `numba` — none of which may be imported into `task.py` per
    SPEC's stdlib-only rule, so even the reusable arithmetic (`(i+k) % n`) would have to be retyped in
    plain Python rather than imported.
  - Not vendored: `BeeGass/Group-Theory-Collection` (the HF dataset itself) — it's a fixed numeric-ID
    dataset with no natural-language rendering, so it adds nothing beyond what the generator code already
    shows; pulling ~100k-row parquet files into `vendor/` for a fact already captured in one file would be
    dead weight.

### Other places checked, nothing usable found

- **BIG-bench** (`google/BIG-bench`): searched for a modular-arithmetic/cyclic-group/Cayley-table task.
  Found `benchmark_tasks/modified_arithmetic` (few-shot arithmetic-plus-one, e.g.
  `three_digit_addition_plus_one`) and `benchmark_tasks/simple_arithmetic` (n-digit addition) — neither is
  a group-word-problem / cyclic-counter task; both are ordinary base-10 arithmetic with no group
  structure or modulus. No task resembling a cyclic-group Cayley-table or modular-counter word problem
  was found in BIG-bench via search. Not vendored.
- **lm-evaluation-harness**: a GitHub code search scoped to this repo for modular-arithmetic/cyclic-group
  content returned only an auth-required response from the API in this sandbox (no PAT configured); no
  further follow-up was done given the "don't spend more than a few minutes" budget. Flag as unchecked
  rather than "confirmed absent."
- **Hugging Face datasets** beyond `BeeGass/Group-Theory-Collection`: not separately searched; the
  BeeGass dataset already answered "does a cyclic-group generator/dataset exist" affirmatively, so the
  search stopped there per the time-box.

## Recommendation

**Nothing usable as-is; nothing usable even with a thin wrapper for the full task.** The one vendored
repo (`BeeGass/Group-Dataset-Generator`) confirms the standard mathematical shape of this task
(k mod N moves, target = running sum mod N, N ≤ ~30) and is worth keeping in `vendor/` as a citable,
independently-authored cross-check that this is the conventional formulation in the shortcuts-to-automata
/ illusion-of-state literature — but it supplies no prompt text, no CoT/step format, no `Instance`/
`generate`/`solve`/`check` scaffolding, and its own dependencies (numpy/datasets/etc.) can't be imported
into a stdlib-only `task.py` regardless. The paper this task is nominally anchored to (Liu et al. 2022)
never released code at all. **Recommendation: nothing usable, must write task.py from scratch** in the
implementation phase, using the vendored repo only as a citation/sanity-check for the group-theoretic
framing (cyclic groups as the TC0/solvable contrast to S5), not as a source of logic to adapt.
