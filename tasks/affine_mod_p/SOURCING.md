# Sourcing report: affine_mod_p

Phase: SOURCING ONLY (per SPEC.md amendment). No task.py, generator, or wrapper code has
been written for this task. `desk.json` in this directory is desk research, not derived
from running code.

## Task recap

State x in Z_p (default p=7). At step i, a fresh random pair (a_i, b_i), a_i in
{1..p-1} (nonzero so the map is invertible), b_i in {0..p-1}, is applied as
`x -> (a_i * x + b_i) mod p`. The full list of (a_i, b_i) pairs for every step is given
upfront in the prompt (not revealed step-by-step), so the model must carry only the
running state x through the scratchpad, not recall which operator applies at step k.
The relevant group is AGL(1,p), the group of invertible affine maps of Z_p under
composition — a solvable group (it is a semidirect product Z_p^* ⋉ Z_p, and both
factors are abelian, hence solvable; it also has a normal subgroup Z_p with abelian
quotient Z_p^*).

## Candidates found

### 1. `synthseq/automata` (HuggingFace dataset) — vendored to `vendor/synthseq_automata/`

- URL: https://huggingface.co/datasets/synthseq/automata
- Vendored commit (HF repo `sha`): `cf2b63456a57d26cb5261cf9fb3b9dc80bdfa69d`, last modified
  2023-02-11T23:56:10Z. Files vendored: `automata.py` (759 lines), `README.md`
  (just a license front-matter stub), `meta.json` (HF API metadata, not part of the
  dataset itself, kept for provenance).
- License: dataset card declares `license: mit`; the `automata.py` file header itself
  says "Licensed under the Apache License, Version 2.0" (Apache-2.0 boilerplate,
  copyright HuggingFace + Liu/Ash/Goel/Krishnamurthy/Zhang 2023). Both notices are
  present in the file; treat as Apache-2.0 for the code, MIT per the repo card — either
  is permissive and compatible with reuse here.
- Citation embedded in the file (verified, this is the authors' own bibkey):
  ```
  @article{liu2022transformers,
    title={Transformers learn shortcuts to automata},
    author={Liu, Bingbin and Ash, Jordan T and Goel, Surbhi and Krishnamurthy, Akshay and Zhang, Cyril},
    journal={arXiv preprint arXiv:2210.10749},
    year={2022}
  }
  ```
  This is THE dataset release for "Transformers Learn Shortcuts to Automata" (ICLR
  2023, arXiv:2210.10749). Confirms the exact author list — note the fifth author's
  surname is **Zhang**, not "Zhu" (an earlier draft of my task brief had "Zhu"; the
  paper page (arXiv, OpenReview, Microsoft Research listing) and this file's own
  citation block all agree on "Cyril Zhang"). Corrected in desk.json/README.
- Contents: a HuggingFace `datasets.GeneratorBasedBuilder` implementing several
  finite-state/group automata as numpy matrix-multiplication samplers:
  `SymmetricAutomaton` (S_n, non-solvable for n>=5), `AlternatingAutomaton` (A_n),
  `CyclicAutomaton` (Z_n under addition — inputs are shift amounts 0..n_actions-1,
  state is cumulative sum mod n), `DihedralAutomaton` (order 2n), `QuaternionAutomaton`.
  Registered under a `AUTOMATON_REGISTRY` dict with keys like `'cyclic'`, `'dihedral'`,
  `'symmetric'`, `'alternating'`, `'quaternion'` (see line ~750).
- **No AGL(1,p) / general affine class exists in this file.** `CyclicAutomaton` is the
  closest relative — it is exactly the sub-case of AGL(1,p) with a_i fixed at 1 for
  every step (pure translation by b_i, no multiplicative scaling), i.e. it tests the
  abelian group (Z_p, +), not the full solvable-but-nonabelian AGL(1,p). It is not a
  substitute for the assigned task: our task specifically wants the a_i term present
  (and varying/nonzero) because that is what makes each step a genuine bijection
  requiring composition of two operations, not just a running sum.
- Interface mismatch even where the group matches: this dataset is built for
  non-autoregressive sequence labeling (numpy arrays of one-hot states via
  `datasets.GeneratorBasedBuilder`), not for producing a natural-language prompt with an
  upfront operator list, a rigid-template CoT scratchpad, or a `solve`/`check`/
  `format_cot`/`step_spans` API. There is no prompt text, no reference solver in the
  SPEC.md sense (the "solver" is just `np.cumsum(x) % self.n`-style linear algebra
  internal to the class, not exposed as a standalone differential check), and no
  notion of `answer` as a short exact-match string.
- **Verdict for this candidate: not usable as-is or as a thin wrapper.** It confirms
  the citation and shows the paper's own operationalization of "group product" tasks,
  but reusing it would mean writing the AGL(1,p) affine-map class from scratch anyway
  (not in the file) AND writing the entire prompt/CoT/interface layer from scratch
  (not in the file). Kept vendored for citation/provenance and as a reference for how
  the original authors parameterized cyclic/dihedral/symmetric/alternating automata.

### 2. Paper's own GitHub repo (`ClaraBing/shortcut_automata`)

- URL: https://github.com/ClaraBing/shortcut_automata (found via GitHub API listing of
  the author's repos; not linked from arXiv/OpenReview directly)
- Description on GitHub: "Website for Transformers Learn Shortcuts to Automata".
  Checked the full file tree (`git/trees/HEAD?recursive=1`): it contains only the
  GitHub Pages project website (`index.html`, CSS, and figure PNGs/PDFs used in the
  paper/talk) — no code, no data, no generator. The project page itself says
  "Github: coming soon!" as of publication; that code release does not appear to have
  materialized in this repo (last pushed 2023-05-05, still website-only).
- **Not vendored** — nothing there but static HTML/images, no license file, nothing
  usable for this task.

### 3. BIG-Bench-Hard / lm-evaluation-harness / other BBH-style suites

- Searched for a modular-arithmetic or affine-group task among BBH's 23 tasks and in
  `EleutherAI/lm-evaluation-harness`. BBH's closest task is "Multi-Step Arithmetic"
  (plain nested arithmetic expressions, no group structure, no modulus) and
  "Navigate"/"Tracking Shuffled Objects" (permutation tracking — closer in spirit to
  `s5_composition` in this suite than to `affine_mod_p`). No affine-mod-p, AGL(1,p), or
  general modular-linear-recurrence task found in BBH or lm-eval-harness. Nothing
  vendored from this line of search.

### 4. General web search for "affine mod p" / "a*x+b mod" CoT benchmarks

- No hits for a purpose-built benchmark or generator matching this exact task shape
  (random per-step affine coefficients over Z_p, given upfront, tracked via CoT).
  Search surfaced unrelated results (image-encoding patents, loop-transformation
  compiler papers, quantization papers using "affine" in a different sense). Nothing
  vendored.

## Recommendation

**Nothing usable — must write from scratch.** The one directly relevant artifact (the
paper's own `synthseq/automata` HF dataset, vendored above) is useful only as citation
provenance and as a naming/parameterization reference; it does not implement AGL(1,p)
and is not shaped like the SPEC.md common interface (no prompt text, no CoT scratchpad,
no `Instance`/`generate`/`solve`/`check`/`format_cot`/`step_spans`/`KNOBS`). The
generator for `affine_mod_p` (random nonzero a_i, random b_i, per-step line format,
upfront operator list, reference solver written independently of the generator) is
straightforward and small; writing it directly against SPEC.md's interface will be
faster and cleaner than adapting the vendored numpy/HF-datasets code. This recommendation
covers only the sourcing decision — actual implementation is deferred to the phase-2
"human review decides to write code" step per the SPEC.md amendment.
