# SOURCING.md — threesum

Phase: **sourcing only**, per SPEC.md AMENDMENT. No task.py / generator / wrapper code has been written.

## Paper

Pfau, Merrill & Bowman (2024), "Let's Think Dot by Dot: Hidden Computation in Transformer
Language Models," arXiv:2404.15758. https://arxiv.org/abs/2404.15758

## Candidates found

### 1. `vendor/fillerTokens` — JacobPfau/fillerTokens (USE THIS ONE)
- URL: https://github.com/JacobPfau/fillerTokens
- Owner: Jacob Pfau — first author of the paper. README links directly to arXiv:2404.15758
  and gives the paper's own BibTeX. This is the paper's own repo.
- Cloned shallow (`--depth 1`) at commit **cb39af6458b7476ba07f25e89a9c8fd339c1e229**
  (2024-04-27 15:13:26 -0700, i.e. 3 days after the arXiv v1 submission).
- License: **none found.** No LICENSE file in the repo, GitHub's license API returns 404
  ("Not Found" — GitHub could not detect a license). Code is publicly visible on GitHub but
  not under an OSI license; treat as "all rights reserved, viewable for research reference"
  and do not redistribute wholesale. Cite the repo and paper; any wrapper we write later
  should be original code that merely reads the same algorithm, not a verbatim copy of
  Pfau's source, to stay safe.
- Contents relevant to `threesum`:
  - `src/match3.py` — the **Match3** class = the actual 3SUM generator used in the paper.
    - `get_instance`: draws `length` vectors of dimension `dimension` from `Z_mod` (default
      `dimension=3, mod=10, length=10` per `scripts/data_match3.py` argparse defaults, though
      the paper's headline experiments use `length` in [6,14] and `dimension` in [1,6]).
    - `solve()`: an O(n^3)-ish independent brute-force checker (nested loop over t, then over
      remaining pairs) — usable as a reference oracle, though it returns bool only for the
      first triple's existence check, not a running trace.
    - `get_true_instance` / `get_corrupted_instance`: construct guaranteed-positive vs.
      probably-negative instances by planting a validated zero-sum triple (`a + b + (-a-b mod
      m) = 0`) then shuffling in noise, with `corruption_rate` controlling how many entries of
      the planted triple get overwritten by random digits (geometric distribution, capped at
      3) before a final re-check that the corruption didn't accidentally create a real match.
    - `serial_solve()`: this **is** the paper's "instance-adaptive"/serial instructive-CoT
      trace generator — it only looks at the first coordinate (`first_digit_inputs =
      inputs[:,0]`) to find a matching triple by index, then reports: (a) the three positional
      indices, each followed by `-` (e.g. `"2-"`), (b) one randomly-chosen coordinate's raw
      digit for each of the three matched vectors, (c) the coordinate-wise intermediate sums
      mod 10 for each remaining dimension in turn, stopping early at the first nonzero sum
      (i.e. it lazily proves non-match on later dimensions), and finally (d) `'True'`/`'False'`.
      This is a genuine multi-step, position-then-content, running-state trace — a strong
      candidate for the `steps`/`states` design once phase 2 begins.
    - `probabilistic_dense_solve()`: a "parallel" CoT variant that emits one label per pair
      (t, t') visited, either `'-'+found_index` (match) or `'F'+digit` (no match, filler
      digit) — this is the O(n^2) "reduce 3SUM to a sequence of 2SUMs" CoT described in the
      paper (pairwise sums table, e.g. `"AB 70 AC 27 ..."`).
    - String builders: `no_filler_serial`, `dot_filler_serial`, `serial_cot` (serial/
      instance-adaptive format), and `rand_cot`, `dot_filler_parallel`, `no_filler_parallel`
      (parallel/pairwise format) — these literally ARE the three training conditions
      (no-CoT / filler / full-CoT) rendered as strings, for both CoT styles.
    - Filler length is fixed at `filler_length = length**2` dots (see
      `GenerateMatch3Dataset`), inserted as `'. '*num_filler` between the `P` prompt-end
      marker and the `A` answer marker — i.e. **filler count = n², one token per dot,
      independent of dimension**, matching the paper's stated "n² intermediate tokens."
      `cot_rate` / `no_filler_rate` (default 0.5 / 0) control the train-time mixture; the
      remainder is filler. This confirms "uniform 50/50 CoT/filler" was the default recipe.
  - `scripts/data_match3.py` — CLI wrapper exposing all of Match3's knobs
    (`--dimension` default 3, `--mod` default 10, `--length` default 10,
    `--true_instance_rate` default 0.5, `--cot_rate` default 0.5, `--no_filler_rate`
    default 0, `--corruption_rate` default 4/3, `--cot_to_string` in {rand_cot, serial}).
  - `scripts/run_match3.py`, `src/train_match3.py` — training loop against a from-scratch
    HF `LlamaForCausalLM`; `misc/llama_d384l4h6.json` gives the exact architecture used:
    `hidden_size=384, num_hidden_layers=4, num_attention_heads=6, intermediate_size=1536,
    vocab_size=32000` → this is the paper's ~34M-parameter model (matches Sec. 4/5 text).
  - `src/match2.py` — **Match2** class = the 2SUM-Transform generator (`get_true_instance`,
    `get_corrupted_instance`, `solve`), plus a "lookup" per-digit additive transform applied
    to each input tuple (`lookup_transform`/`random_lookup_params`) whose inverse-key is
    appended only at the very end of the sequence (`transform_params`) — matching the paper's
    description of a permutation/offset revealed only in the final tokens so the network can't
    front-load the 2SUM computation. String builders `b10_no_filler_string`,
    `b10_repeat_filler_string` (filler = repeated `'A'` token, one per input row, i.e. filler
    count = length, NOT n² here — a different, per-task filler count from Match3), and
    `b10_basic_string` (full CoT: decoded transform params, then each pair's decoded values
    and their sum, i.e. explicit intermediate 2SUM arithmetic).
  - `scripts/data_match2.py` is referenced by README example commands but is **not present**
    in this commit's `scripts/` directory (only `run_match2.py`, `data_match3.py`,
    `run_match3.py`, `decode_filler.py`, `filter_serial.py` exist) — a minor
    incompleteness/drift between README and this exact commit; the generator logic itself
    (`src/match2.py`) is complete and importable directly, so this is not blocking.
  - No published dataset/checkpoints are vendored in the repo (`data/tmp.csv` and
    `output_dir/tmp.csv` are empty placeholder stubs, not real data).
  - No pytest/unit tests; no fixed "canonical" benchmark instances — this is a
    parametric generator (procedurally generated, matching SPEC.md's stated preference).

### 2. `arbdwj/filler_tokens` — NOT vendored separately
- URL: https://github.com/arbdwj/filler_tokens
- GitHub API confirms this is a **fork** (`fork: true`, `source.full_name` /
  `parent.full_name` both = `JacobPfau/fillerTokens`). No commits ahead were checked in
  detail, but as a plain fork of the exact repo above it adds no independent value; not
  cloned separately to avoid duplicate vendor content per SPEC.md's "shallow-clone every
  plausibly useful candidate" — a fork of an already-vendored repo is not a distinct
  candidate.

### Other searches (no additional usable candidates)
- BIG-bench / BBH: 3SUM and 2SUM-Transform are not BIG-bench tasks (they are from-scratch
  synthetic tasks trained into small transformers, not eval-only prompts for pretrained
  LLMs) — nothing to check there.
- lm-evaluation-harness: no match (searched for "3sum", "match3", "filler token" task
  names; this paper's tasks are not part of that harness — they require training, not
  just prompting).
- HuggingFace Hub / HF datasets: paper's HF papers page (huggingface.co/papers/2404.15758)
  exists but links no dataset or model artifacts.
- No follow-up/re-implementation repos of the 3SUM/2SUM-Transform generators specifically
  were found in general web search (only the filler-token *concept* has since been reused
  by other pause/filler-token papers, which is out of scope for vendoring this task).

## Closeness to the common interface (SPEC.md)

The vendored `Match3`/`match3.py` code is much closer to usable than a typical from-scratch
situation:
- **Generator present**: yes (`Match3.get_true_instance`/`get_corrupted_instance`), fully
  parametric over `(dimension, mod, length, corruption_rate)`, seedable via
  `np.random.default_rng()` (currently unseeded — a thin wrapper would need to thread a
  `seed` argument into the RNG construction to satisfy SPEC's determinism requirement).
- **Fixed dataset**: no — good, matches SPEC's stated preference for procedural generation.
- **Instructive-CoT scratchpad format**: yes, and it is genuinely stateful — `serial_solve()`
  is a step-by-step, position-then-content trace with a running "which pairs already
  checked" / "current best candidate" implicit state, which is exactly the shape SPEC.md
  wants for `steps`/`states`. It would need to be **rewritten**, not reused verbatim, to
  emit `Instance.steps`/`Instance.states` as separate rigid-template lines (SPEC wants one
  step per line, state at a predictable position) rather than a flat token list — Pfau's
  `labels` list is a flat sequence of tokens, not pre-split into full lines. This is
  "extract the *algorithm*, rewrite the *rendering*," which SPEC's "wrapped in the common
  interface" language anticipates.
- **Reference solver independent of generate()**: `Match3.solve()` exists and is a genuine
  brute-force check independent of the CoT-construction code path (`serial_solve`/
  `probabilistic_dense_solve`), satisfying SPEC's differential-check requirement in spirit —
  though its return value is a single bool for the whole instance, so a phase-2 wrapper
  would still write its own small O(n^2)/O(n^3) solve() per SPEC's letter ("solve() must be
  written independently from generate()") rather than import Pfau's `solve()` directly, to
  keep the independence guarantee airtight and stdlib-only (Pfau's code needs numpy/torch).
- **Filler-token protocol**: fully documented and exact (see README.md draft below) —
  this satisfies the task's role as the calibration row for the suite's filler-token metric.
- **Published numbers**: paper Figure 2 / Table 1 numbers extracted below and duplicated
  into desk.json `published_data`.
- **Stdlib-only constraint**: Pfau's code is numpy+torch+transformers; none of that can be
  imported at runtime per SPEC ("pure python + stdlib only, no torch/transformers"). A
  phase-2 wrapper must reimplement `Match3`'s generation/solve logic in bare Python (this
  is mechanical — it's just modular arithmetic over small integer tuples — but it is real
  work, not a drop-in import).

## Recommendation

**Usable with a thin wrapper** — more specifically, "usable as an algorithmic reference to
reimplement against," since the stdlib-only / no-torch constraint rules out literally
importing `vendor/fillerTokens/src/match3.py` (it depends on numpy/torch/transformers) or
`src/match2.py` (numpy/torch). The generator logic itself (index arithmetic, corruption
scheme, serial-CoT trace shape, filler-count formula, model/data hyperparameters) is fully
specified and should be transcribed faithfully in pure Python rather than re-derived from
the paper's prose alone. No task.py has been written in this phase per the AMENDMENT; this
recommendation is for the human review that decides whether/how phase 2 proceeds.

## Filler-token protocol (exact, for the calibration row)

- **3SUM (`src/match3.py`)**: filler symbol is a single space-separated dot character
  `'.'`, repeated `filler_length = length**2` times (`'. '*num_filler`), inserted between
  the `P` prompt/CoT-boundary marker and the `A` answer marker. Count is a pure function of
  sequence length `n` (`n²`), independent of `dimension`/`mod`. Example rendered instance
  (parallel CoT paper example, quoted from the arXiv text via WebFetch extraction):
  `"A05 B75 C22 D13 : . . . . . . . . . . . . ANS True"`.
- **2SUM-Transform (`src/match2.py`)**: filler symbol is the token `'A'`, one per input row
  (i.e. filler count = `length`, not `length**2` — a different ratio from 3SUM;
  `b10_repeat_filler_string`), inserted immediately after each row's digits and again once
  after the transform-params ("A" for sum supervision), before the final `L<label>` answer
  token.
- **Training mixture**: `GenerateMatch3Dataset`/`GenerateMatch2Dataset` draw each training
  example's rendering independently per-example from a 3-way categorical
  `{cot_rate, filler_rate, no_filler_rate}` (defaults `cot_rate=0.5, no_filler_rate=0` ⇒
  `filler_rate=0.5`), i.e. a dense, per-example 50/50 mixture of full-CoT and filler-token
  supervision at train time (not a curriculum/schedule) — matching the paper's statement
  that filler-token learning "requires specific, dense supervision to converge."

## Published results (extracted from arXiv:2404.15758 via WebFetch; verify against PDF
tables before final publication — this was a text-extraction pass, not a manual table read)

- Model: from-scratch `LlamaForCausalLM`, `hidden_size=384, num_hidden_layers=4,
  num_attention_heads=6, intermediate_size=1536` (config file
  `vendor/fillerTokens/misc/llama_d384l4h6.json`) — approx. 34M parameters, per paper text.
- **3SUM** (length-12, dimension-3, Figure 2): no-filler/immediate-answer accuracy ≈ 66%;
  filler-token accuracy = 100%.
- **2SUM-Transform** (Table 1): Chain-of-thought 95.1%; Filler 93.6%; No intermediate
  tokens 78.7%; majority-class baseline 63%.
- Additional note: on easier length-10, dimension-1 data (solvable without filler),
  filler-token models reached 100% accuracy with only ~2% of the original filler-token
  count, illustrating filler-token redundancy once capacity is sufficient.
- Dimension scaling (Figure 5, length-8): the no-filler/filler accuracy gap widens starting
  around dimension-6.

These numbers are carried into `desk.json`'s `published_data` field.
