# entity_tracking_boxes

The "boxes" entity-tracking task: a world of N boxes holds objects; a sequence of
`move` / `remove` / `put` operations is applied; the model reports the contents of
one queried box.

## Source and citation

Primary source: **Najoung Kim and Sebastian Schuster, "Entity Tracking in Language
Models", ACL 2023, pp. 3835–3855.** <https://aclanthology.org/2023.acl-long.213/>
(arXiv:2305.02363)

```
@inproceedings{kim-schuster-2023-entity,
    title = "Entity Tracking in Language Models",
    author = "Kim, Najoung and Schuster, Sebastian",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023)",
    year = "2023", publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.213", pages = "3835--3855"
}
```

Repo: <https://github.com/sebschu/entity-tracking-lms> @ `8400de051ef4ad9483cc37dee3017b37a48dafd7`,
vendored at `vendor/entity-tracking-lms/`.

**License: none.** The repo ships no LICENSE file; treat as all-rights-reserved /
academic-courtesy use and cite the paper. The authors distribute the dataset and the
model outputs in password-protected zips (password `iamnotaLM`, given in their README)
specifically to keep the eval out of future training corpora, and ask that the
uncompressed files not be committed to any repository. We honour that: the zips stay
sealed in `vendor/`, nothing is extracted, and no dataset record or prompt file is
reproduced in `task.py` or `examples.txt`.

## Vendored vs. written

Vendored (read at runtime):
- `vendor/entity-tracking-lms/data/objects_with_bnc_frequency.csv` — the paper's
  100-word, BNC-frequency-filtered object vocabulary. Read by `task.py` (knob `vocab="bnc"`).
- `vendor/entity-tracking-lms/data/objects_not_in_bnc.csv` — the paper's disjoint-vocabulary
  split (knob `vocab="disjoint"`).

Reused as a specification, reimplemented in stdlib (the original imports numpy):
- `WorldState` semantics and the operation sampler from
  `vendor/entity-tracking-lms/src/dataset_generation/generate_boxes_data.py`:
  Poisson(`expected_items_per_box`) initial occupancy clipped at `max_items_per_box`;
  uniform choice among `move` / `remove` / `put`; `move` and `remove` act on a random
  non-empty coin-flip subset of a non-empty box; `put` adds Poisson(max(1, e//2)) fresh
  objects from the unplaced pool; illegal draws are resampled. `numpy.random.poisson`
  is replaced by Knuth's sampler over `random.Random`, and every set iteration is
  sorted so that generation is deterministic.
- The surface strings are the paper's verbatim templates: `_OPERATIONS_DICT`
  ("Move the X from Box 1 to Box 2.", "Remove the X from Box 1.", "Put the X into Box 1.")
  and `WorldState._describe_box` in its `zero_shot=True` form
  ("Box 3 contains the apple and the key" / "Box 3 contains nothing"), which is the form
  used in the authors' own few-shot prompt file.

Written fresh:
- the whole `task.py` (generator, gold trace renderer, parser-based `solve()`, tolerant
  `check()`, `exemplars()`, CLI). Nothing is imported from the vendored code.

The vendored dataset itself is unusable for this harness even setting the licence aside:
it is a T5 masked-cloze corpus (`sentence_masked` / `masked_content`) with no reasoning
trace, and it is fixed rather than depth-parameterised.

## Operation vocabulary

`move`, `remove`, `put` — this is the whole published vocabulary. `WorldState.empty_box`
exists in the vendored code but its body is `raise NotImplementedError`, and `"empty"` is
absent from `_OPERATIONS_DICT`, so no published example ever contains it. It is excluded
here rather than invented.

## Format decision

`desk.json` records `trace_available: false`, and that is correct: **Kim & Schuster
publish no chain-of-thought or scratchpad variant.** Their models answer the cloze in one
shot ("… Box 2 contains ▁"), few-shot or finetuned. Per AMENDMENT 3 the format is
therefore chosen here, and the choice is "the simplest thing consistent with how the paper
describes the task":

- **One line per operation, restating only the box(es) that operation touched**, in the
  paper's own state-description wording:

  ```
  After op 8: Box 5 contains the brick and the cup and the watch.
  After op 11: Box 6 contains the coffee, Box 2 contains the shell and the tie.
  ```

  A `move` restates the source box then the destination box (the order in which the
  operation's own sentence names them); `remove` and `put` restate one box.
- **The closing line is the paper's "Statement" form** — `Box 5 contains nothing.` /
  `Box 2 contains the cup and the shell.` — which is literally the string the paper's
  models are asked to produce. Then the harness-mandated `Answer: <canonical>` line.

Rationale for restating only the affected boxes rather than the full 7-box state after
every op: (a) it is what the paper's own state machine does — an operation is defined as
touching at most two boxes; (b) a full-state restatement would make the trace O(N) per
step and let the model copy the answer off the previous line without tracking anything;
(c) with only the affected boxes restated, the queried box's contents must actually be
threaded through the subset of operations that touch it, with the unaffected boxes held
in the prompt. `Instance.states` still carries the **full** world state after each step
(`0:boat,shoe|1:-|2:cake`), so per-step blinding / probing has the complete ground truth
even though the rendered trace does not.

`Instance.prompt` is the description (published rendering, verbatim templates) plus one
query line:

```
Box 0 contains the hat and the tie, Box 1 contains the bag, … Box 6 contains the coffee. Put the shell into Box 6. Move the tie from Box 0 to Box 6. …
Question: What does Box 5 contain?
```

Two deviations from the published surface, both deliberate:
1. The paper's few-shot template wraps this in `Description:` / `Statement: Box 0 contains`
   and asks for a statement about **all** boxes. AMENDMENT 3 reserves the instruction
   wrapper for the harness, so the wrapper is dropped and the cloze is turned into an
   explicit question. Querying **one** box is not a deviation: the paper's unit of
   evaluation is exactly one box at one point in time (their test set is 7 boxes × 13
   states per scenario, and `compute_metrics.py` scores per box), and the released T5 data
   is one record per (scenario, box).
2. `Box N contains nothing` is used for empty boxes (the paper's `zero_shot=True`
   rendering, as in their `prompt_incontext.txt`) rather than `Box N is empty` (their
   T5-finetuning rendering). This makes the empty case a normal content word and keeps
   `ANSWER_FORMAT` uniform.

## Depth semantics

**`depth` = number of operations in the scenario**, i.e. `len(steps) == len(states) == depth`
exactly. The paper's default scenario is 12 operations.

The paper's own difficulty axis (Figures 2 and 5) is `numops` — the number of operations
that *changed the queried box* — which it reads out post hoc from a fixed-length scenario.
That cannot be the interface's `depth` because the harness needs one step per serial state
update. Instead, total ops is the knob and the queried box is chosen to make the two
correlate: `query_policy="most_changed"` (default) picks a box touched by the maximum
number of operations, so `meta["numops_queried_box"]` rises with depth
(≈1.3 at depth 2, 4.2 at depth 12, 10.5 at depth 36 with the paper's 7-box default).
`query_policy="uniform"` reproduces the paper's own uniform-over-boxes sampling if you
want the published distribution rather than the harder tail.

`DEPTHS = [2, 4, 8, 12, 18, 26, 36]`. Justification: 2 and 4 are near-trivial (the queried
box is touched once or twice, and the answer is often copyable from a single sentence);
12 is the paper's default scenario length; 18/26/36 push `numops` on the queried box to
roughly 6/8/10, i.e. at and beyond the point (numops ≈ 7) where the best model in the
paper, `text-davinci-003`, still got the full contents right in only ~25% of cases with
few-shot prompting and no CoT. A 7B should be comfortably saturated at 2–4 and broken by
26–36 without CoT.

## Redaction

`redact_prompt(inst, k)` (AMENDMENT 4) replaces the first `k` operation sentences with one
`[…]` each and leaves everything else — the initial box-contents description, operations
`k+1..depth`, and the question line — verbatim. `REDACTION_MEANINGFUL = True`.
`redact_prompt(inst, 0)` is the identity; `redact_prompt(inst, inst.depth)` leaves the
description, `depth` placeholders and the question. `k` outside `[0, depth]` raises
`ValueError`.

The initial description is **kept**, which is a deliberate departure from the amendment's
"redact the initial box contents and the first k operation sentences". Reason: the
published trace format restates only the boxes an operation *touched* (see "Format
decision"), so a box no operation has reached by step `k` is described nowhere in the trace
prefix. If the queried box is first touched at some step > `k`, dropping the description
would leave the question **under-determined** — unanswerable from prompt + trace by any
means — rather than merely un-recomputable, which is not what the blinding is meant to
test. Removing the operators for steps 1..`k` is already sufficient: the state after step
`k` cannot be re-derived from the redacted prompt alone, so it has to come from the trace.
(`Instance.states` still carries the full world state after every step for harnesses that
want to feed complete state instead.)

Rendered example — `generate(8, 1002)`, `k = depth // 2 = 4`:

```
Box 0 contains the coat and the file, Box 1 contains nothing, Box 2 contains the boat and the game and the ring, Box 3 contains nothing, Box 4 contains nothing, Box 5 contains the brick, Box 6 contains the milk. Put the shirt into Box 6. Put the plane into Box 3. Remove the coat from Box 0. Put the drink and the phone into Box 1. Move the game from Box 2 to Box 6. Remove the brick from Box 5. Put the card and the tissue into Box 4. Move the drink and the phone from Box 1 to Box 0.
Question: What does Box 1 contain?
```

becomes

```
Box 0 contains the coat and the file, Box 1 contains nothing, Box 2 contains the boat and the game and the ring, Box 3 contains nothing, Box 4 contains nothing, Box 5 contains the brick, Box 6 contains the milk. […] […] […] […] Move the game from Box 2 to Box 6. Remove the brick from Box 5. Put the card and the tissue into Box 4. Move the drink and the phone from Box 1 to Box 0.
Question: What does Box 1 contain?
```

## ANSWER_FORMAT rationale

```
the contents of the queried box: object names in alphabetical order separated by ", "
(for example "apple, key"), or the single word "nothing" if the box is empty
```

A box holds a *set* of 0–3 objects, so the answer is not naturally a single token. The
canonicalisation (alphabetical, comma-separated, `nothing` for empty) makes it
exact-match comparable while `check()` stays tolerant in the spirit of the authors'
`compute_metrics.py`: it accepts any order, accepts the paper's `the X and the Y`
conjunction form, accepts a full `Box 3 contains …` restatement, and treats
`nothing` / `empty` / `is empty` / `none` as the same answer. `check()` reads the last
`Answer:` line, falling back to the last non-empty line, per the amendment.

## Knobs

| knob | default | notes |
|---|---|---|
| `num_boxes` | 7 | paper default |
| `max_items_per_box` | 3 | paper default (`--max_items_per_box 3`) |
| `expected_items_per_box` | 2 | paper default (`--expected_num_items_per_box 2`) |
| `vocab` | `"bnc"` | `"disjoint"` selects the paper's held-out object names |
| `query_policy` | `"most_changed"` | also `"uniform"`, `"most_changed_nonempty"` |

Defaults reproduce `scripts/data_generation/sample_dataset_nso_exp2_max3.sh`, the script
that generated the paper's main split.

## Caveats

- **No published CoT baseline exists for this task.** Kim & Schuster measure direct-answer
  few-shot and finetuned accuracy only, so there is no literature number to compare our
  acc(CoT) against — only the no-CoT curves in their Figures 2/5/6 (GPT-3/GPT-3.5/Flan-T5,
  not Qwen). The gold trace here is our construction.
- **The empty box is a large guessable class.** With the published sampler, `remove`
  permanently deletes objects while `put` adds ~1, so boxes drift empty: with defaults,
  the answer is `nothing` for 28% of instances at depth 2 and 42% at depth 36. A model that
  always answers `nothing` therefore scores ~0.3–0.4. The paper has the same property (its
  `compute_metrics.py` tracks `is_empty` and hallucination rates for this reason), so the
  default stays faithful — but `query_policy="most_changed_nonempty"` removes the shortcut
  entirely (0 empty answers at every depth) and is the better choice if the sweep is meant
  to read as accuracy rather than accuracy-over-a-majority-baseline.
- The answer space is unbounded in principle (any ≤3-subset of 100 objects, ~166k), so
  chance is ~0 apart from the empty-class effect above. Setting `max_items_per_box=1`
  collapses it to a 101-way choice if a bounded answer space is wanted.
- `solve()` is a text parser: it re-reads `Instance.prompt` and replays the operations into
  a fresh dict of sets, sharing no code path with the generator's simulation. It will raise
  on a prompt it cannot parse rather than guessing.
- `generate(depth, seed, **knobs)` is seeded from a string containing every knob value, so
  changing any knob re-rolls the instance; instances at different knob settings are not
  paired.
- `exemplars(k, seed)` are all generated. A published few-shot exemplar does exist
  (`prompt_incontext.txt` inside the sealed dataset zip) but it is a direct-answer,
  all-boxes exemplar with no reasoning trace, and the authors ask that it not be
  redistributed; `exemplars(...)[0]` mirrors its shape (6 operations, paper-default world)
  instead of reproducing it, and is forced non-empty (`query_policy="most_changed_nonempty"`)
  so the first demonstration shows the full answer format; later exemplars use the default
  policy, so the shot set is not biased against answering `nothing`.
- `desk.json` is unchanged — its `knob` field describes the published `numops` axis, which
  is correct for the published task even though this implementation's `depth` is total
  operations (see "Depth semantics").

## Files

- `task.py` — the interface. `python3 task.py --selftest` (200 random (depth, seed) pairs), `python3 task.py --demo`.
- `examples.txt` — `--demo` output: 3 instances at depth 2 and 3 at depth 36.
- `SOURCING.md`, `desk.json` — phase-1 artefacts.
- `vendor/` — the upstream repo (zips sealed) and the paper PDF.
