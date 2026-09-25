# SOURCING — addition

Phase 1 sourcing search for a usable existing implementation/dataset of multi-digit addition
with a carry scratchpad, per SPEC.md's AMENDMENT. Candidates checked:

## Nye et al. 2021, "Show Your Work: Scratchpads for Intermediate Computation with Language Models" (Google Research)

- **What it is**: the paper that introduces the scratchpad technique, and uses multi-digit
  addition with carry-tracking as one of its illustrative tasks.
- **What was checked**: searched for an official code/data release accompanying the paper
  (arXiv page, associated Google Research GitHub orgs, follow-up citations pointing to a
  release). No public repository or dataset release for the addition-scratchpad task was
  found.
- **Contents if it existed**: would presumably have been a generator + fixed dataset of
  addition problems with gold scratchpads, plus whatever prompt template they used.
- **Fit to common interface**: N/A — nothing found to assess.
- **Verdict**: nothing usable found.

## Lanham et al. 2023, "Measuring Faithfulness in Chain-of-Thought Reasoning" (Anthropic)

- **What it is**: a paper studying CoT faithfulness (early answering, adding mistakes,
  paraphrasing, filler tokens) across a range of reasoning benchmarks.
- **Correction (verified directly from the paper's text, see desk.json notes)**: Lanham et al.
  2023 *does* use addition as an explicit task — Section 3.2 "Addition Tasks" evaluates
  synthetic multi-operand addition problems (2/4/8/16 operands, 2-3 digits each) and Table 5
  shows worked examples (e.g. "264 + 964 = = 264 + (900 + 64) = ... = 1228"). This is
  free-form associative-regrouping CoT, not a digit-column carry scratchpad, and no
  generator/dataset artifact was released alongside it — the paper's contribution is a
  measurement methodology (does the model give the same answer with/without CoT, across model
  sizes), not a reusable addition-instance generator or a carry-tracking gold trace.
- **What was checked**: searched for a released addition-generator or scratchpad-format
  artifact tied to this paper. None found — the paper's contribution is a measurement
  methodology applied to (in this case, synthetic) addition problems, not a new addition
  dataset/generator release.
- **Fit to common interface**: N/A — nothing found to assess.
- **Verdict**: nothing usable found; cited in README/desk.json for methodology/motivation only.

## Dziri et al. 2023, "Faith and Fate: Limits of Transformers on Compositionality"

- **Repo**: github.com/nouhadziri/faith-and-fate
- **License**: MIT
- **Commit checked**: `1e90edb54b4ed0fa150259a72c994b0fee90d388`
- **What it contains**: `dynamic_programming/`, `logic_puzzle/`, `multiplication/`, and
  `data/*.zip`. The `multiplication/` directory has a generator for multi-digit multiplication
  problems (with a columnar/partial-product compositional-graph framing) and associated data.
- **What was checked**: the full repo tree was enumerated looking for an `addition/`
  directory or any addition-specific generator/dataset file. None exists — multiplication is
  the only arithmetic task implemented in this repo.
- **Fit to common interface if we *had* found an addition generator here**: the
  `multiplication/` generator (as an analog) produces raw problem/answer pairs and a
  compositional "computation graph" representation, not a rigid line-per-step scratchpad
  text format matching this suite's `Instance`/`format_cot`/`step_spans` contract, so even a
  hypothetical addition counterpart would likely have needed a wrapper to (a) reformat its
  intermediate representation into one-line-per-step text, (b) add a `solve()` independent of
  the generator's own step logic, and (c) add `check()`/`step_spans()`. Moot here since no
  addition module exists at all.
- **Verdict**: nothing usable for addition (multiplication only); not applicable to this
  slug.

## BIG-Bench-Hard / lm-evaluation-harness / HuggingFace datasets (spot check)

- No BBH task matches "addition with carry scratchpad" (BBH's arithmetic-adjacent tasks are
  things like `object_counting`, `multistep_arithmetic_two`, `navigate` — not a columnar
  addition-with-explicit-carry-state format). `multistep_arithmetic_two` in particular tests
  order-of-operations arithmetic expressions, not digit-by-digit carry tracking, and has no
  scratchpad/state-per-step format matching this suite's interface.
- No HuggingFace dataset was found specifically packaging "grade-school column addition with
  carry scratchpad, one line per digit" in the common interface's shape; general arithmetic
  QA datasets exist but are answer-only, with no gold per-step carry-tracking CoT.
- These were spot-checked, not exhaustively cloned, since the three cited papers were the
  primary sourcing targets per the original brief and none pointed toward a reusable BBH/HF
  artifact for this exact task shape.

## Note on task.py

`task.py` (along with the rigid one-line-per-step `examples.txt` it generates) is a separate,
prior-phase artifact: an implementation of this task in *our own* reformatted scratchpad
notation, written before the sourcing-only phase ordering was declared for this slug. It is
unrelated to the desk-metrics review in `desk.json` — those metrics describe the task AS
PUBLISHED in the source literature (see `published_trace.txt`), not task.py's own format, and
task.py's self-test/demo runs are not evidence for or inputs to any of the published-source
findings above. Whether task.py's own format is adopted for actual implementation is a
separate decision for later human review.

## Recommendation

**Nothing usable, must write.** All three cited sources (Nye et al. 2021, Lanham et al. 2023,
Dziri et al. 2023/faith-and-fate) were checked and none has a released addition-with-carry
generator or dataset; a spot check of BBH/lm-evaluation-harness/HuggingFace turned up nothing
matching this task's specific columnar carry-scratchpad shape either. (Nye et al. 2021 does
provide a published worked example of the carry-scratchpad format itself, extracted verbatim
into `published_trace.txt` and analyzed in `desk.json` — but that is a single figure example,
not a generator or dataset to vendor.)
