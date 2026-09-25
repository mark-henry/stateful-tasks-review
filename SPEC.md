# stateful-tasks: task contract

A library of "stateful" chain-of-thought tasks for a comparative review of which tasks force a language model
to carry load-bearing state in its CoT tokens. Every task exposes the same interface, so one harness can run
the CoT / no-CoT / filler / knockout / prompt-blinding / mistake-propagation conditions over all of them
without per-task code. Tasks never import the harness; the harness loads `tasks/<slug>/task.py` by path.

## Sourcing rule

Each task reproduces a task from the literature. Existing implementations and datasets were searched for
first, cloned (commit, license and citation recorded in `tasks/<slug>/SOURCING.md`), and wrapped. A task was
written from scratch only where nothing usable existed. Instances are procedurally generated from a seed.
Where the canonical dataset is fixed (BBH, CRUXEval), the fixed set is vendored to cross-check the solver,
and a generator produces fresh instances of the same form.

## Per-task files

    tasks/<slug>/
      task.py               the interface below; pure Python + stdlib
      README.md             source + citation, license, vendored vs written, format decision, depth semantics,
                            ANSWER_FORMAT rationale, redaction, variants, caveats
      SOURCING.md           every candidate implementation found, its commit and license, and the verdict
      desk.json             desk metrics of the task AS PUBLISHED (schema below)
      published_trace.txt   a verbatim CoT/scratchpad exemplar from the source, with a header naming it
      examples.txt          output of `python task.py --demo`
      vendor/               only the upstream files task.py reads at runtime (most tasks: none)

Tasks screened out at the desk stage have only `SOURCING.md`, `desk.json` and, where one exists,
`published_trace.txt`.

## Format

- **The gold trace reproduces the published format** of the task's primary source, as recorded in
  `published_trace.txt`. A step may span several lines if the published format does.
- Where no published trace exists (`*` tasks: cellular_automaton, entity_tracking_boxes), the format is the
  simplest one consistent with how the paper describes the task, justified in README "Format decision".
- Where the source taught the format by fine-tuning or from-scratch training (`†` tasks: addition,
  random_lookup_table, s5_composition, synthetic_program_trace, threesum), the published format is still the
  default; four of them also have an SFT-free variant (below).
- **One normalization:** `format_cot()` ends with the published closing (e.g. BBH's "So the answer is X.")
  followed by a final line `Answer: X`. `check()` reads the last `Answer:` line, falling back to the last
  non-empty line, and exact-matches after task-specific normalization.
- `Instance.prompt` is the problem statement only, in the published wording: no scratchpad instruction and no
  exemplars. The harness adds instructions and few-shot turns.

## Interface

```python
@dataclass
class Instance:
    prompt: str          # problem statement only, published wording
    steps: list[str]     # gold trace, one element per serial step (a state update); may be multi-line
    states: list[str]    # tracked state after each step, short canonical string; len == len(steps)
    answer: str          # exact-match target
    depth: int           # number of serial steps requested
    meta: dict           # operands, seed, knob values (including `format` where the task has one)

def generate(depth: int, seed: int, **knobs) -> Instance   # pure function of its arguments
def solve(inst) -> str                                  # reference solver written independently of generate()
def check(inst, completion: str) -> bool                # answer extraction as above
def format_cot(inst) -> str                             # gold trace in the instance's format + "\nAnswer: <answer>"
def exemplars(k: int, seed: int, **knobs) -> list[Instance]
    # [0] is the verbatim published exemplar when one exists; the rest are generated at a modest depth.
def redact_prompt(inst, k: int) -> str                  # prompt blinding, below
def corrupt_step(inst, k: int, seed: int) -> tuple[str, str]   # mistake injection, below (batch-3 tasks)
def step_spans(inst) -> list[tuple[int, int]]           # optional: char offsets of each step in format_cot()
ANSWER_FORMAT: str          # one line used in the no-CoT instruction, e.g. "either True or False"
DEPTHS: list[int]           # recommended sweep, from trivial to beyond where a ~9B model fails; see README
KNOBS: dict                 # name -> (default, description); depth is not a knob
REDACTION_MEANINGFUL: bool
```

The harness requires `Instance, generate, solve, check, format_cot, exemplars, ANSWER_FORMAT, DEPTHS, KNOBS`
(`harness/stateful/registry.py:REQUIRED`). `len(steps) == len(states) == depth` unless the README documents
why not (e.g. addition, where depth = digits per operand). The four tasks with an ergonomic variant accept
`format` in `exemplars(**knobs)` and return exemplars in the same format; the other tasks take `(k, seed)`
only, so their exemplars always use the task's default rendering.

## Prompt redaction (`redact_prompt`)

Return the problem statement with everything that would let a model *recompute* the state after step k
removed, and everything needed to *continue* from step k+1 kept. Replace the initial state and the
operators/inputs consumed by steps 1..k with the literal placeholder `[…]` (once per removed item, or once per
removed span). Keep the operators for steps k+1..depth, the question, and static material (rule tables,
lookup tables, program text, production rules). `redact_prompt(inst, 0) == inst.prompt`;
`redact_prompt(inst, inst.depth)` leaves only the question and static material.

Task-specific rules:
- nested_arithmetic, boolean_expressions: the expression is the operator list; sub-expressions evaluated by
  steps 1..k are replaced by `[…]` in place.
- cruxeval: the input argument is redacted for k >= 1; the code stays (the closest available analogue).
- blocksworld: the initial state description and the first k actions.
- entity_tracking_boxes: only the first k operation sentences. The initial box contents stay: the trace
  restates only the boxes each operation touched, so removing them would under-determine the answer.
- turing_machine (a tag system): the initial queue only; the production rules are static.
- addition, multiplication: redaction is not meaningful (every step needs both operands). `redact_prompt`
  returns the prompt unchanged and `REDACTION_MEANINGFUL = False`.

## Mistake injection (`corrupt_step`)

Returns `(step_text, corrupted_state)`: step k (1-based) rewritten so that the state it reports is a plausible
wrong state of the same shape, plus that state in the form of `inst.states`. The corruption differs from
`states[k-1]`, is legal-looking (same length, alphabet, structure), is deterministic in `(inst, k, seed)`,
leaves the step's action text unchanged, and works in every `format`. Minimal edits are preferred: swap two
permutation items, flip one bit, move one object, change one value by ±1, replace one stack symbol, report the
wrong branch of a check, alter one digit. In multi-line steps exactly the lines that report state are
rewritten. Implemented for the ten tasks that ran in batch 3: cellular_automaton, cup_shuffling, dyck, hanoi,
nested_arithmetic, random_lookup_table, s5_composition, synthetic_program_trace, threesum, turing_machine.

## Variants (knobs that change the trace format)

The instance distribution, `answer`, `check`, `solve` and depth semantics are identical across a task's
formats (asserted in `--selftest`); only the prompt and trace rendering change.

- `format="ergonomic"` on addition, random_lookup_table, s5_composition, threesum (default `"published"`):
  a trace designed to be computed in by a prompted model, writing the full state at every step in a
  redundant natural style, with a plain-language prompt. Examples:
  s5 `step 3: apply 51243 to 21543 -> take positions 5,1,2,4,3 of 21543 -> 3 2 1 4 5 -> 32145`;
  random_lookup_table `step 2: apply F3 to X7. F3: X7 -> X0. Now at X0.`;
  threesum one line per candidate triple with digit-wise sums and a verdict, stopping at the first hit;
  addition schoolbook right to left, `ones: 6 + 5 + 0 = 11 -> write 1, carry 1`, ..., `sum: 1009831`.
- hanoi `formulation`: `"execute"` (default) gives a legal move list and asks for the resulting
  configuration, one `move k: [d,f,t] -> [[..],[..],[..]]` line per move. `"plan"` is the batch-1
  formulation: plan the first `depth` optimal moves from a random start (Apple 2025 `moves = [...]` format).
- cellular_automaton `format`: `"cells"` (default) writes one line per cell naming its neighbours, the rule
  lookup and a running row, then a `row:` line. `"rows"` is the batch-1 format, one row per generation.

## Self-test and demo

`python task.py --selftest` (exit 0 on pass) covers 200 random (depth, seed) pairs in every format:
`solve() == answer`; `check(inst, format_cot(inst))` is true and `check(inst, "Answer: <wrong>")` is false;
`generate` is deterministic; step/state counts; `exemplars(3, 0)` renders; redaction for 20 instances at
k in {0, 1, depth//2, depth}; corruption for 20 instances at k in {1, depth//2, depth} where implemented.
`python task.py --demo` prints instances at `DEPTHS[0]` and `DEPTHS[-1]` (saved as `examples.txt`).

## desk.json

Metrics describe the task as originally published, measured on `published_trace.txt` with the real
tokenizers of google/gemma-2-2b, Meta-Llama-3-8B and Qwen2-7B. Where no published trace exists, trace-derived
fields are null and `trace_available` is false. Collated in `docs/DESK.md`.

    slug, primary_source {paper, year, url, repo}, trace_available, trace_source, format_description,
    steps_in_trace, tokens_per_step {gemma2, llama3, qwen2}, tokens_total {...}, step_delimitable,
    state_at_fixed_position, state_bits, state_bounded, state_atomic_tokens,
    theory_class ("solvable/TC0" | "NC1-hard" | "P-complete" | "unclear"), theory_citation,
    local_redundancy ("high" | "medium" | "low"), prompt_recomputable, answer_space (null = unbounded),
    error_diagnostic, knob {name, granularity, tokens_flat_across_knob}, counterfactual_defined,
    ground_truth_cost ("trivial" | "solver" | "judge"), contamination_risk ("low" | "medium" | "high"),
    paraphrasable, format_natural, interp_tractable, published_data [{paper, metric, models, values}], notes

## History

- The first draft of this spec required a rigid one-line-per-step format. It was withdrawn before
  implementation because no source uses it; some SOURCING.md files still judge candidates against it.
- hanoi's default changed from `plan` to `execute`, and cellular_automaton's from `rows` to per-cell, after
  batch 1. The later runs passed no knob, so their log sample ids collide with batch-1 ids (see
  `harness/results/tidy/README.md`). The first per-cell revision (`cells2` in the results) no longer exists
  in code; the current `cells` format is revision 3.
- `corrupt_step` was added only to the tasks that went into batch 3. multiplication was dropped after batch 1
  (no CoT gap on the 9B) and has no `redact_prompt`.
