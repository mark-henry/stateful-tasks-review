# SOURCING — hanoi

Phase 1 (sourcing only, per SPEC.md AMENDMENT). No task.py, generator, or wrapper code was written.
This document inventories candidates found and recommends whether/how to build on them; the build
decision itself is left to human review.

## Apple's own release

Checked: the arXiv page (arxiv.org/abs/2506.06941), Apple's ML Research page
(machinelearning.apple.com/research/illusion-of-thinking), and the `apple` GitHub org.
**Apple released no code or dataset alongside "The Illusion of Thinking" (Shojaee, Mirzadeh et al.,
arXiv:2506.06941, June 2025).** This is typical for Apple ML papers. Nothing to vendor from the
primary source; all candidates below are third-party reproductions or unrelated Hanoi/LLM projects.

## Candidates evaluated

### 1. `taisazero/illusion_of_thinking` — VENDORED (`vendor/illusion_of_thinking/`, commit
`fc1bc18128b914b54579397c36bb88cf281ef404`, Apache-2.0)

An independent (non-Apple) reproduction of the paper's Hanoi experiment. Contents:
- `run_hanoi_experiment.py` — builds a system+user prompt (`SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`)
  asking an LLM (via Anthropic/OpenAI/vLLM clients in `shared/llm_clients.py`) to solve N-disk Hanoi
  from scratch and return the **full move list** as `moves = [[disk_id, from_peg, to_peg], ...]`. This
  is the **planning** formulation (equivalent to our `plan=True` knob), not the tracking/scratchpad
  formulation that is primary in SPEC.md — it does not ask the model to report intermediate peg state
  after each move, only to emit the final move list.
- `evaluate_hanoi.py` — a `HanoiSimulator` class that replays a `[disk, from_peg, to_peg]` move list
  against the standard Hanoi legality rules (top-disk-only, no-larger-on-smaller) and checks the
  result reaches the goal state in the minimum number of moves (`2^N - 1`). This is a solid,
  independent reference for **validating** a move sequence and could be adapted into `check()`/`solve()`
  helpers, but it does not itself compute an optimal solution (no generator/solver for the tracking
  variant) and doesn't produce per-step state strings in our required line format.
- `test_hanoi.py` — unit tests for the simulator.
- No seed-parameterized `generate()`, no `Instance`-shaped output, no step-line template, no
  `state_bits`/token accounting.

**Closeness to common interface:** far from a drop-in. Reusable as a validated reference
implementation of Hanoi's legality rules and as a design reference for the paper's original prompt
wording, but a real `generate()`/`solve()`/`format_cot()`/`step_spans()` for the SPEC's tracking
format would need to be written new. Would count as "usable with a thin wrapper" only for the
validity-checking logic inside `solve()`/`check()`, not for the generator or prompt template.

### 2. `sarahshakeri/tower-of-hanoi-reasoning-benchmark` — NOT vendored (no license)

GitHub reports no LICENSE file (default all-rights-reserved). Repo (`tower_of_hanoi.py`,
`validator.py`, `simulated_model.py`, `experiment.py`, plots) looks like a from-scratch
Apple-paper-inspired benchmark with a validator and accuracy/complexity plots, pushed 2026-09-06 (very
recent, likely low review/maturity). Noted for reference only — do not clone or copy code from it
without contacting the author for a license, since GitHub ToS does not itself grant reuse rights
absent one.

### 3. `chowdhury-mahjabin/RecurrReason` (ICLR 2026 Workshop paper "Recurrent Reasoning on Symbolic
Puzzles with Sequence Models") — NOT vendored (torch-dependent code; dataset noted only)

Repo itself is GPT-2/T5 fine-tuning scripts (`training/tower_of_hanoi/{t5,gpt2}_{scratch,pretrained}.py`)
— out of scope per SPEC's "don't install torch or run any model." No repo LICENSE file either.
However its underlying dataset, **`gmannem/RecurrReason` on HuggingFace, is CC BY 4.0** and is
structurally the closest match found to our tracking format: each Hanoi record has `puzzle_id`, `N`
(disk count), `K`, `start_state`, `goal_state`, `current_state`, `next_state`, `move`, and `num_moves`
— i.e., a BFS-optimal move-by-move state trace, train split N=1–7 and a held-out OOD test split
N=8–10. This is a **fixed dataset**, per SPEC.md's rule ("if the canonical dataset is fixed, vendor it
AND write a generator that produces fresh, non-contaminated instances of the same form") this is
exactly the shape of thing that should be vendored (a handful of example rows, for format reference)
while writing an independent seeded generator that reproduces the same schema-of-thought (state-trace
tuples) but is not restricted to the same fixed N range or dataset rows (avoiding contamination from
this specific published dataset too). Not downloaded in this pass to keep phase 1 network-light and
because no code needed it yet; flagged here so a human/build phase knows to pull a sample via
`datasets.load_dataset("gmannem/RecurrReason", "tower_of_hanoi")` if useful.

### 4. Misc other GitHub hits (surveyed, not vendored)

`NeurometricAI/illusion-of-thinking`, `saurabhg2083/illusion-of-thinking`,
`lcrosenbaum/illusion_of_thinking`, `attilammagyar/illusion-of-thinking-collapse`,
`RomainLENTZ/reasoning_model_limitation`, `fmPeretti/LLM-HANOI`, `Gajesh2007/tower-of-hanoi-llm`,
`Bora-Bastab/tower-of-hanoi-llm-benchmark`, `grankko/hanoi-redemption`, `jg1011/clanker-in-hanoi`, and
several generic "Tower of Hanoi game" repos (JS/HTML puzzle widgets, unrelated to LLM eval). None
inspected in depth beyond title/description — they read as one-off blog-post-style reproductions or
critiques of the Apple paper (several explicitly argue the paper's methodology is flawed), not
libraries offering a generator+solver+scratchpad interface. Nothing here looked more promising than
candidate #1 on a title/description scan; not worth the clone budget in a sourcing-only pass. A human
reviewer could spot-check `attilammagyar/illusion-of-thinking-collapse` (frames itself as identifying
an evaluation-framework bug in the original paper — could matter for how we define `check()`
tolerance) if that nuance becomes important later.

### BIG-bench / BBH / lm-evaluation-harness / SmartPlay

No dedicated Tower of Hanoi task found in BIG-bench, BIG-Bench-Hard, or EleutherAI's
lm-evaluation-harness task list. SmartPlay (a separate LLM-agent benchmark suite) includes a Hanoi
game environment but as an interactive RL-style env (state/action API, not a static CoT
generate/solve pair) — not evaluated further, likely a bigger lift to wrap than writing from scratch.

## Recommendation

**Nothing usable as-is; usable-with-a-thin-wrapper only for the move-legality validator** (candidate
#1's `HanoiSimulator` in `evaluate_hanoi.py`). No candidate provides a seeded `generate()` matching
SPEC.md's `Instance` shape, the required step-line template, `step_spans()`, or the tracking-mode
prompt this assignment calls the primary formulation. Tower of Hanoi's generation and optimal-solution
logic is also simple, well-specified, and easy to differentially test (recursive move generator vs. a
BFS/simulate-based solver), so the cost of writing it from scratch is low relative to adapting any of
the above. **Recommendation: nothing usable, must write** — treat `vendor/illusion_of_thinking/`
as a reference for prompt wording and a working legality-checker to cross-check against when
`solve()`/`check()` are implemented in the build phase, and treat the `gmannem/RecurrReason` HF
dataset (CC BY 4.0) as a further generator-design and non-contamination reference to fetch a sample of
if useful at build time.
