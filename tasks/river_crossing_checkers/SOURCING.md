# river_crossing_checkers — SOURCING (phase 1)

Scope: the River Crossing and Checker Jumping puzzles from Shojaee, Mirzadeh, Alizadeh, Horton, Bengio,
Farajtabar, "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via
the Lens of Problem Complexity", arXiv:2506.06941 (v1 2025-06-07, v3 2025-11-20, NeurIPS 2025).
https://arxiv.org/abs/2506.06941 — Apple page: https://machinelearning.apple.com/research/illusion-of-thinking
Hanoi and Blocks World are sibling tasks (tasks/hanoi, tasks/blocksworld); not covered here.

Target formulation for this library (per assignment): TRACKING, not planning — a legal move sequence is
given, the model tracks the state. Nothing found implements tracking; everything below is planning
(model emits a full solution, a simulator validates it). This is the main gap in every candidate.

## Apple's own code/data: none

Checked the arXiv abstract page (v1–v3: no code/data links, only PDF/HTML/TeX), the Apple ML Research page
(only link is the arXiv entry), and github.com/apple (no repo). Two search-engine summaries claimed "Apple
publishes the full simulator suite" — false on inspection. The paper describes the simulators in prose
(App. A.2.2 / A.2.3) and prints the prompts verbatim, which is enough to reimplement exactly.

## The paper's formulations (App. A.2.2, A.2.3; Sec. 3.1; Fig. 3, Fig. 10)

Checker Jumping (A.2.2). Line of 2N+1 cells: N red 'R' on the left, one '_' in the middle, N blue 'B' on
the right. Goal: mirror it (B...B _ R...R). Moves: slide into the adjacent empty cell, or jump over
exactly one opposite-colour checker into the empty cell. No backward moves (R only rightward, B only
leftward). Minimum solution (N+1)^2 - 1 moves. Solution format `moves = [[color, from, to], ...]`,
0-indexed. Simulator validates bounds, colour at source, empty target, distance 1 (slide) or 2 (jump with
opposite colour in the middle), direction. Complexity knob: N.

River Crossing (A.2.3). N actors a1..aN and N agents A1..AN, all on the left bank with the boat. Boat
carries at most k, cannot travel empty; k=2 for N<=3, k=3 for larger N (v3 text; the QA appendix says the
paper's analysis was refined to N<6 because at N>=6 the "optimal boat capacity k=4 fundamentally changes
the problem"). Safety: an actor may not be with another agent unless their own agent is present — applies
on both banks and in the boat. Solution format: list of boat loads, alternating L->R, R->L. Solution
lengths near-linear in N (Fig. 10); 11 moves at N=3. Note the Opus & Lawsen comment (arXiv:2506.09250) and
the Rethinking repo: with k=3 the puzzle is unsolvable for N>5 (N <= 2k-1 for k in {2,3}; any N for k>=4),
so Apple's N>=6 instances were impossible. Irrelevant for a tracking task (any legal walk is fine) but
must be documented, and a generator must not promise solvability it can't deliver.

## Candidates

### 1. vendor/illusion-of-thinking — NeurometricAI, Apache-2.0, commit 8a71f82
Gradio/Ollama reproduction of all four puzzles. `puzzles.py` (433 lines, stdlib + re/json) has
`CheckerJumping(n)` and `RiverCrossing(n)` classes: Apple's system + user prompts verbatim (with the
paper's `A_1`/`a_1` underscore naming), `parse_solution` (regex on `moves = [[...]]`), `play`/`move`
validating simulators. Deviations from the paper: the CheckerJumping `move()` checks colour-at-source and
empty target but NOT direction, NOT slide/jump distance, NOT the opposite-colour-in-the-middle rule (so it
accepts illegal moves); RiverCrossing correctly enforces k = 2 if n<=3 else 3, non-empty boat, and the
safety rule on boat, departing bank and arriving bank. Both simulators are a state-update-per-move, i.e.
exactly what a tracking `solve()` needs, but the checker one would need the missing legality checks
added (or used only as the "apply" step with legality guaranteed by the generator).
Contains: prompts (yes), simulator (yes, partial for CJ), generator (only `n`; no random instances), fixed
dataset (no), CoT/scratchpad format (no), reference solver (no — validates, doesn't solve), published
numbers (no).
Distance to the common interface: no `generate(depth, seed)`, no random legal-move walk, no step-line
format, no `check()`; prompts are planning prompts. Usable as the base for an independent `solve()`
(differential check partner) and as the source of the verbatim rule text for the prompt.

### 2. vendor/Rethinking-The-Illusion-of-Thinking — Dellibarda Varela, NO LICENSE, commit d6b7625
Companion to "Rethinking the Illusion of Thinking", arXiv:2507.01231 (CSIC-UPM). Gemini-2.5-Pro-preview
experiments. Relevant pieces: `RiverCrossing/movementValidator.py` (77-line `RiverCrossingChecker(N, k,
moves)`: capacity, non-empty, on-correct-bank, safety on banks after each move — but NOT in the boat),
`RiverCrossing/RiverCrossingSolver.py` (solvability rule and prompt builder; the "solver" is the LLM call),
`CheckerJumping/CheckerJumpingSolver.py` (`simulate_moves`: colour at source, empty target, slide/jump
distance, opposite colour in middle — but NOT direction), `CheckerJumping/CheckerJumpingSteps.py`
(stepwise prompting: model emits p moves at a time from a given current board — the closest thing to a
tracking prompt found anywhere, though still planning). Result CSVs give per-run success and token
totals (see published numbers). No generator, no dataset, no scratchpad format, no reference solver.
Unlicensed: keep as evidence/reference only; do not copy code.

### 3. vendor/thinking_is_not_an_illusion — Song, Yue, Zhang, NO LICENSE, commit 1075515
Official code for "Thinking Isn't an Illusion" arXiv:2507.17699. Colab notebooks per model
(DeepSeek-R1/V3, Qwen3 thinking/non-thinking) x strategy (direct, PoT, scratchpad, think-and-execute) x
puzzle. Each notebook embeds a validator: `CheckerJumpingValidator` (bounds, colour, empty, direction,
slide/jump with opposite-colour middle — the most complete CJ validator found) and `verify_moves` for RC
(capacity 2 if n<=3 else 3, non-empty, on-bank, safety in boat / departing bank / arriving bank). No
generator, no dataset, no scratchpad format for our purposes (their "scratchpad" is a tool-augmented
prompting strategy), no solver. README tabulates River Crossing accuracy for DeepSeek-R1/V3 under tool
augmentation at N=3..13 (they did not filter unsolvable N>=6, k=3 instances either). Unlicensed.

### Not vendored
open-thought/reasoning-gym (Apache-2.0; cloned, grepped, has neither puzzle, removed); taisazero/
illusion_of_thinking (already in tasks/hanoi/vendor; Hanoi only); BIG-bench/BBH and lm-evaluation-harness
(no such tasks); HF datasets (none under this formulation); classic fixed-instance solvers on GitHub
(rtens/riverpuzzle, eckucukoglu/river-crossing-puzzle-solver, riceissa/river_crossing, an anonymous
"Jumping Checkers" gist) — wolf/goat/cabbage or 3-couple only, no parametric N, no LLM format.

## Published numbers (details in desk.json)

Apple paper: accuracy-vs-N for CJ and RC appears only as curves (Fig. 5: Claude 3.7 Sonnet thinking vs
non-thinking, DeepSeek-R1 vs V3; Fig. 6: o3-mini medium/high, DeepSeek-R1, R1-Distill-Qwen-32B, Claude 3.7
Sonnet thinking, accuracy + thinking tokens; Fig. 18 tokens only; App. A.6 QwQ-32B vs Qwen2.5-32B). No
table. Text-extractable facts: RC "performance of models mostly collapse earlier from N=3" (11-move
solution); Claude 3.7 thinking's first invalid move in RC N=3 occurs at move 4; QwQ-32B collapse ~N=2 for
CJ, ~N=3 for RC; algorithm-provision on CJ (Fig. 8, 17) does not move the collapse point; v3 restricts RC
analysis to N<6.
Rethinking repo (Gemini 2.5 Pro preview 06-05, Apple prompts, 10 runs): CJ success N=2 2/4, N=4 8/11,
N=5 6/12, N=6 7/9, N=7 3/10, N=8 2/10, N=9 1/10, N=10 2/10, N=11 1/10, N=12 1/10, N=13 0/10, total tokens
~7k (N=2) to ~22k (N>=5). RC unfiltered: N=2,k=2 10/10; N=3,k=2 8/10; N=4,k=3 4/10; N=5,k=3 1/10;
N=6–8,k=3 0/28 (unsolvable). RC solvable-only run: N=2 100%, N=3 50%, N=4 20%, N=5,k=3 0%; k=4: N=5 100%,
N=10 70%, N=20 80%, N=50 80%, N=100 90%.

## A design caveat for the human review (not a decision)

In the tracking formulation with moves given as (colour, from, to) / (boat load, direction), the state
after step k is reconstructable without carrying state: CJ cell p = colour of the last move whose `to`=p
(or '_' if the last touch was a `from`), and the empty cell is simply `from` of move k; RC bank of person
x = parity of the number of given moves listing x. That is a last-write / parity lookup over the prompt
(TC0, high local redundancy), so this task may exercise "find the relevant move in the prompt" more than
"carry state in CoT". If the reviewer wants load-bearing state, moves would have to be under-specified
(e.g. CJ: "R jumps" with position omitted; RC: boat loads named by role only), which the rule set does not
uniquely resolve without state. Flagging, not resolving.

## Recommendation

River Crossing: **nothing usable as-is; simulator usable with a thin wrapper**. The only licensed code
(NeurometricAI, Apache-2.0) gives a correct RC state-update + legality check that can back `solve()` and
the verbatim rule text; the generator (seeded random legal walk of length `depth`, with the N>5/k=3
non-solvability documented), prompt, step/state string format, `check()` and `format_cot()` must be
written. Checker Jumping: **same verdict**, with the extra note that the licensed simulator lacks the
direction and jump-middle checks and would need them added (the complete validators exist only in the two
unlicensed repos and can be used as behavioural references, not copied). Both are ~100–150 lines of
stdlib Python; the paper's prose + printed prompts fully specify the rules.
