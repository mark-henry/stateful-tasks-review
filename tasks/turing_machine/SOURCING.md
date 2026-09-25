# SOURCING — turing_machine

Task as assigned: simulate a small random Turing machine or register machine for `depth` steps;
transition table generated per-instance and given in the prompt; state line shows
(head state, head position, tape) per step; answer = tape (or symbol under head) after `depth`
steps. Tiny tape (e.g. 6 cells, binary). Knobs: depth, tape length, number of machine states.
Theory: P-complete in general (this is the benchmark's "maximally serial" anchor row).

## Search performed
- GitHub repo search (API): "turing machine simulator benchmark transformer", "turing machine
  simulator json states transitions dataset", "busy beaver simulator python", "register machine
  reasoning llm", "small turing machine benchmark llm chain of thought", "P-complete reasoning
  benchmark llm".
- GitHub code search (API) — unauthenticated `search/code` returned no usable results (heavily
  rate-limited/empty without a token); cross-checked candidates via repo search and HF hub instead.
- HuggingFace Hub dataset search: "turing machine", "turing machine reasoning".
- Checked DeepMind's CLRS-30 algorithmic-reasoning benchmark (`google-deepmind/clrs`) algorithm
  list directly (`clrs/_src/algorithms/`) — covers sorting/searching/graphs/geometry/DP/greedy/
  strings only; no automaton or Turing-machine-family task.
- lm-evaluation-harness and BIG-bench task directory listings grepped for "turing"/"automat" — no
  matches.
- jopetty/word-problem (S5 / group-multiplication word problems, used elsewhere in this repo for
  `s5_composition`) checked for a TM/register-machine variant — it only covers finite-group word
  problems, not Turing machines.

## Candidates vendored

### `vendor/Turing-Machine-Bench/` — Haitao999/Turing-Machine-Bench (HuggingFace dataset, Apache-2.0)
From the paper "Computational Reasoning of Large Language Models" (arXiv:2504.20771). Despite the
name, this is **not** a head/state/tape Turing machine — it's a **semi-Thue string-rewriting /
tag-system** benchmark: each sample has an `init_str`, a symbol→string rewrite `rule` dict, a fixed
`delete_count` (chars removed from the front each step, tag-system style), and `step_results`
(the string after each rewrite). Four variants ship (Latin/Greek/numeric/special-character symbol
sets); `TMBench.json` alone has samples up to `max_step: 31`. Includes a HF `dataset.py` loader
script but no reference solver/checker beyond the precomputed `step_results` (which do function as
a differential ground truth once you re-implement the one-line rewrite rule).
- Contains: generator-shaped data (symbol alphabet, rule dict, delete_count, per-step string) +
  precomputed gold trajectories. No `Instance`/`prompt`/`answer` wrapper, no head state or head
  position (there is no head at all — it's post-style rewriting, not a state-transition automaton),
  no seed-based `generate()` (the JSON is a fixed sample bank, not a generator function).
- Distance from common interface: moderate on "serial state carried in a line," far on the literal
  task shape the SPEC asks for (state-machine head/position/tape). Tag systems are themselves known
  Turing-complete/P-hard-flavored (Cocke & Minsky), so it's a legitimate P-complete-style anchor
  task, but it is a *different* task than "Turing machine with head state + position + tape," and
  the SPEC says not to substitute a different task for the assigned one.
- What would be missing to reuse as-is: the entire `Instance` wrapper, a seed-parameterized
  `generate()` (would have to write a generator from scratch anyway since the shipped JSON is
  fixed, non-extensible sample data — reusing it directly would also leave `generate(depth, seed)`
  non-deterministic-from-scratch as required, since it's a lookup into a fixed set rather than a
  true generator), and the head/state/position framing the assigned task explicitly wants.
- Useful regardless of final task shape: good citation for `desk.json.published_data` and
  `theory_class` (P-hardness of tag/rewriting systems is a close cousin of TM P-completeness), and
  a second real precedent (alongside Neary & Woods 2006) that "serial symbolic rewriting" is a
  studied LLM reasoning axis.

## Candidates found and rejected (not vendored)
- Busy Beaver simulators (`mullen0612/BusyBeaver-rewrite-in-python-`, `HYnam/Busy-Beaver`,
  `mjkid221/COMP2048-Busy-Beaver`) — small student/hobby scripts that run a TM to a halting
  condition (unbounded steps, looking for maximal tape activity), not a fixed-depth simulation with
  a serialized (state, position, tape) line per step. No LM prompt/answer framing, no differential
  solver, no seeded per-instance generator (transition tables are hardcoded famous Busy Beaver
  machines, not randomly generated). The core simulation loop (look up (state, symbol) in a
  transition table, write/move/transition) is ~10 lines and not worth adapting from these.
- `PranavAchar01/error-propagation-llm-reasoning` — studies error propagation in LLM multi-step
  reasoning generically; skimmed via GitHub search description only, no TM/register-machine content
  apparent from metadata, not fetched further given time budget.
- No CLRS, BIG-bench, lm-evaluation-harness, or BBH task matches Turing/register machines at all —
  confirmed by directly listing each project's task/algorithm directory rather than relying on
  possibly-stale search indexes.

## Recommendation
**Nothing usable as-is; must write**, with one caveat: `Turing-Machine-Bench` is worth keeping
vendored as a citation and as a cross-check reference for the "serial symbolic state machine"
theory framing, but its tag-system formalism doesn't match the assigned task's explicit
(head state, head position, tape) requirement, and its data is a fixed sample bank rather than a
`generate(depth, seed)` function, so wrapping it would violate both "don't substitute a different
task" and the determinism/procedural-generation requirements. The actual small-TM /
head-state-tape simulator (transition table as `(state, symbol) -> (write, move, new_state)`,
tape as a short binary list, one line per step) is straightforward to write from scratch and
differentially check against an independently-written reference `solve()`, per SPEC's normal
(non-phase-1) path.
