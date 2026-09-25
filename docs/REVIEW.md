# Phase-1 sourcing review (2026-09-11)

Verdicts collated from each tasks/<slug>/SOURCING.md. "Draft" = a task.py written before the sourcing-only
amendment landed; untrusted until reviewed.

| task | verdict | best find | draft |
|---|---|---|---|
| s5_composition | thin wrapper (agent said must-write) | belindal/state-tracking, MIT, real S3/S5 generator | Y |
| random_lookup_table | must write | compositional_capabilities (bijection-only, vectorized) | |
| khop_pointer_chasing | must write | nothing | |
| cup_shuffling | must write generator | BBH data + published numbers; no generator anywhere | |
| entity_tracking_boxes | thin wrapper | Kim & Schuster repo has working generator, no CoT | |
| modn_counter | must write | DeepMind chomsky-hierarchy cycle_navigation (cited) | |
| affine_mod_p | must write | nothing | |
| addition | must write | nothing | Y (passes selftest) |
| multiplication | thin wrapper / must write | faith-and-fate scratchpad generator (algorithm only) | |
| nested_arithmetic | thin wrapper | google/BIG-bench task.py has a real generator | |
| threesum | thin wrapper | Pfau fillerTokens repo, torch-dependent, transcribe | |
| parity_coinflip | must write | Kojima zero-shot-CoT data, no scratchpad | |
| web_of_lies | data as-is, generator must write | BBH; no-CoT chance vs CoT ~perfect | |
| last_letter_concat | must write | Kojima data | |
| grid_navigation | thin wrapper (binary only); rich variant must write | BBH data | |
| dyck | thin wrapper | princeton-nlp/dyck-transformer DyckPDFA generator (torch) | |
| boolean_expressions | thin wrapper | google/BIG-bench task.py seeded generator | |
| cellular_automaton | must write | nothing | |
| turing_machine | must write | Turing-Machine-Bench is a tag system, not a TM | |
| hanoi | must write | Apple illusion-of-thinking validator | |
| blocksworld | thin wrapper | PlanBench PDDL domain | |
| river_crossing_checkers | thin wrapper | NeurometricAI simulator (licensed) | |
| cruxeval | thin wrapper | facebookresearch/cruxeval dataset + verifier | |
| synthetic_program_trace | must write | Zaremba 2014 Lua only | |

Tally: 0 usable as-is (except fixed BBH datasets), ~10 thin wrapper, ~13 must write.

## Notes for phase 2
- desk.json fields are inconsistently typed across agents (answer_space sometimes prose, tokens_per_step sometimes
  null or a dict). Normalize before building the grid. Only the cup_shuffling/entity_tracking agent used real tokenizers.
- Liu et al. 2022 "Shortcuts to Automata" never released code; only a HF dataset. The whole automaton family is hand-written.
- vendor/ is 1.2 GB, mostly dyck (571M) and blocksworld (379M). Prune before any git init.
- Two task.py drafts exist: addition, s5_composition.
