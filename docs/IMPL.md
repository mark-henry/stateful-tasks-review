# Implementation status — batch-1 contract (SPEC.md AMENDMENT 3), 2026-09-16

All 16 survivors implemented by Opus agents, each passing its own `--selftest` and `harness/check_task.py` (registry load, gold trace scores 1.0 through the harness, no-CoT path runs). Harness: `harness/HARNESS.md`.

| task | DEPTHS | answer | published exemplar in few-shot | deviations / caveats |
|---|---|---|---|---|
| addition | 2,4,6,8,12,16,24 | a single integer written with no spaces, commas, or other se | yes | Nye 2021 Fig. 2 byte-for-byte; depth = digits per operand; tokens quadratic in depth |
| blocksworld | 2,4,6,10,16,24 | the resulting state, written as a list of conditions in the  | yes | Stechly 2024 upb.txt step blocks byte-for-byte; state-TRACKING reformulation (given plan → resulting state), not planning; answer = full predicate list; ~160 tok/step, 15k-token traces at depth 24 |
| boolean_expressions | 2,3,4,6,8,12,16 | exactly one word, either True or False | yes | BBH format; generator builds a dependency PATH so depth is exact; dependence constraint reduces the chain to 1 bit of polarity (parity-like); BBH fixed set is depth 1–4 |
| cellular_automaton | 2,4,8,12,16,24 | a bit string with one character per cell, each 0 or 1, no sp | yes | no published trace; one row per generation, periodic boundary, width 8, rule 110; orbits cycle past ~depth 24 at width 8 |
| cruxeval | 1,2,4,6,9,12,15 | a Python literal, the value returned by the function | yes | CRUXEval-O [THOUGHT]/[ANSWER] format; fixed 800 set bucketed by executed-line count (sys.settrace), depth = bucket; steps rendered mechanically; contamination high |
| cup_shuffling | 2,3,5,8,12,16,24 | a multiple-choice option letter in parentheses, like (A) | yes | BBH tracking_shuffled_objects verbatim; depth = swaps, decoupled from object count; 33% floor at 3 objects |
| dyck | 4,8,16,24,32,48 | a sequence of closing brackets separated by single spaces, e | yes | BBH dyck_languages verbatim; depth = symbols; stack empties ~10% of steps (reset rescue); answer-length parity is free |
| entity_tracking_boxes | 2,4,8,12,18,26,36 | the contents of the queried box: object names in alphabetica | yes | no published CoT; one line per op restating affected boxes; sweep uses query_policy=most_changed_nonempty (published default has 28–42% 'nothing' answers) |
| hanoi | 2,5,10,20,40,80 | a peg configuration written like [[3, 2], [1], []] — three l | yes | Apple 2025 move-list format; reformulated: first k optimal moves from a random start, answer = resulting peg config (published answer IS the trace); O(n) closed-form shortcut exists |
| multiplication | 2,3,6,8,12,20,30 | a single integer (the product), written in plain decimal dig | yes | thin wrapper over Dziri 2023 generate_scratchpads.py; depth = digits_y*(digits_x+1); ~65 tok/step; 10-digit ceiling |
| nested_arithmetic | 2,3,5,7,10,14 | a single integer, possibly negative, with no commas or units | yes | BBH multistep_arithmetic_two format; depth = named sub-expression groups; magnitude gate (|x|≤1e6) biases against * at depth; naming sentence hands the model the decomposition |
| random_lookup_table | 2,4,8,16,24,32,48 | a single alphabet symbol written like X7 | dropped | Ramesh 2023 tokens, ASCII digits, one step per line, tables in prompt; bijective maps by default (non-bijective collapses: 84% start-independent at depth 32); published exemplar dropped from few-shot (different prompt shape) |
| s5_composition | 2,4,8,12,16,24,32 | a permutation of the digits 1-5 written as a single 5-digit  | yes | belindal format; state added inline per step (`step k: before -> after`, ~13 tok/step vs published 6) because published trace never writes state; composition convention stated in prompt |
| synthetic_program_trace | 6,12,20,32,48,72 | a single integer, the final value of output (e.g. 24) | yes | Nye 2021 Appendix C verbatim; depth = executed statements, loops unrolled; value cap 1000 throttles *; answer = output |
| threesum | 2,4,8,12,22,36 | either True or False | yes | Pfau 2024 Match3 serial CoT verbatim; ONE step = one candidate-triple check (~18 tok), not the 2.2 tok/step desk figure; both labels forced to exactly `depth` checks; opaque-symbol prompt, 50% floor |
| turing_machine | 2,4,8,12,20,30 | the final queue state as a bracketed, space-separated symbol | yes | TMBench format byte-for-byte; it is an m-TAG SYSTEM, not a head/tape TM; state unbounded (+1 symbol/step); desk.json corrected |

Sweep knob overrides: {'entity_tracking_boxes': {'query_policy': 'most_changed_nonempty'}}

Desk corrections made during implementation: threesum steps/tok-per-step; synthetic_program_trace tok-per-step; turing_machine state_bounded/state_bits/answer_space; nested_arithmetic knob description; blocksworld state_bits (6.0 → 6.97).
