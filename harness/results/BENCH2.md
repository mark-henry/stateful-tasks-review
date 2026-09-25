# Batch 2 — knockout curve, filler recovery, redacted continuation

Each task at its batch-2 (model, depth) from plans/batch2.json. acc(CoT)/acc(no-CoT) are batch-1 values on the same instances. knockout@f = the model's own trace cut to the first f of its lines, then `Answer:` prefilled. filler = the trace replaced by the same number of ` .` tokens. filler recovery = filler − no-CoT. continue@k: gold trace through step k prefilled; r = prompt redacted (initial state + first k operators removed), p = plain prompt.

| task | model | depth | n | acc(CoT) | no-CoT | ko@0 | ko@.25 | ko@.5 | ko@.75 | filler | filler rec. | cont@k r | cont@k p | kept reasoning (ko@.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| addition* | ds | 2 | 50 | 1.00 | 0.08 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.92 | · | · | 0.02 |
| blocksworld | qwen9b | 2 | 50 | 0.92 | 0.64 | 0.62 | 0.00 | 0.44 | 0.92 | 0.66 | 0.02 | 0.76 | 0.94 | 0.04 |
| boolean_expressions | qwen9b | 12 | 50 | 0.70 | 0.34 | 0.46 | 0.48 | 0.48 | 0.54 | 0.52 | 0.18 | 1.00 | 1.00 | 0.22 |
| cruxeval | qwen9b | 6 | 50 | 0.86 | 0.44 | 0.38 | 0.38 | 0.44 | 0.86 | 0.40 | -0.04 | 0.36 | 0.72 | 0.20 |
| cup_shuffling | qwen9b | 12 | 50 | 0.98 | 0.22 | 0.26 | 0.26 | 0.34 | 0.28 | 0.14 | -0.08 | 0.30 | 0.98 | 0.14 |
| dyck | qwen9b | 8 | 50 | 0.88 | 0.20 | 0.20 | 0.02 | 0.10 | 0.00 | 0.14 | -0.06 | 0.38 | 0.88 | 0.52 |
| entity_tracking_boxes* | qwen9b | 8 | 50 | 0.82 | 0.40 | 0.40 | 0.10 | 0.24 | 0.58 | 0.36 | -0.04 | 0.28 | 0.78 | 0.50 |
| nested_arithmetic | qwen9b | 7 | 50 | 0.96 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.44 |
| random_lookup_table* | ds | 24 | 50 | 1.00 | 0.14 | 0.06 | 0.08 | 0.12 | 0.12 | 0.12 | -0.02 | 0.68 | 0.88 | 0.00 |
| s5_composition* | ds | 16 | 50 | 0.88 | 0.00 | 0.00 | 0.04 | 0.00 | 0.00 | 0.02 | 0.02 | 0.00 | 0.00 | 0.32 |
| synthetic_program_trace* | qwen9b | 32 | 50 | 0.94 | 0.08 | 0.02 | 0.02 | 0.04 | 0.14 | 0.04 | -0.04 | 0.96 | 0.96 | 0.30 |
| turing_machine | qwen9b | 2 | 50 | 0.90 | 0.04 | 0.02 | 0.00 | 0.00 | 0.00 | 0.02 | -0.02 | 0.20 | 0.92 | 0.00 |
