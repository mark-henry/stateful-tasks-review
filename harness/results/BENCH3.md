# Batch 3 — step-aligned gold knockout and mistake propagation (Qwen3.5-9B)

Gold trace prefilled through step k. goldko: then `Answer:` is forced (k = 0, d/4, d/2, 3d/4, d−1). mistake: step k's reported state is corrupted (task.corrupt_step) and the model continues; scored against the ORIGINAL answer, paired with the uncorrupted continuation at the same k (`cont`). propagation = cont − mistake: the share of answers the corruption changed.

| task | depth | goldko k=0 | d/4 | d/2 | 3d/4 | d−1 | mistake@d/4 (cont) | @d/2 (cont) | @3d/4 (cont) | mean propagation |
|---|---|---|---|---|---|---|---|---|---|---|
| nested_arithmetic | 7 | 0.00 | 0.00 | 0.00 | 0.04 | 0.42 | 0.06 (1.00) | 0.04 (1.00) | 0.02 (1.00) | 0.96 |
| turing_machine | 4 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.34 (0.80) | 0.46 (0.78) | 0.16 (0.92) | 0.51 |
| synthetic_program_trace† | 32 | 0.02 | 0.04 | 0.04 | 0.14 | 1.00 | 0.26 (0.92) | 0.32 (0.96) | 0.34 (0.98) | 0.65 |
| dyck | 8 | 0.20 | 0.00 | 0.10 | 0.04 | 0.02 | 0.32 (0.86) | 0.40 (0.88) | 0.26 (0.84) | 0.53 |
| cup_shuffling | 12 | 0.26 | 0.28 | 0.36 | 0.28 | 0.40 | 0.22 (1.00) | 0.34 (0.98) | 0.24 (1.00) | 0.73 |
| s5_composition† | 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 (0.82) | 0.00 (0.88) | 0.00 (0.96) | 0.89 |
| random_lookup_table† | 24 | 0.06 | 0.10 | 0.06 | 0.12 | 0.10 | 0.00 (1.00) | 0.02 (1.00) | 0.00 (1.00) | 0.99 |
| threesum† | 8 | 0.48 | 0.46 | 0.46 | 0.46 | 0.46 | 0.54 (0.34) | 0.54 (0.42) | 0.54 (0.54) | -0.11 |
| hanoi | 16 | 0.00 | 0.06 | 0.02 | 0.00 | 0.00 | 0.66 (0.84) | 0.54 (0.84) | 0.42 (0.94) | 0.33 |
| cellular_automaton* | 3 | 0.00 | 0.06 | 0.00 | 0.00 | 0.00 | 0.00 (0.84) | 0.00 (0.90) | 0.00 (0.90) | 0.88 |
