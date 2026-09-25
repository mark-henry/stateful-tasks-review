# Second chances — as published vs variant

Cell = PASS/fail under the batch-1 rules (gap ≥ 0.2, McNemar p < 0.05, acc(CoT) ≥ 0.7) at the best passing depth (else best-gap depth): acc(CoT) / acc(no-CoT) / gap. As-published columns come from the batch-1 tables; variant columns from the second-chance runs. `†` = SFT/from-scratch format in the source; `*` = no published trace.

| task | variant | Qwen3.5-9B as published | Qwen3.5-9B variant | DeepSeek-V4-Flash as published | DeepSeek-V4-Flash variant |
|---|---|---|---|---|---|
| hanoi | execution formulation | fail d5: 0.08 / 0.00 / +0.08 | PASS d8: 0.96 / 0.10 / +0.86 | fail d2: 0.50 / 0.08 / +0.42 | PASS d32: 1.00 / 0.02 / +0.98 |
| cellular_automaton* | per-cell format (rev 2) | fail d2: 0.16 / 0.00 / +0.16 | fail d1: 0.59 / 0.02 / +0.57 | fail d2: 0.54 / 0.00 / +0.54 | PASS d1: 0.78 / 0.02 / +0.76 |
| cellular_automaton* | per-cell + named neighbours + running row (rev 3) | fail d2: 0.16 / 0.00 / +0.16 | PASS d1: 1.00 / 0.02 / +0.98 | fail d2: 0.54 / 0.00 / +0.54 | PASS d1: 0.98 / 0.02 / +0.96 |
| s5_composition† | ergonomic (SFT-free) format | fail d2: 0.06 / 0.02 / +0.04 | PASS d2: 0.92 / 0.00 / +0.92 | PASS d16: 0.88 / 0.00 / +0.88 | PASS d4: 1.00 / 0.00 / +1.00 |
| random_lookup_table† | ergonomic (SFT-free) format | fail d2: 0.44 / 0.12 / +0.32 | PASS d8: 1.00 / 0.04 / +0.96 | PASS d24: 1.00 / 0.14 / +0.86 | PASS d8: 1.00 / 0.08 / +0.92 |
| threesum† | ergonomic (SFT-free) format | fail d2: 0.22 / 0.42 / -0.20 | PASS d4: 0.98 / 0.52 / +0.46 | fail d2: 0.28 / 0.44 / -0.16 | PASS d4: 1.00 / 0.48 / +0.52 |
| addition† | ergonomic (SFT-free) format | fail d24: 0.00 / 0.00 / +0.00 | fail d4: 1.00 / 1.00 / +0.00 | PASS d2: 1.00 / 0.08 / +0.92 | fail d6: 1.00 / 1.00 / +0.00 |
