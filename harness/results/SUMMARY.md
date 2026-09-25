# Batch 1 summary — survival by model

Model roster, fixed 2026-09-16 after the first two runs and before the third: Qwen3.5-9B (thinking off), Llama-3.3-70B, DeepSeek-V4-Flash (thinking off). DeepSeek was added because Llama-3.3 (Dec 2024) underperformed the 9B with CoT and ignores published formats; it runs on the FULL grid, not only on the 70B's failures, and it is the last model for batch 1. Survival rules were fixed before any model ran and are unchanged. A task survives batch 1 if it passes on at least one model; `*` tasks pass through regardless.

Rule 1: gap ≥ 0.2 with paired McNemar p < 0.05. Rule 2: acc(CoT) ≥ 0.7 at the same depth. Working depth = shallowest passing depth; best = passing depth with the largest gap.

| task | Qwen3.5-9B: working / best (acc, gap) | Llama-3.3-70B: working / best (acc, gap) | DeepSeek-V4-Flash: working / best (acc, gap) | verdict |
|---|---|---|---|---|
| addition* | — (no gap; best gap 0.00 at d8, acc 0.00) | — (acc<0.7; best gap 0.50 at d2, acc 0.62) | d2 / d2 (1.00, 0.92) | survives (1 model) |
| blocksworld | d2 / d2 (0.92, 0.28) | d16 / d24 (0.96, 0.38) | d24 / d24 (0.90, 0.26) | survives (all models) |
| boolean_expressions | d4 / d12 (0.70, 0.36) | d4 / d4 (0.88, 0.26) | d4 / d8 (0.96, 0.40) | survives (all models) |
| cellular_automaton* | — (no gap; best gap 0.16 at d2, acc 0.16) | — (no gap; best gap 0.04 at d2, acc 0.04) | — (acc<0.7; best gap 0.54 at d2, acc 0.54) | passes through with * (format never prompted in source: no published CoT trace; format is ours) |
| cruxeval | d1 / d6 (0.86, 0.42) | — (acc<0.7; best gap 0.30 at d15, acc 0.56) | d1 / d6 (0.86, 0.36) | survives (2 models) |
| cup_shuffling | d2 / d12 (0.98, 0.76) | d2 / d3 (1.00, 0.74) | d2 / d3 (1.00, 0.78) | survives (all models) |
| dyck | d4 / d8 (0.88, 0.68) | d4 / d8 (0.82, 0.30) | d4 / d16 (0.96, 0.68) | survives (all models) |
| entity_tracking_boxes* | d4 / d8 (0.82, 0.42) | — (no gap; best gap 0.10 at d26, acc 0.84) | d8 / d36 (0.98, 0.38) | survives (2 models) |
| hanoi | — (no gap; best gap 0.08 at d5, acc 0.08) | — (no gap; best gap 0.10 at d2, acc 0.14) | — (acc<0.7; best gap 0.42 at d2, acc 0.50) | fails batch 1 |
| multiplication | — (no gap; best gap 0.04 at d30, acc 0.08) | — (no gap; best gap 0.06 at d20, acc 0.12) | · | DROPPED: Qwen3.5-9B: acc(no-CoT)=1.0 through 3x3 digits, scratchpad only hurts; no CoT gap to study |
| nested_arithmetic | d2 / d5 (1.00, 0.96) | d2 / d7 (0.96, 0.94) | d3 / d7 (0.92, 0.92) | survives (all models) |
| random_lookup_table* | — (acc<0.7; best gap 0.32 at d2, acc 0.44) | — (acc<0.7; best gap 0.40 at d4, acc 0.52) | d2 / d2 (1.00, 0.86) | survives (1 model) |
| s5_composition* | — (no gap; best gap 0.04 at d2, acc 0.06) | — (acc<0.7; best gap 0.48 at d4, acc 0.48) | d12 / d16 (0.88, 0.88) | survives (1 model) |
| synthetic_program_trace* | d12 / d32 (0.94, 0.86) | d12 / d48 (1.00, 0.96) | d20 / d72 (1.00, 1.00) | survives (all models) |
| threesum* | — (no gap; best gap -0.20 at d2, acc 0.22) | — (no gap; best gap 0.04 at d4, acc 0.52) | — (no gap; best gap -0.16 at d2, acc 0.28) | passes through with * (format never prompted in source: Pfau 2024: trained from scratch on the token format) |
| turing_machine | d2 / d2 (0.90, 0.86) | d2 / d2 (0.94, 0.92) | d2 / d8 (1.00, 1.00) | survives (all models) |
