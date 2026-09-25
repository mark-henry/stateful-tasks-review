# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/Qwen/Qwen3.5-9B`. Temperature 0. n per cell in table. Total tokens: 5,302,505. Samples with hidden reasoning: 0/2700.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| addition† | 2 | 50 | 1.00 | 1.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 64 |
| addition† | 4 | 50 | 1.00 | 1.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 116 |
| addition† | 6 | 50 | 0.94 | 1.00 | -0.06 | 0/3 | 0.250 | 0.00 | 0.00 | 169 |
| addition† | 8 | 50 | 0.72 | 0.98 | -0.26 | 0/13 | <.001 | 0.00 | 0.00 | 223 |
| addition† | 12 | 50 | 0.12 | 0.94 | -0.82 | 0/41 | <.001 | 0.02 | 0.00 | 498 |
| addition† | 16 | 50 | 0.00 | 0.92 | -0.92 | 0/46 | <.001 | 0.06 | 0.00 | 886 |
| addition† | 24 | 50 | 0.02 | 0.56 | -0.54 | 1/28 | <.001 | 0.54 | 0.00 | 4696 |
| random_lookup_table† | 2 | 50 | 1.00 | 0.14 | 0.86 | 43/0 | <.001 | 0.00 | 0.00 | 57 |
| random_lookup_table† | 4 | 50 | 1.00 | 0.08 | 0.92 | 46/0 | <.001 | 0.00 | 0.00 | 109 |
| random_lookup_table† | 8 | 50 | 1.00 | 0.04 | 0.96 | 48/0 | <.001 | 0.00 | 0.00 | 213 |
| random_lookup_table† | 16 | 50 | 0.98 | 0.16 | 0.82 | 41/0 | <.001 | 0.00 | 0.00 | 428 |
| random_lookup_table† | 24 | 50 | 1.00 | 0.16 | 0.84 | 42/0 | <.001 | 0.00 | 0.00 | 644 |
| random_lookup_table† | 32 | 50 | 0.96 | 0.14 | 0.82 | 41/0 | <.001 | 0.00 | 0.00 | 860 |
| random_lookup_table† | 48 | 50 | 0.96 | 0.08 | 0.88 | 44/0 | <.001 | 0.00 | 0.00 | 1292 |
| s5_composition† | 2 | 50 | 0.92 | 0.00 | 0.92 | 46/0 | <.001 | 0.00 | 0.00 | 123 |
| s5_composition† | 4 | 50 | 0.80 | 0.04 | 0.76 | 38/0 | <.001 | 0.00 | 0.00 | 237 |
| s5_composition† | 8 | 50 | 0.78 | 0.00 | 0.78 | 39/0 | <.001 | 0.00 | 0.00 | 478 |
| s5_composition† | 12 | 50 | 0.68 | 0.00 | 0.68 | 34/0 | <.001 | 0.00 | 0.00 | 713 |
| s5_composition† | 16 | 50 | 0.56 | 0.02 | 0.54 | 28/1 | <.001 | 0.00 | 0.00 | 949 |
| s5_composition† | 24 | 50 | 0.40 | 0.00 | 0.40 | 20/0 | <.001 | 0.00 | 0.00 | 1424 |
| s5_composition† | 32 | 50 | 0.36 | 0.00 | 0.36 | 18/0 | <.001 | 0.00 | 0.00 | 1974 |
| threesum† | 2 | 50 | 0.92 | 0.46 | 0.46 | 23/0 | <.001 | 0.00 | 0.00 | 1242 |
| threesum† | 4 | 50 | 0.98 | 0.52 | 0.46 | 24/1 | <.001 | 0.00 | 0.00 | 2241 |
| threesum† | 8 | 50 | 0.92 | 0.54 | 0.38 | 21/2 | <.001 | 0.04 | 0.00 | 4949 |
| threesum† | 12 | 50 | 0.30 | 0.50 | -0.20 | 3/13 | 0.021 | 0.70 | 0.00 | 7708 |
| threesum† | 22 | 50 | 0.18 | 0.46 | -0.28 | 6/20 | 0.009 | 0.80 | 0.00 | 7456 |
| threesum† | 36 | 50 | 0.16 | 0.50 | -0.34 | 4/21 | <.001 | 0.80 | 0.00 | 7535 |
