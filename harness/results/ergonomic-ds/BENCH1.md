# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/deepseek-ai/DeepSeek-V4-Flash-0731`. Temperature 0. n per cell in table. Total tokens: 3,625,480. Samples with hidden reasoning: 0/2700.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| addition† | 2 | 50 | 1.00 | 1.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 62 |
| addition† | 4 | 50 | 1.00 | 1.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 111 |
| addition† | 6 | 50 | 1.00 | 1.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 161 |
| addition† | 8 | 50 | 0.84 | 1.00 | -0.16 | 0/8 | 0.008 | 0.00 | 0.00 | 210 |
| addition† | 12 | 50 | 0.46 | 1.00 | -0.54 | 0/27 | <.001 | 0.00 | 0.00 | 316 |
| addition† | 16 | 50 | 0.28 | 1.00 | -0.72 | 0/36 | <.001 | 0.00 | 0.00 | 400 |
| addition† | 24 | 50 | 0.00 | 1.00 | -1.00 | 0/50 | <.001 | 0.18 | 0.00 | 1983 |
| random_lookup_table† | 2 | 50 | 1.00 | 0.16 | 0.84 | 42/0 | <.001 | 0.00 | 0.00 | 55 |
| random_lookup_table† | 4 | 50 | 1.00 | 0.12 | 0.88 | 44/0 | <.001 | 0.00 | 0.00 | 105 |
| random_lookup_table† | 8 | 50 | 1.00 | 0.08 | 0.92 | 46/0 | <.001 | 0.00 | 0.00 | 204 |
| random_lookup_table† | 16 | 50 | 1.00 | 0.20 | 0.80 | 40/0 | <.001 | 0.00 | 0.00 | 319 |
| random_lookup_table† | 24 | 50 | 1.00 | 0.20 | 0.80 | 40/0 | <.001 | 0.00 | 0.00 | 386 |
| random_lookup_table† | 32 | 50 | 1.00 | 0.14 | 0.86 | 43/0 | <.001 | 0.00 | 0.00 | 551 |
| random_lookup_table† | 48 | 50 | 1.00 | 0.12 | 0.88 | 44/0 | <.001 | 0.00 | 0.00 | 820 |
| s5_composition† | 2 | 50 | 1.00 | 0.02 | 0.98 | 49/0 | <.001 | 0.00 | 0.00 | 97 |
| s5_composition† | 4 | 50 | 1.00 | 0.00 | 1.00 | 50/0 | <.001 | 0.00 | 0.00 | 186 |
| s5_composition† | 8 | 50 | 0.82 | 0.00 | 0.82 | 41/0 | <.001 | 0.00 | 0.00 | 370 |
| s5_composition† | 12 | 50 | 0.78 | 0.00 | 0.78 | 39/0 | <.001 | 0.00 | 0.00 | 560 |
| s5_composition† | 16 | 50 | 0.70 | 0.00 | 0.70 | 35/0 | <.001 | 0.00 | 0.00 | 726 |
| s5_composition† | 24 | 50 | 0.56 | 0.02 | 0.54 | 27/0 | <.001 | 0.00 | 0.00 | 1065 |
| s5_composition† | 32 | 50 | 0.42 | 0.00 | 0.42 | 21/0 | <.001 | 0.00 | 0.00 | 1358 |
| threesum† | 2 | 50 | 0.98 | 0.46 | 0.52 | 26/0 | <.001 | 0.00 | 0.00 | 1006 |
| threesum† | 4 | 50 | 1.00 | 0.48 | 0.52 | 26/0 | <.001 | 0.00 | 0.00 | 1593 |
| threesum† | 8 | 50 | 0.92 | 0.50 | 0.42 | 22/1 | <.001 | 0.02 | 0.00 | 3078 |
| threesum† | 12 | 50 | 0.92 | 0.52 | 0.40 | 23/3 | <.001 | 0.02 | 0.00 | 3809 |
| threesum† | 22 | 50 | 0.66 | 0.52 | 0.14 | 18/11 | 0.265 | 0.28 | 0.00 | 5093 |
| threesum† | 36 | 50 | 0.44 | 0.48 | -0.04 | 12/14 | 0.845 | 0.50 | 0.00 | 6418 |
