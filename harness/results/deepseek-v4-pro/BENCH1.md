# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/deepseek-ai/DeepSeek-V4-Pro-0813`. Temperature 0. n per cell in table. Total tokens: 665,542. Samples with hidden reasoning: 0/1298.

`*` = the published format was never prompted in its source (fine-tuned / trained from scratch) or no published CoT exists; see harness/stateful/registry.py ASTERISK.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| s5_composition* | 2 | 50 | 0.08 | 0.04 | 0.04 | 3/1 | 0.625 | 0.00 | 0.00 | 30 |
| s5_composition* | 4 | 50 | 0.00 | 0.02 | -0.02 | 0/1 | 1.000 | 0.00 | 0.00 | 54 |
| s5_composition* | 8 | 50 | 0.00 | 0.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 102 |
| s5_composition* | 12 | 50 | 0.00 | 0.06 | -0.06 | 0/3 | 0.250 | 0.00 | 0.00 | 150 |
| s5_composition* | 16 | 50 | 0.00 | 0.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 198 |
| s5_composition* | 24 | 50 | 0.00 | 0.00 | 0.00 | 0/0 | 1.000 | 0.00 | 0.00 | 294 |
| s5_composition* | 32 | 50 | 0.00 | 0.02 | -0.02 | 0/1 | 1.000 | 0.00 | 0.00 | 390 |
| threesum* | 2 | 50 | 0.42 | 0.44 | -0.02 | 13/14 | 1.000 | 0.04 | 0.00 | 406 |
| threesum* | 4 | 50 | 0.46 | 0.48 | -0.02 | 15/16 | 1.000 | 0.04 | 0.00 | 428 |
| threesum* | 8 | 50 | 0.54 | 0.50 | 0.04 | 12/10 | 0.832 | 0.08 | 0.00 | 750 |
| threesum* | 12 | 49 | 0.57 | 0.50 | 0.07 | 10/7 | 0.629 | 0.00 | 0.00 | 120 |
| threesum* | 22 | 49 | 0.53 | 0.54 | -0.01 | 12/13 | 1.000 | 0.04 | 0.00 | 457 |
| threesum* | 36 | 50 | 0.50 | 0.50 | 0.00 | 18/18 | 1.000 | 0.08 | 0.00 | 775 |
