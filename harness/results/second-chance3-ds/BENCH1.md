# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/deepseek-ai/DeepSeek-V4-Flash-0731`. Temperature 0. n per cell in table. Total tokens: 942,485. Samples with hidden reasoning: 0/600.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| cellular_automaton* | 1 | 50 | 0.98 | 0.02 | 0.96 | 48/0 | <.001 | 0.00 | 0.00 | 320 |
| cellular_automaton* | 2 | 50 | 0.96 | 0.02 | 0.94 | 47/0 | <.001 | 0.00 | 0.00 | 630 |
| cellular_automaton* | 3 | 50 | 0.92 | 0.00 | 0.92 | 46/0 | <.001 | 0.00 | 0.00 | 943 |
| cellular_automaton* | 4 | 50 | 0.82 | 0.02 | 0.80 | 40/0 | <.001 | 0.00 | 0.00 | 1140 |
| cellular_automaton* | 6 | 50 | 0.66 | 0.00 | 0.66 | 33/0 | <.001 | 0.00 | 0.00 | 1589 |
| cellular_automaton* | 10 | 50 | 0.46 | 0.00 | 0.46 | 23/0 | <.001 | 0.00 | 0.00 | 2130 |
