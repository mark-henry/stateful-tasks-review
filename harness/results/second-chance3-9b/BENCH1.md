# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/Qwen/Qwen3.5-9B`. Temperature 0. n per cell in table. Total tokens: 2,004,754. Samples with hidden reasoning: 0/600.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| cellular_automaton* | 1 | 50 | 1.00 | 0.02 | 0.98 | 49/0 | <.001 | 0.00 | 0.00 | 371 |
| cellular_automaton* | 2 | 50 | 0.84 | 0.02 | 0.82 | 41/0 | <.001 | 0.00 | 0.00 | 730 |
| cellular_automaton* | 3 | 50 | 0.82 | 0.00 | 0.82 | 41/0 | <.001 | 0.00 | 0.00 | 1089 |
| cellular_automaton* | 4 | 50 | 0.58 | 0.02 | 0.56 | 28/0 | <.001 | 0.00 | 0.00 | 1448 |
| cellular_automaton* | 6 | 50 | 0.46 | 0.00 | 0.46 | 23/0 | <.001 | 0.00 | 0.00 | 2166 |
| cellular_automaton* | 10 | 50 | 0.10 | 0.00 | 0.10 | 5/0 | 0.062 | 0.00 | 0.00 | 3603 |
