# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/Qwen/Qwen3.5-9B`. Temperature 0. n per cell in table. Total tokens: 2,276,461. Samples with hidden reasoning: 0/1177.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| cellular_automaton* | 1 | 27 | 0.59 | 0.02 | 0.57 | 16/0 | <.001 | 0.00 | 0.00 | 134 |
| cellular_automaton* | 2 | 50 | 0.44 | 0.02 | 0.42 | 21/0 | <.001 | 0.00 | 0.00 | 256 |
| cellular_automaton* | 4 | 50 | 0.12 | 0.02 | 0.10 | 5/0 | 0.062 | 0.00 | 0.00 | 500 |
| cellular_automaton* | 6 | 50 | 0.02 | 0.00 | 0.02 | 1/0 | 1.000 | 0.00 | 0.00 | 744 |
| cellular_automaton* | 10 | 50 | 0.02 | 0.00 | 0.02 | 1/0 | 1.000 | 0.00 | 0.00 | 1233 |
| cellular_automaton* | 14 | 50 | 0.00 | 0.02 | -0.02 | 0/1 | 1.000 | 0.00 | 0.00 | 1725 |
| hanoi | 2 | 50 | 0.98 | 0.64 | 0.34 | 18/1 | <.001 | 0.00 | 0.00 | 74 |
| hanoi | 4 | 50 | 0.98 | 0.34 | 0.64 | 32/0 | <.001 | 0.00 | 0.00 | 137 |
| hanoi | 8 | 50 | 0.96 | 0.10 | 0.86 | 43/0 | <.001 | 0.00 | 0.00 | 245 |
| hanoi | 16 | 50 | 0.86 | 0.02 | 0.84 | 42/0 | <.001 | 0.04 | 0.00 | 626 |
| hanoi | 32 | 50 | 0.42 | 0.00 | 0.42 | 21/0 | <.001 | 0.42 | 0.00 | 1721 |
| hanoi | 64 | 50 | 0.02 | 0.00 | 0.02 | 1/0 | 1.000 | 0.96 | 0.00 | 2690 |
