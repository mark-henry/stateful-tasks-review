# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth

Model: `together/deepseek-ai/DeepSeek-V4-Flash-0731`. Temperature 0. n per cell in table. Total tokens: 1,622,095. Samples with hidden reasoning: 0/1200.

`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.

Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).

| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |
|---|---|---|---|---|---|---|---|---|---|---|
| cellular_automaton* | 1 | 50 | 0.78 | 0.02 | 0.76 | 38/0 | <.001 | 0.00 | 0.00 | 123 |
| cellular_automaton* | 2 | 50 | 0.54 | 0.02 | 0.52 | 26/0 | <.001 | 0.00 | 0.00 | 269 |
| cellular_automaton* | 4 | 50 | 0.36 | 0.02 | 0.34 | 17/0 | <.001 | 0.00 | 0.00 | 557 |
| cellular_automaton* | 6 | 50 | 0.24 | 0.00 | 0.24 | 12/0 | <.001 | 0.00 | 0.00 | 954 |
| cellular_automaton* | 10 | 50 | 0.04 | 0.00 | 0.04 | 2/0 | 0.500 | 0.04 | 0.00 | 1508 |
| cellular_automaton* | 14 | 50 | 0.12 | 0.00 | 0.12 | 6/0 | 0.031 | 0.20 | 0.00 | 1934 |
| hanoi | 2 | 50 | 1.00 | 0.96 | 0.04 | 2/0 | 0.500 | 0.00 | 0.00 | 182 |
| hanoi | 4 | 50 | 1.00 | 0.82 | 0.18 | 9/0 | 0.004 | 0.00 | 0.00 | 293 |
| hanoi | 8 | 50 | 1.00 | 0.32 | 0.68 | 34/0 | <.001 | 0.00 | 0.00 | 542 |
| hanoi | 16 | 50 | 1.00 | 0.12 | 0.88 | 44/0 | <.001 | 0.00 | 0.00 | 798 |
| hanoi | 32 | 50 | 1.00 | 0.02 | 0.98 | 49/0 | <.001 | 0.00 | 0.00 | 1424 |
| hanoi | 64 | 50 | 0.68 | 0.06 | 0.62 | 32/1 | <.001 | 0.32 | 0.00 | 2321 |
