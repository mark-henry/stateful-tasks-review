#!/bin/sh
# Second-chance batch-1 rerun for tasks whose format/formulation changed after batch 1 (cellular_automaton, hanoi).
# Separate log dirs and tables so the original rows stay intact for the post.
# NOTE: this passes no knob, so it runs whatever the task defaults are *now*: hanoi formulation="execute"
# (what ran) and cellular_automaton format="cells" (revision 3). The first run (logs batch1-2nd-*) used
# revision 2 of the per-cell format, which no longer exists in code; revision 3 went to batch1-2nd3-*.
set -e
cd "$(dirname "$0")/.."
SLUGS=${SLUGS:-cellular_automaton,hanoi}
caffeinate -i uv run python run_batch1.py --slugs "$SLUGS" --n 50 --max-connections 32 --log-dir results/logs/batch1-2nd-9b
STATEFUL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' caffeinate -i uv run python run_batch1.py \
  --model together/deepseek-ai/DeepSeek-V4-Flash-0731 --slugs "$SLUGS" --n 50 --max-connections 32 --log-dir results/logs/batch1-2nd-ds
uv run python collect.py --log-dir results/logs/batch1-2nd-9b --out results/second-chance-9b
uv run python collect.py --log-dir results/logs/batch1-2nd-ds --out results/second-chance-ds
echo "[second-chance] done: results/second-chance-{9b,ds}/BENCH1.md"
