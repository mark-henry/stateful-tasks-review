#!/bin/sh
# SFT-free ("ergonomic", SPEC AMENDMENT 5) variants of the four failing † tasks, both models, own tables.
set -e
cd "$(dirname "$0")/.."
caffeinate -i uv run python run_batch1.py --plan plans/second-chance-daggers.json --n 50 --max-tokens-cot 8192 --max-connections 32 --log-dir results/logs/batch1-erg-9b
STATEFUL_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' caffeinate -i uv run python run_batch1.py \
  --model together/deepseek-ai/DeepSeek-V4-Flash-0731 --plan plans/second-chance-daggers.json --n 50 --max-tokens-cot 8192 --max-connections 32 --log-dir results/logs/batch1-erg-ds
uv run python collect.py --log-dir results/logs/batch1-erg-9b --out results/ergonomic-9b
uv run python collect.py --log-dir results/logs/batch1-erg-ds --out results/ergonomic-ds
echo "[ergonomic] done: results/ergonomic-{9b,ds}/BENCH1.md"
