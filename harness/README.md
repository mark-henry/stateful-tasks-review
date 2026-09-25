# Harness runbook

The harness runs the tasks in `../tasks/` through every condition of the review. It is built on
[Inspect AI](https://inspect.aisi.org.uk/) and loads each `tasks/<slug>/task.py` by path; the task contract is
`../SPEC.md`. Every command below runs from this directory.

## Layout

    stateful/registry.py    load tasks by path; SURVIVORS, DROPPED, markers (DAGGER †, ASTERISK *), SWEEP_KNOBS
    stateful/prompts.py     prompt assembly for the cot and nocot conditions
    stateful/batch1.py      @task batch1: acc(CoT) and acc(no-CoT) over depth
    stateful/batch2.py      @task batch2: knockout, filler, continue, goldko, mistake (batches 2 and 3)
    stateful/traces.py      reads each instance's own batch-1 CoT trace from the logs (knockout and filler need it)
    run_batch1.py           batch-1 driver (eval_set, so reruns resume); --plan restricts slugs/depths/n/knobs
    run_batch2.py           batch-2 driver over plans/batch2.json
    run_batch3.py           batch-3 driver over plans/batch3.json
    collect.py              batch-1 logs -> batch1.csv + BENCH1.md (one pair per set of log dirs)
    summarize.py            per-model batch1.csv/BENCH1.md -> results/SUMMARY.md (survival verdicts)
    summarize2.py           -> results/SECOND_CHANCE.md (as published vs variant formats)
    make_batch2_plan.py     picks (model, depth) per task for batch 2 -> plans/batch2.json
    collect2.py, collect3.py  batch-2 / batch-3 logs -> BENCH2.md + batch2.csv / BENCH3.md
    tidy.py                 every log -> chart-ready CSVs in results/tidy/ (see results/tidy/README.md)
    density.py              load-bearing token density estimate -> results/DENSITY.md, results/tidy/density.csv
    check_task.py           contract check for one task or --all (mock model, no key needed)
    scripts/                shell wrappers for the second-chance runs (macOS: they use caffeinate)
    tests/stub_task/        a minimal contract-conforming task for smoke tests
    post/                   chart and table scripts for the blog post

## Setup

    uv sync
    cp .env.example .env        # then fill in TOGETHER_API_KEY

The drivers read `.env` before Inspect starts. `STATEFUL_MODEL` is the default model, and
`STATEFUL_EXTRA_BODY` is merged into every request. For the Qwen3.5 and DeepSeek-V4 hybrid-thinking models it
turns thinking off. The scorer records `hidden_reasoning` per sample; it was 0 on every run. Set it empty for
models without a thinking mode.

Temperature is 0 everywhere and generations are cached (`generate(cache=True)`), so rerunning a finished
command costs nothing.

## Conditions

- **cot / nocot** (batch 1): both arms get the same three few-shot instances. In cot the assistant turns are
  the gold trace in the task's format plus `Answer: X`. In nocot they are `Answer: X` only, the system prompt
  forbids reasoning, and `max_tokens` is sized to the answer. Chat APIs allow no prefill, so no-CoT is
  enforced by instruction; `leaked_reasoning` (more than one non-empty line) is logged and reported.
  `truncated` flags CoT completions that hit `max_tokens` (default: sized from the longest gold trace, floor 1024).
- **knockout@f** (batch 2): the model's own batch-1 trace cut to its first fraction f of lines, then
  `Answer:` prefilled. **filler**: the trace replaced by the same number of ` .` tokens, then `Answer:`.
  **continue@k**: the gold trace through step k is prefilled and the model continues, with the prompt either
  plain or redacted by `task.redact_prompt(inst, k)`.
- **goldko@k** (batch 3): gold trace through step k, then `Answer:` forced, for k = 0, d/4, d/2, 3d/4, d−1.
  **mistake@k**: step k's reported state is corrupted by `task.corrupt_step` and the model continues. The
  result is scored against the original answer and paired with the uncorrupted continuation at the same k
  (propagation = cont − mistake).

Batches 2 and 3 prefill a partial assistant turn, so they depend on the provider honouring
`continue_final_message` (Together does). Their model strings are fixed per tag in
`stateful/traces.py:MODEL_LOGS` (`qwen9b`, `ds`, `dspro`, `llama70b`). knockout and filler also read the
batch-1 traces from the log dirs listed there, so they need the batch-1 logs (or a fresh batch-1 run into
those dirs). goldko, continue and mistake need only the task code.

## Rerunning the review

The log dirs under `results/logs/` are not in the repository (about 300 MB of Inspect `.eval` files). Every
table in `results/` was built from them. The commands below write to the same dirs the published run used;
the header of each log confirms its flags. `--max-connections` and the `caffeinate` wrapper used on the
original runs are left out.

**Batch 1: depth sweep, four models.**

    uv run python run_batch1.py --log-dir results/logs/batch1                                         # Qwen3.5-9B (.env default)
    uv run python run_batch1.py --plan plans/batch1b-qwen9b.json --log-dir results/logs/batch1b-9b    # extra depths, n=150 cells
    STATEFUL_EXTRA_BODY= uv run python run_batch1.py --model together/meta-llama/Llama-3.3-70B-Instruct-Turbo --log-dir results/logs/batch1-70b
    uv run python run_batch1.py --model together/deepseek-ai/DeepSeek-V4-Flash-0731 --log-dir results/logs/batch1-ds
    uv run python run_batch1.py --model together/deepseek-ai/DeepSeek-V4-Flash-0731 --slugs hanoi,cellular_automaton,s5_composition,threesum --max-tokens-cot 8192 --log-dir results/logs/batch1-ds-cap
    uv run python run_batch1.py --model together/deepseek-ai/DeepSeek-V4-Pro-0813 --slugs s5_composition,threesum --max-tokens-cot 8192 --log-dir results/logs/batch1-dspro

    uv run python collect.py --log-dir results/logs/batch1 --log-dir results/logs/batch1b-9b --out results/qwen35-9b
    uv run python collect.py --log-dir results/logs/batch1-70b --out results/llama33-70b
    uv run python collect.py --log-dir results/logs/batch1-ds --log-dir results/logs/batch1-ds-cap --out results/deepseek-v4-flash
    uv run python collect.py --log-dir results/logs/batch1-dspro --out results/deepseek-v4-pro
    uv run python summarize.py                     # -> results/SUMMARY.md

`collect.py` with no arguments writes `results/batch1.csv` and `results/BENCH1.md`: the 9B table before the
`batch1b` extension. Later log dirs override earlier ones on the same sample id. multiplication was dropped
after the 9B run (`registry.DROPPED`, no CoT gap), so the default slug list now skips it; name it in
`--slugs` to include it. entity_tracking_boxes always runs with `query_policy=most_changed_nonempty`
(`registry.SWEEP_KNOBS`).

**Second chances: variant formats.**

    sh scripts/run_second_chance_daggers.sh        # ergonomic format of the four † tasks, 9B + DeepSeek-V4-Flash
    sh scripts/run_second_chance.sh                # hanoi (execute) and cellular_automaton (current per-cell format)
    uv run python summarize2.py                    # -> results/SECOND_CHANCE.md

The ergonomic runs are fully specified by `plans/second-chance-daggers.json`. The hanoi and cellular reruns
are not, because they ran on changed task *defaults* with no knob:

- `results/second-chance-{9b,ds}` (logs `batch1-2nd-*`) hold hanoi `execute` and cellular per-cell revision 2,
  which no longer exists in code.
- `results/second-chance3-{9b,ds}` (logs `batch1-2nd3-*`) hold revision 3, the current `format="cells"`,
  run as `run_batch1.py --slugs cellular_automaton --max-tokens-cot 8192`.

The batch-1 formats are still reachable as `formulation="plan"` (hanoi, batch-1 depths 2,5,10,20,40,80) and
`format="rows"` (cellular_automaton, batch-1 depths 2,4,8,12,16,24). Neither task's `exemplars()` forwards the
knob, though, so passing it through `--plan` today pairs plan/rows instances with few-shot exemplars in the
default format. The batch-1 rows for those two tasks were produced by the defaults of the time.

**Batch 2: knockout, filler, continuation.**

    uv run python make_batch2_plan.py              # -> plans/batch2.json, from the 9B and DeepSeek batch-1 tables
    uv run python run_batch2.py --continue
    uv run python run_batch2.py --no-knockout --no-filler --continue --log-dir results/logs/batch2c2
    uv run python collect2.py --log-dir results/logs/batch2 --log-dir results/logs/batch2c2   # -> results/BENCH2.md, results/batch2.csv

`batch2c2` is a rerun of the continuation conditions that supersedes those cells in `batch2`. Each task runs
at its best-gap passing depth (ties go to the deeper depth): on the 9B where it passes there, otherwise on
DeepSeek-V4-Flash. hanoi and cellular_automaton were not in batch 2.

**Batch 3: gold knockout and mistake propagation (Qwen3.5-9B).**

    uv run python run_batch3.py --goldko --mistake
    uv run python collect3.py                      # -> results/BENCH3.md

`plans/batch3.json` fixes the depth per task. For most tasks it is the batch-2 depth, not the deepest passing
9B depth; `results/tidy/README.md` has the comparison. s5_composition, random_lookup_table and threesum run
in the ergonomic format. hanoi and cellular_automaton ran on their current defaults (execute, and per-cell
revision 3).

**Derived tables.**

    uv run python tidy.py       # ~1 min; every CSV in results/tidy/, cross-checked against the markdown tables
    uv run python density.py    # -> results/DENSITY.md, results/tidy/density.csv

## Another provider or model

Any Inspect model string works for batch 1. Any OpenAI-compatible server works as
`openai-api/<name>/<model>`, with `<NAME>_BASE_URL` and `<NAME>_API_KEY` set:

    VLLM_BASE_URL=http://host:8000/v1 VLLM_API_KEY=x STATEFUL_EXTRA_BODY= \
      uv run python run_batch1.py --model openai-api/vllm/Qwen/Qwen2.5-7B-Instruct --slugs dyck --n 20 --log-dir results/logs/mine
    uv run python collect.py --log-dir results/logs/mine --out results/mine

Leave `STATEFUL_EXTRA_BODY` empty unless the server understands `chat_template_kwargs`. For batches 2 and 3,
add a tag to `MODEL_LOGS` in `stateful/traces.py`, and check that the server continues a prefilled assistant
message: `CONTINUE_BODY` in `stateful/batch2.py` sends `continue_final_message`, which vLLM also accepts.

`--batch` sends batch-1 generation through the provider's batch API (on Together, half price and
asynchronous). Inspect 0.3.263's batcher nests `extra_body` inside the request body instead of merging it,
which silently drops the thinking-off flag. The driver patches the batcher, and `hidden_reasoning` would
catch a regression.

## Smoke test without a key

    uv run python check_task.py --all
    uv run python run_batch1.py --slugs stub_task --n 3 --model mockllm/model --log-dir results/logs/smoke

## Cost

The 9B batch-1 sweep (16 tasks, about 6 depths each, n=50, both arms: 10,400 samples) used 12.5M tokens.
The token total for every run is in the header of its `BENCH1.md`.
