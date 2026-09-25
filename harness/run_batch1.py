"""Batch-1 sweep driver. Runs every (slug, condition) as its own Inspect task under eval_set, so reruns
resume/retry instead of re-spending.

    uv run python run_batch1.py --model openrouter/qwen/qwen-2.5-7b-instruct --n 50
    uv run python run_batch1.py --slugs s5_composition,threesum --n 5 --model mockllm/model   # smoke

Model strings are Inspect model strings. Any OpenAI-compatible server works via
    --model openai-api/<name>/<model>  with <NAME>_BASE_URL and <NAME>_API_KEY set.
"""
import argparse, json, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))
from inspect_ai import eval_set
from stateful.batch1 import batch1
from stateful.registry import SURVIVORS, SWEEP_KNOBS, implemented, load_task

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("STATEFUL_MODEL", "openrouter/qwen/qwen-2.5-7b-instruct"))
ap.add_argument("--slugs", default=None, help="comma list; default = all implemented survivors")
ap.add_argument("--conditions", default="cot,nocot")
ap.add_argument("--n", type=int, default=50, help="instances per depth")
ap.add_argument("--depths", default=None, help="override DEPTHS, comma list (applies to every slug)")
ap.add_argument("--shots", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-connections", type=int, default=32)
ap.add_argument("--max-tokens-cot", type=int, default=0, help="0 = sized from gold traces")
ap.add_argument("--log-dir", default="results/logs/batch1")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--retry-attempts", type=int, default=3)
ap.add_argument("--plan", default=None, help='JSON {slug: {"depths": [...], "n": N}}; only listed slugs run')
ap.add_argument("--batch", action="store_true", help="use the provider batch API (async, half price, slow)")
a = ap.parse_args()

plan = json.load(open(a.plan)) if a.plan else None
slugs = list(plan) if plan else (a.slugs.split(",") if a.slugs else implemented())
missing = [s for s in SURVIVORS if s not in implemented(include_dropped=True)]
if missing and not a.slugs:
    print(f"[run_batch1] not yet implemented, skipping: {missing}", file=sys.stderr)
for s in slugs:
    load_task(s)  # fail fast on contract violations before spending tokens

def spec(s):
    if plan: return plan[s].get("depths", a.depths), plan[s].get("n", a.n), {**SWEEP_KNOBS.get(s, {}), **plan[s].get("knobs", {})}
    return a.depths, a.n, SWEEP_KNOBS.get(s, {})
tasks = [batch1(slug=s, condition=c, n=spec(s)[1], depths=spec(s)[0], shots=a.shots, seed=a.seed,
                max_tokens_cot=a.max_tokens_cot, knobs=spec(s)[2])
         for s in slugs for c in a.conditions.split(",")]
extra = {}
if os.environ.get("STATEFUL_EXTRA_BODY"):   # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
    extra["extra_body"] = json.loads(os.environ["STATEFUL_EXTRA_BODY"])
if a.batch:
    extra["batch"] = True
    # Inspect 0.3.263: the OpenAI-style batcher copies client kwargs into the JSONL body verbatim, so
    # `extra_body` becomes a nested key instead of being merged (the live path merges it via the SDK).
    # Hoist it so provider-specific flags (e.g. chat_template_kwargs.enable_thinking) survive batching.
    from inspect_ai.model._providers import _openai_batch as _ob
    _orig = _ob.OpenAIBatcher._jsonl_line_for_request
    def _patched(self, request, custom_id):
        line = _orig(self, request, custom_id)
        body = dict(line["body"]); eb = body.pop("extra_body", None)
        if isinstance(eb, dict): body.update(eb)
        line["body"] = body
        return line
    _ob.OpenAIBatcher._jsonl_line_for_request = _patched
ok, logs = eval_set(tasks, model=a.model, log_dir=a.log_dir, max_connections=a.max_connections,
                    limit=a.limit, retry_attempts=a.retry_attempts, fail_on_error=0.2,
                    log_dir_allow_dirty=True, **extra)
print("[run_batch1] all complete" if ok else "[run_batch1] INCOMPLETE — rerun to resume")
