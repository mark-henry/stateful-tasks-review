"""Batch-2 driver. Reads plans/batch2.json ({slug: {model, depth, ...}}) and runs, per slug:
knockout at fractions 0, .25, .5, .75; filler; and (when --continue) continue@k=depth//2 redacted vs plain.

    uv run python run_batch2.py [--slugs a,b] [--n 50] [--continue] [--log-dir results/logs/batch2]
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(Path(__file__).parent / ".env")
from inspect_ai import eval_set
from stateful.batch2 import batch2
from stateful.registry import load_task

ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="plans/batch2.json")
ap.add_argument("--slugs", default=None)
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--fractions", default="0,0.25,0.5,0.75")
ap.add_argument("--no-filler", action="store_true")
ap.add_argument("--no-knockout", action="store_true")
ap.add_argument("--continue", dest="cont", action="store_true", help="also run redacted/plain continuation at k=depth//2")
ap.add_argument("--max-connections", type=int, default=32)
ap.add_argument("--log-dir", default="results/logs/batch2")
a = ap.parse_args()

plan = json.load(open(a.plan))
slugs = a.slugs.split(",") if a.slugs else list(plan)
tasks = []
for s in slugs:
    m, d = plan[s]["model"], plan[s]["depth"]
    if not a.no_knockout:
        tasks += [batch2(slug=s, condition="knockout", model_tag=m, depth=d, n=a.n, fraction=float(f)) for f in a.fractions.split(",")]
    if not a.no_filler:
        tasks.append(batch2(slug=s, condition="filler", model_tag=m, depth=d, n=a.n))
    if a.cont:
        t = load_task(s)
        if getattr(t, "REDACTION_MEANINGFUL", False):
            k = max(1, d // 2)
            tasks += [batch2(slug=s, condition="continue", model_tag=m, depth=d, n=a.n, k=k, redacted=r) for r in (True, False)]
ok, logs = eval_set(tasks, log_dir=a.log_dir, max_connections=a.max_connections, retry_attempts=3,
                    fail_on_error=0.2, log_dir_allow_dirty=True)
print("[run_batch2] all complete" if ok else "[run_batch2] INCOMPLETE — rerun to resume")
