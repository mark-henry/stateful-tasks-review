"""Batch 3: step-aligned gold knockout and mistake propagation, on plans/batch3.json (deepest passing 9B depth).
    uv run python run_batch3.py [--goldko] [--mistake] [--n 50]
goldko k in {0, d/4, d/2, 3d/4, d-1}; mistake k in {d/4, d/2, 3d/4} plus the matching plain continuation (cached from batch 2 where it exists).
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(Path(__file__).parent / ".env")
from inspect_ai import eval_set
from stateful.batch2 import batch2
from stateful.registry import load_task
ap = argparse.ArgumentParser()
ap.add_argument("--plan", default="plans/batch3.json"); ap.add_argument("--slugs", default=None)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--goldko", action="store_true"); ap.add_argument("--mistake", action="store_true")
ap.add_argument("--max-connections", type=int, default=32); ap.add_argument("--log-dir", default="results/logs/batch3")
a = ap.parse_args()
plan = json.load(open(a.plan)); slugs = a.slugs.split(",") if a.slugs else list(plan)
def ks(d): return sorted({max(1, round(d * f)) for f in (0.25, 0.5, 0.75)})
tasks = []
for s in slugs:
    m, d, kn = plan[s]["model"], plan[s]["depth"], plan[s].get("knobs", {})
    if a.goldko:
        tasks += [batch2(slug=s, condition="goldko", model_tag=m, depth=d, n=a.n, k=k, knobs=kn) for k in sorted({0, *ks(d), d - 1})]
    if a.mistake:
        if not hasattr(load_task(s), "corrupt_step"): print(f"[batch3] {s}: no corrupt_step yet, skipping mistake", file=sys.stderr); continue
        for k in ks(d):
            tasks.append(batch2(slug=s, condition="mistake", model_tag=m, depth=d, n=a.n, k=k, knobs=kn))
            tasks.append(batch2(slug=s, condition="continue", model_tag=m, depth=d, n=a.n, k=k, redacted=False, knobs=kn))
ok, logs = eval_set(tasks, log_dir=a.log_dir, max_connections=a.max_connections, retry_attempts=3, fail_on_error=0.2, log_dir_allow_dirty=True)
print("[run_batch3] all complete" if ok else "[run_batch3] INCOMPLETE — rerun to resume")
