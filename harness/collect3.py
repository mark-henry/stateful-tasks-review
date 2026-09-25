"""Batch 3 table: step-aligned gold knockout curve and mistake propagation. Writes results/BENCH3.md."""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from inspect_ai.log import list_eval_logs, read_eval_log
from stateful.registry import display_name
ap = argparse.ArgumentParser(); ap.add_argument("--log-dir", action="append"); ap.add_argument("--plan", default="plans/batch3.json"); ap.add_argument("--out", default="results")
a = ap.parse_args(); plan = json.load(open(a.plan))
seen = {}
for d in (a.log_dir or ["results/logs/batch3"]):
    for info in sorted(list_eval_logs(d), key=lambda i: i.name):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples: continue
        for s in log.samples:
            sc = s.scores.get("b2_check") if s.scores else None
            if sc is not None: seen[s.id] = (s.metadata, sc)
cells = defaultdict(lambda: [0, 0])
for md, sc in seen.values():
    key = (md["slug"], md["condition"], md["k"]); cells[key][0] += 1; cells[key][1] += (sc.value == "C")
def acc(slug, cond, k):
    n, c = cells.get((slug, cond, k), (0, 0)); return None if not n else c / n
f = lambda x: "·" if x is None else f"{x:.2f}"
L = ["# Batch 3 — step-aligned gold knockout and mistake propagation (Qwen3.5-9B)", "",
     "Gold trace prefilled through step k. goldko: then `Answer:` is forced (k = 0, d/4, d/2, 3d/4, d−1). mistake: step k's reported state is corrupted (task.corrupt_step) and the model continues; "
     "scored against the ORIGINAL answer, paired with the uncorrupted continuation at the same k (`cont`). propagation = cont − mistake: the share of answers the corruption changed.", "",
     "| task | depth | goldko k=0 | d/4 | d/2 | 3d/4 | d−1 | mistake@d/4 (cont) | @d/2 (cont) | @3d/4 (cont) | mean propagation |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for slug, p in plan.items():
    d = p["depth"]; kcols = [max(1, round(d * x)) for x in (0.25, 0.5, 0.75)]   # may repeat at small depth
    if not any(k[0] == slug for k in cells): continue
    g = [acc(slug, "goldko", k) for k in [0] + kcols + [d - 1]]
    ms = [(acc(slug, "mistake", k), acc(slug, "continue", k)) for k in kcols]
    props = [c - m for m, c in ms if m is not None and c is not None]
    L.append(f"| {display_name(slug)} | {d} | " + " | ".join(f(x) for x in g) + " | " + " | ".join(f"{f(m)} ({f(c)})" for m, c in ms) + f" | {f(sum(props)/len(props)) if props else '·'} |")
Path(a.out).mkdir(exist_ok=True); (Path(a.out) / "BENCH3.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
