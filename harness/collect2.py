"""Collate batch-2 logs into results/BENCH2.md: per task, the knockout curve, filler recovery, and continuation."""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from inspect_ai.log import list_eval_logs, read_eval_log
from stateful.registry import display_name

ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", action="append")
ap.add_argument("--plan", default="plans/batch2.json")
ap.add_argument("--out", default="results")
a = ap.parse_args()
plan = json.load(open(a.plan))

seen = {}
for d in (a.log_dir or ["results/logs/batch2"]):
    for info in sorted(list_eval_logs(d), key=lambda i: i.name):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples: continue
        for s in log.samples:
            sc = s.scores.get("b2_check") if s.scores else None
            if sc is not None: seen[s.id] = (s.metadata, sc)
cells = defaultdict(lambda: {"n": 0, "c": 0, "kept": 0, "trunc": 0})
for md, sc in seen.values():
    key = (md["slug"], md["condition"], md.get("fraction") if md["condition"] == "knockout" else (f"k{md['k']}{'r' if md['redacted'] else 'p'}" if md["condition"] == "continue" else ""))
    r = cells[key]; r["n"] += 1; r["c"] += (sc.value == "C"); r["kept"] += bool((sc.metadata or {}).get("kept_reasoning")); r["trunc"] += bool((sc.metadata or {}).get("truncated"))
def acc(slug, cond, sub=""):
    r = cells.get((slug, cond, sub)); return None if not r or not r["n"] else r["c"] / r["n"]
f = lambda x: "·" if x is None else f"{x:.2f}"
L = ["# Batch 2 — knockout curve, filler recovery, redacted continuation", "",
     "Each task at its batch-2 (model, depth) from plans/batch2.json. acc(CoT)/acc(no-CoT) are batch-1 values on the same instances. "
     "knockout@f = the model's own trace cut to the first f of its lines, then `Answer:` prefilled. filler = the trace replaced by the same number of ` .` tokens. "
     "filler recovery = filler − no-CoT. continue@k: gold trace through step k prefilled; r = prompt redacted (initial state + first k operators removed), p = plain prompt.", "",
     "| task | model | depth | n | acc(CoT) | no-CoT | ko@0 | ko@.25 | ko@.5 | ko@.75 | filler | filler rec. | cont@k r | cont@k p | kept reasoning (ko@.5) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
rows = []
for slug, p in sorted(plan.items()):
    n = cells.get((slug, "knockout", 0.5), {}).get("n") or cells.get((slug, "filler", ""), {}).get("n") or 0
    if not n: continue
    ko = [acc(slug, "knockout", x) for x in (0.0, 0.25, 0.5, 0.75)]
    fi = acc(slug, "filler"); an = p["acc_nocot"]
    ks = sorted({k[2] for k in cells if k[0] == slug and k[1] == "continue"})
    cr = next((acc(slug, "continue", k) for k in ks if k.endswith("r")), None); cp = next((acc(slug, "continue", k) for k in ks if k.endswith("p")), None)
    kept = cells.get((slug, "knockout", 0.5), {}); keptr = kept["kept"] / kept["n"] if kept.get("n") else None
    L.append(f"| {display_name(slug)} | {p['model']} | {p['depth']} | {n} | {p['acc_cot']:.2f} | {an:.2f} | {f(ko[0])} | {f(ko[1])} | {f(ko[2])} | {f(ko[3])} | {f(fi)} | {f(None if fi is None else fi - an)} | {f(cr)} | {f(cp)} | {f(keptr)} |")
    rows.append([slug, p["model"], p["depth"], n, p["acc_cot"], an] + ko + [fi, cr, cp, keptr])
Path(a.out).mkdir(exist_ok=True)
(Path(a.out) / "BENCH2.md").write_text("\n".join(L) + "\n")
with open(Path(a.out) / "batch2.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["slug", "model", "depth", "n", "acc_cot", "acc_nocot", "ko0", "ko25", "ko50", "ko75", "filler", "cont_redacted", "cont_plain", "kept_reasoning_ko50"]); w.writerows(rows)
print("\n".join(L))
