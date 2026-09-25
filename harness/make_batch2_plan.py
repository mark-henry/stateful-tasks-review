"""Pick (model, depth) per survivor for batch 2: prefer Qwen3.5-9B; else DeepSeek. Depth = the passing depth
with the largest gap (ties -> deeper). Writes plans/batch2.json."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, ".")
from stateful.registry import DROPPED
GAP, ACC = 0.2, 0.7
def passing(d):
    rows = {}
    for r in csv.DictReader(open(Path(d) / "batch1.csv")):
        rows.setdefault((r["slug"], int(r["depth"])), {})[r["condition"]] = float(r["acc"])
    ps = {}
    for line in open(Path(d) / "BENCH1.md"):
        if line.startswith("| ") and not line.startswith("| task"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 8: ps[(c[0].rstrip("*"), int(c[1]))] = 0.0005 if c[7].startswith("<") else (float(c[7]) if c[7] != "·" else 1.0)
    out = {}
    for (slug, dep), v in rows.items():
        if "cot" in v and "nocot" in v and v["cot"] - v["nocot"] >= GAP and ps.get((slug, dep), 1) < 0.05 and v["cot"] >= ACC:
            out.setdefault(slug, []).append((v["cot"] - v["nocot"], dep, v["cot"], v["nocot"]))
    return out
plan = {}
for tag, d in [("qwen9b", "results/qwen35-9b"), ("ds", "results/deepseek-v4-flash")]:
    for slug, cells in passing(d).items():
        if slug in DROPPED or slug in plan: continue
        gap, dep, ac, an = max(cells, key=lambda c: (round(c[0], 2), c[1]))
        plan[slug] = {"model": tag, "depth": dep, "acc_cot": ac, "acc_nocot": an, "gap": round(gap, 2)}
Path("plans").mkdir(exist_ok=True); json.dump(plan, open("plans/batch2.json", "w"), indent=1)
for s, v in sorted(plan.items()): print(f"{s:26s} {v['model']:7s} d={v['depth']:<3d} acc={v['acc_cot']:.2f} nocot={v['acc_nocot']:.2f} gap={v['gap']:.2f}")
