"""Apply the batch-1 survival rules to one or more model tables and write results/SUMMARY.md.

Rule 1: gap >= 0.2 with paired McNemar p < 0.05 at some depth.
Rule 2: acc(CoT) >= 0.7 at that same depth.
Working depth = shallowest depth satisfying both; "best" = the depth maximising gap subject to both.
"""
import csv, sys
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from stateful.registry import display_name, DROPPED, ASTERISK, DAGGER

GAP, P, ACC = 0.2, 0.05, 0.7
models = {"Qwen3.5-9B": "results/qwen35-9b", "Llama-3.3-70B": "results/llama33-70b", "DeepSeek-V4-Flash": "results/deepseek-v4-flash"}

def load(d):
    rows = {}
    for r in csv.DictReader(open(Path(d) / "batch1.csv")):
        rows.setdefault((r["slug"], int(r["depth"])), {})[r["condition"]] = r
    return rows

def p_from_bench(d):
    # BENCH1.md carries b/c and p; parse them back so we don't recompute from logs
    out = {}
    for line in open(Path(d) / "BENCH1.md"):
        if not line.startswith("| ") or line.startswith("| task"): continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) < 8: continue
        slug = c[0].rstrip("*"); dep = int(c[1]); pv = c[7]
        out[(slug, dep)] = 0.0005 if pv.startswith("<") else (float(pv) if pv != "·" else 1.0)
    return out

verdicts = {}
for mname, d in models.items():
    if not (Path(d) / "batch1.csv").exists(): continue
    rows, ps = load(d), p_from_bench(d)
    for (slug, dep), conds in rows.items():
        if "cot" not in conds or "nocot" not in conds: continue
        ac, an = float(conds["cot"]["acc"]), float(conds["nocot"]["acc"])
        gap, pv = ac - an, ps.get((slug, dep), 1.0)
        v = verdicts.setdefault(slug, {}).setdefault(mname, {"cells": [], "pass": []})
        v["cells"].append((dep, ac, an, gap, pv))
        if gap >= GAP and pv < P and ac >= ACC: v["pass"].append((dep, ac, an, gap))

slugs = sorted(verdicts)
L = ["# Batch 1 summary — survival by model", "",
     "Model roster, fixed 2026-09-16 after the first two runs and before the third: Qwen3.5-9B (thinking off), Llama-3.3-70B, DeepSeek-V4-Flash (thinking off). "
     "DeepSeek was added because Llama-3.3 (Dec 2024) underperformed the 9B with CoT and ignores published formats; it runs on the FULL grid, not only on the 70B's failures, and it is the last model for batch 1. "
     "Survival rules were fixed before any model ran and are unchanged. A task survives batch 1 if it passes on at least one model; `†` (format taught by SFT/from-scratch training in the source) and `*` (no published trace) tasks pass through regardless.", "",
     f"Rule 1: gap ≥ {GAP} with paired McNemar p < {P}. Rule 2: acc(CoT) ≥ {ACC} at the same depth. Working depth = shallowest passing depth; best = passing depth with the largest gap.", "",
     "| task | " + " | ".join(f"{m}: working / best (acc, gap)" for m in models) + " | verdict |",
     "|---|" + "---|" * len(models) + "---|"]
for slug in slugs:
    cells = []; alive = 0
    for m in models:
        v = verdicts[slug].get(m)
        if not v: cells.append("·"); continue
        if v["pass"]:
            alive += 1
            w = min(v["pass"]); b = max(v["pass"], key=lambda x: x[3])
            cells.append(f"d{w[0]} / d{b[0]} ({b[1]:.2f}, {b[3]:.2f})")
        else:
            best = max(v["cells"], key=lambda x: x[3])
            reason = "acc<0.7" if any(c[3] >= GAP and c[4] < P for c in v["cells"]) or any(c[3] >= GAP for c in v["cells"]) else "no gap"
            cells.append(f"— ({reason}; best gap {best[3]:.2f} at d{best[0]}, acc {best[1]:.2f})")
    if slug in DROPPED: verdict = "DROPPED: " + DROPPED[slug]
    elif alive == len([m for m in models if (Path(models[m]) / "batch1.csv").exists()]): verdict = "survives (all models)"
    elif alive >= 1: verdict = f"survives ({alive} model{'s' if alive > 1 else ''})"
    elif slug in DAGGER: verdict = "passes through with † (" + DAGGER[slug] + ")"
    elif slug in ASTERISK: verdict = "passes through with * (" + ASTERISK[slug] + ")"
    else: verdict = "fails batch 1"
    L.append(f"| {display_name(slug)} | " + " | ".join(cells) + f" | {verdict} |")
Path("results/SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
