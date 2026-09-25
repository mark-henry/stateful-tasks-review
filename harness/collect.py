"""Collate batch-1 Inspect logs into results/batch1.csv and results/BENCH1.md.

    uv run python collect.py [--log-dir results/logs/batch1]
"""
import argparse, csv, math, sys
from collections import defaultdict
from pathlib import Path
from inspect_ai.log import list_eval_logs, read_eval_log
import sys as _sys; _sys.path.insert(0, str(Path(__file__).parent))
from stateful.registry import display_name

ap = argparse.ArgumentParser()
ap.add_argument("--log-dir", action="append", help="repeatable; later dirs override earlier on the same sample id (default results/logs/batch1)")
ap.add_argument("--out", default="results")
a = ap.parse_args()

from math import comb
paired = defaultdict(dict)   # (slug, depth, inst_seed) -> {condition: bool}
def mcnemar_exact(b, c):
    """two-sided exact binomial test on discordant pairs; b = CoT right & noCoT wrong, c = reverse"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
rows = defaultdict(lambda: {"n": 0, "correct": 0, "leak": 0, "trunc": 0, "hidden": 0, "out_tok": 0, "in_tok": 0, "no_ans": 0})
model = None
seen = {}   # sample id -> (metadata, score); later log dirs override earlier
dirs = a.log_dir or ["results/logs/batch1"]
for d_ in dirs:
    for info in sorted(list_eval_logs(d_), key=lambda i: i.name):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue
        model = log.eval.model
        for s in log.samples:
            sc = s.scores.get("task_check") if s.scores else None
            if sc is not None: seen[s.id] = (s.metadata, sc)
for md, sc in seen.values():
    if True:
        r = rows[(md["slug"], md["condition"], md["depth"])]
        paired[(md["slug"], md["depth"], md["inst_seed"])][md["condition"]] = (sc.value == "C")
        r["n"] += 1
        r["correct"] += (sc.value == "C")
        m = sc.metadata or {}
        r["leak"] += bool(m.get("leaked_reasoning")); r["hidden"] += bool(m.get("hidden_reasoning")); r["trunc"] += bool(m.get("truncated"))
        r["no_ans"] += (not m.get("has_answer_line"))
        r["out_tok"] += m.get("output_tokens") or 0; r["in_tok"] += m.get("input_tokens") or 0

out = Path(a.out); out.mkdir(exist_ok=True)
with open(out / "batch1.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["slug", "condition", "depth", "n", "acc", "stderr", "leak_rate", "trunc_rate", "hidden_reasoning_rate", "no_answer_rate", "mean_out_tok", "mean_in_tok"])
    for (slug, cond, d), r in sorted(rows.items()):
        n = r["n"]; p = r["correct"] / n
        w.writerow([slug, cond, d, n, f"{p:.3f}", f"{math.sqrt(p*(1-p)/n):.3f}", f"{r['leak']/n:.2f}",
                    f"{r['trunc']/n:.2f}", f"{r['hidden']/n:.2f}", f"{r['no_ans']/n:.2f}", f"{r['out_tok']/n:.0f}", f"{r['in_tok']/n:.0f}"])

def discordant(slug, d):
    b = c = 0
    for (s_, d_, _), v in paired.items():
        if s_ == slug and d_ == d and "cot" in v and "nocot" in v:
            b += v["cot"] and not v["nocot"]; c += v["nocot"] and not v["cot"]
    return b, c

# BENCH1.md: one row per (slug, depth): acc cot / acc nocot / gap / McNemar p
slugs = sorted({k[0] for k in rows}); tot_tok = sum(r["out_tok"] + r["in_tok"] for r in rows.values())
hidden_tot = sum(r["hidden"] for r in rows.values()); n_tot = sum(r["n"] for r in rows.values()) or 1
L = [f"# Batch 1 — acc(CoT), acc(no-CoT), CoT gap by depth", f"", f"Model: `{model}`. Temperature 0. n per cell in table. Total tokens: {tot_tok:,}. Samples with hidden reasoning: {hidden_tot}/{n_tot}.", "",
     "`†` = the source taught the format by fine-tuning / from-scratch training; `*` = no published CoT trace exists (format is ours). See harness/stateful/registry.py.", "",
     "Gap is paired (same instances in both arms); p is the two-sided exact McNemar test on discordant pairs (b = CoT right/no-CoT wrong, c = reverse).", "",
     "| task | depth | n | acc(CoT) | acc(no-CoT) | gap | b/c | p | CoT trunc | no-CoT leak | tok/CoT |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for slug in slugs:
    depths = sorted({k[2] for k in rows if k[0] == slug})
    for d in depths:
        c = rows.get((slug, "cot", d)); nc = rows.get((slug, "nocot", d))
        def acc(r): return None if not r else r["correct"] / r["n"]
        ac, an = acc(c), acc(nc)
        gap = None if ac is None or an is None else ac - an
        f = lambda x: "·" if x is None else f"{x:.2f}"
        b_, c_ = discordant(slug, d); pv = mcnemar_exact(b_, c_) if (c and nc) else None
        pf = "·" if pv is None else (f"{pv:.3f}" if pv >= 0.001 else "<.001")
        L.append(f"| {display_name(slug)} | {d} | {(c or nc)['n']} | {f(ac)} | {f(an)} | {f(gap)} | {b_}/{c_} | {pf} | "
                 f"{f(c['trunc']/c['n']) if c else '·'} | {f(nc['leak']/nc['n']) if nc else '·'} | "
                 f"{(c['out_tok']//c['n']) if c else '·'} |")
(out / "BENCH1.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
