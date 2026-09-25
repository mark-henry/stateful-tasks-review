"""Second-chance table: as-published batch-1 verdict vs the variant, per model. Writes results/SECOND_CHANCE.md."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from stateful.registry import display_name
GAP, P, ACC = 0.2, 0.05, 0.7
def best(d, slug):
    """(passing best depth, acc, nocot, gap) or (best-gap cell, ...) with pass flag."""
    if not (Path(d) / "batch1.csv").exists(): return None
    rows, ps = {}, {}
    for r in csv.DictReader(open(Path(d) / "batch1.csv")):
        if r["slug"] == slug: rows.setdefault(int(r["depth"]), {})[r["condition"]] = float(r["acc"])
    for line in open(Path(d) / "BENCH1.md"):
        if line.startswith("| ") and not line.startswith("| task"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 8 and c[0].rstrip("*†") == slug: ps[int(c[1])] = 0.0005 if c[7].startswith("<") else (float(c[7]) if c[7] != "·" else 1.0)
    cells = [(v["cot"] - v["nocot"], dep, v["cot"], v["nocot"], ps.get(dep, 1.0)) for dep, v in rows.items() if "cot" in v and "nocot" in v]
    if not cells: return None
    ok = [c for c in cells if c[0] >= GAP and c[4] < P and c[2] >= ACC]
    c = max(ok or cells, key=lambda x: (round(x[0], 2), x[1]))
    return (bool(ok), c[1], c[2], c[3], c[0])
def fmt(x):
    if x is None: return "·"
    ok, dep, ac, an, gap = x
    return f"{'PASS' if ok else 'fail'} d{dep}: {ac:.2f} / {an:.2f} / {gap:+.2f}"
variants = [  # slug, variant label, (9b dir, ds dir) for the variant
    ("hanoi", "execution formulation", ("results/second-chance-9b", "results/second-chance-ds")),
    ("cellular_automaton", "per-cell format (rev 2)", ("results/second-chance-9b", "results/second-chance-ds")),
    ("cellular_automaton", "per-cell + named neighbours + running row (rev 3)", ("results/second-chance3-9b", "results/second-chance3-ds")),
    ("s5_composition", "ergonomic (SFT-free) format", ("results/ergonomic-9b", "results/ergonomic-ds")),
    ("random_lookup_table", "ergonomic (SFT-free) format", ("results/ergonomic-9b", "results/ergonomic-ds")),
    ("threesum", "ergonomic (SFT-free) format", ("results/ergonomic-9b", "results/ergonomic-ds")),
    ("addition", "ergonomic (SFT-free) format", ("results/ergonomic-9b", "results/ergonomic-ds")),
]
L = ["# Second chances — as published vs variant", "",
     f"Cell = PASS/fail under the batch-1 rules (gap ≥ {GAP}, McNemar p < {P}, acc(CoT) ≥ {ACC}) at the best passing depth (else best-gap depth): acc(CoT) / acc(no-CoT) / gap. "
     "As-published columns come from the batch-1 tables; variant columns from the second-chance runs. `†` = SFT/from-scratch format in the source; `*` = no published trace.", "",
     "| task | variant | Qwen3.5-9B as published | Qwen3.5-9B variant | DeepSeek-V4-Flash as published | DeepSeek-V4-Flash variant |", "|---|---|---|---|---|---|"]
for slug, label, (d9, dds) in variants:
    L.append(f"| {display_name(slug)} | {label} | {fmt(best('results/qwen35-9b', slug))} | {fmt(best(d9, slug))} | {fmt(best('results/deepseek-v4-flash', slug))} | {fmt(best(dds, slug))} |")
Path("results/SECOND_CHANCE.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
