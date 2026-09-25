"""Ballpark "load-bearing token density" of each task's gold CoT format.

    cd ~/stateful-tasks/harness && uv run python density.py [--examples] [--check]

For every step of the gold trace (split exactly as batch 3's gold_prefix splits it), count
  step_tokens  = Qwen3.5-9B tokens of the step as it sits in the assistant turn (format_cot tokenized whole,
                 tokens attributed to the step their first character falls in), and
  state_tokens = the subset of those tokens overlapping the step's STATE field(s), located per task by a
                 regex (`SPANS`) and validated against task.corrupt_step (every character the corruption
                 rewrites must fall inside an extracted span).
state_frac = mean state tokens / mean step tokens; density_est = state_frac * mean propagation (BENCH3.md).
Writes results/tidy/density.csv and results/DENSITY.md.
"""
from __future__ import annotations
import csv, difflib, json, math, re, statistics as st, sys
from pathlib import Path

from stateful.registry import load_task, SWEEP_KNOBS, TASKS_DIR
from stateful.batch1 import instance_seed
from stateful.batch2 import gold_prefix

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
N, SEED = 30, 0
TOKENIZER = "Qwen/Qwen3.5-9B"

# ---------------------------------------------------------------- configurations
BATCH3 = json.loads((HERE / "plans" / "batch3.json").read_text())
BATCH2 = json.loads((HERE / "plans" / "batch2.json").read_text())
CORRUPTIBLE = set(BATCH3)

def configs():
    """(slug, format_label, depth, knobs, depth_source, ran_in_batch3)."""
    out = []
    for slug, p in BATCH3.items():
        kn = dict(p["knobs"])
        label = kn.get("format", {"hanoi": "execute", "cellular_automaton": "cells"}.get(slug, "published"))
        out.append((slug, label, p["depth"], kn, "batch3", True))
        if kn.get("format") == "ergonomic":           # published format of the same dagger task, same depth
            out.append((slug, "published", p["depth"], {k: v for k, v in kn.items() if k != "format"}, "batch3", False))
    # addition†: fourth ergonomic task, not in batch 3; batch-2/second-chance depth 2
    d = BATCH2["addition"]["depth"]
    out += [("addition", "ergonomic", d, {"format": "ergonomic"}, "batch2", False),
            ("addition", "published", d, {}, "batch2", False)]
    for slug in ["boolean_expressions", "blocksworld", "entity_tracking_boxes", "cruxeval"]:
        out.append((slug, "published", BATCH2[slug]["depth"], dict(SWEEP_KNOBS.get(slug, {})), "batch2", False))
    return out

# ---------------------------------------------------------------- state-span extractors
# Each returns a list of (start, end) char spans inside the step text. Values only (field labels such as
# "stack:" or "Now at" are scaffolding); a structured value (a list, dict, name:role list) is one span.
# ONE_COPY marks formats that write the same post-step state more than once: index of the canonical copy.

def _groups(pat, flags=0):
    rx = re.compile(pat, flags)
    def f(s):
        m = rx.search(s)
        if not m: return []
        return [m.span(g) for g in range(1, (rx.groups or 0) + 1) if m.group(g) is not None]
    return f

def _all(*pats):
    rxs = [re.compile(p, re.M) for p in pats]
    def f(s):
        return sorted(m.span(1) for rx in rxs for m in rx.finditer(s))
    return f

def _threesum_published(s):
    # "<d>- <e>- <f>- <x> <y> <z> <s1> [<s2> ...]": the digit-wise sums after the 3 indices + 3 digits
    toks = [m.span() for m in re.finditer(r"\S+", s)]
    return [(toks[6][0], toks[-1][1])] if len(toks) > 6 else []

SPANS = {
    ("nested_arithmetic", "published"): _groups(r"= (-?\d+)\.?(?:\s*So the answer is (-?\d+)\.)?\s*$"),
    ("turing_machine", "published"): _groups(r"Queue State: (\[.*?\])"),
    ("synthetic_program_trace", "published"): _groups(r"state: (.*)$", re.M),
    ("dyck", "published"): _groups(r"stack: (.*)$"),
    ("cup_shuffling", "published"): _groups(r"^\(\d+\)[^:]*: (.*?)\.?$"),
    ("s5_composition", "ergonomic"): _groups(r"of \d+ -> (.*?) -> (\d+)$"),
    ("s5_composition", "published"): _groups(r"-> (\d+)$"),
    ("random_lookup_table", "ergonomic"): _groups(r"-> (\S+?)\. Now at (\S+?)\.$"),
    ("random_lookup_table", "published"): _groups(r" (\S+)$"),
    ("threesum", "ergonomic"): _groups(r"= (\([^)]*\)) -> (\([^)]*\)) mod \d+ -> (yes|no)$"),
    ("threesum", "published"): _threesum_published,
    ("hanoi", "execute"): _groups(r"-> (.*)$"),
    ("cellular_automaton", "cells"): _all(r"-> [01]{3} -> ([01])", r"row so far: ([01_]+)", r"^\s*row: ([01]+)"),
    ("addition", "ergonomic"): _groups(r"write (\d+), carry (\d+)"),
    ("addition", "published"): _groups(r", ?(.*?) ?C: (\d+)$"),
    ("boolean_expressions", "published"): _groups(r"= (True|False)\.(?:\s*So the answer is (\w+)\.)?\s*$"),
    ("blocksworld", "published"): _groups(r"Resulting State: (.*)$", re.M),
    ("entity_tracking_boxes", "published"): _groups(r"^After op \d+: (.*?)\.?$"),
    ("cruxeval", "published"): None,
}
ONE_COPY = {("s5_composition", "ergonomic"): -1, ("random_lookup_table", "ergonomic"): -1,
            ("threesum", "ergonomic"): -1, ("cellular_automaton", "cells"): -1,
            ("nested_arithmetic", "published"): 0, ("boolean_expressions", "published"): 0}

# ---------------------------------------------------------------- inputs: propagation, bits
def propagation():
    out = {}
    for line in (RES / "BENCH3.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) > 5 and cells[0].rstrip("†*") in CORRUPTIBLE:
            try: out[cells[0].rstrip("†*")] = float(cells[-1])
            except ValueError: pass
    return out

def bits(slug, inst):
    if slug == "hanoi":   # desk.json's 4.75 is 3 disks; bits scale as n*log2(3)
        n = inst.meta.get("disks") or inst.meta.get("n") or sum(len(p) for p in json.loads(inst.states[0]))
        return n * math.log2(3), f"hanoi bits = {n} disks x log2(3)"
    b = json.loads((TASKS_DIR / slug / "desk.json").read_text()).get("state_bits")
    return (b if isinstance(b, (int, float)) else None), ""

# ---------------------------------------------------------------- measurement
def tokenizer():
    from tokenizers import Tokenizer
    try:
        return Tokenizer.from_pretrained(TOKENIZER), TOKENIZER
    except Exception as e:   # fall back to the cached Qwen3-8B tokenizer
        print(f"hub download of {TOKENIZER} failed ({e}); falling back to Qwen/Qwen3-8B", file=sys.stderr)
        return Tokenizer.from_pretrained("Qwen/Qwen3-8B"), "Qwen/Qwen3-8B (fallback)"

def step_offsets(t, inst):
    """(start, end) of each step in format_cot, via batch 3's gold_prefix."""
    offs = []
    for k in range(1, len(inst.steps) + 1):
        end = len(gold_prefix(t, inst, k)) - 1
        offs.append((end - len(inst.steps[k - 1]), end))
    return offs

def measure_instance(tok, t, inst, fn, one_idx):
    full = t.format_cot(inst)
    enc = tok.encode(full, add_special_tokens=False)
    toks = [o for o in enc.offsets if o[1] > o[0]]
    rows = []
    for (a, b), step in zip(step_offsets(t, inst), inst.steps):
        mine = [o for o in toks if a <= o[0] < b]
        spans = fn(step) if fn else []
        if fn and not spans:
            raise ValueError(f"no state span in step: {step[:80]!r}")
        def count(sp):
            return sum(1 for o in mine if any(o[0] < a + e and o[1] > a + s for s, e in sp))
        one = [spans[one_idx]] if (one_idx is not None and spans) else spans
        rows.append((len(mine), count(spans), count(one)))
    return rows

def corruption_coverage(t, inst, fn):
    """Share of corrupt_step edits (k in 1, d//2, d) that fall inside the extracted state spans."""
    d = len(inst.steps); hit = tot = 0
    for k in sorted({1, max(1, d // 2), d}):
        s = inst.steps[k - 1]; c, _ = t.corrupt_step(inst, k, SEED)
        spans = fn(s)
        for op, i1, i2, _, _ in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
            if op == "equal": continue
            tot += 1
            hit += any(sa <= i1 and i2 <= se for sa, se in spans)
    return hit, tot

def bracket(step, spans):
    out, pos = [], 0
    for s, e in spans:
        out += [step[pos:s], "⟦", step[s:e], "⟧"]; pos = e
    return "".join(out + [step[pos:]])

def run(show_examples=False):
    tok, tok_name = tokenizer()
    prop = propagation()
    rows, examples = [], {}
    for slug, fmt, depth, knobs, dsrc, ran3 in configs():
        t = load_task(slug); fn = SPANS[(slug, fmt)]; one_idx = ONE_COPY.get((slug, fmt))
        insts = [t.generate(depth, instance_seed(SEED, depth, i), **knobs) for i in range(N)]
        b, bnote = bits(slug, insts[0])
        notes = [f"depth from {dsrc} plan"] + ([bnote] if bnote else [])
        per = [measure_instance(tok, t, inst, fn, one_idx) for inst in insts]
        steps = [r for p in per for r in p]
        step_mean = st.mean(r[0] for r in steps)
        sd_within = st.mean(st.pstdev([r[0] for r in p]) for p in per)
        first = st.mean(p[0][0] for p in per); last = st.mean(p[-1][0] for p in per)
        mid = [r[0] for p in per for r in p[1:-1]]
        spread = f"step tokens: first {first:.1f}, middle {st.mean(mid):.1f}, last {last:.1f}, within-trace sd {sd_within:.1f}" if mid else \
                 f"step tokens: first {first:.1f}, last {last:.1f}, within-trace sd {sd_within:.1f}"
        rec = dict(slug=slug, format=fmt, depth=depth, n=N, step_tokens_mean=round(step_mean, 2),
                   state_tokens_mean="", state_frac="", mean_propagation="", density_est="",
                   bits_per_step=(round(b, 3) if b is not None else ""), bits_per_state_token="",
                   estimable="no", note="", _spread=spread, _one="", _cov="")
        if fn:
            state_mean = st.mean(r[1] for r in steps); frac = state_mean / step_mean
            rec.update(state_tokens_mean=round(state_mean, 2), state_frac=round(frac, 3))
            if one_idx is not None:
                one_mean = st.mean(r[2] for r in steps)
                rec["_one"] = f"{one_mean:.1f} tok / {one_mean / step_mean:.3f}"
                notes.append(f"state written more than once per step; canonical copy only: {one_mean:.1f} tok, frac {one_mean / step_mean:.3f}")
            if b is not None: rec["bits_per_state_token"] = round(b / state_mean, 3)
            has_prop = ran3 and slug in prop
            if has_prop:
                rec["mean_propagation"] = prop[slug]
                rec["density_est"] = round(frac * prop[slug], 3)
            if slug in CORRUPTIBLE:
                h = tt = 0
                for inst in insts:
                    a, c = corruption_coverage(t, inst, fn); h += a; tt += c
                rec["_cov"] = f"{h}/{tt}"
                notes.append(f"corrupt_step edits inside span: {h}/{tt}")
            rec["estimable"] = "yes" if has_prop else "partial"
        rows.append((rec, notes))
        if fn:
            examples[(slug, fmt)] = [(s, fn(s)) for s in insts[0].steps[:3]]
    apply_judgement(rows)
    return rows, examples, tok_name

# Task-specific judgement (why a number is not trustworthy); keyed (slug, format).
WILD = {
    ("threesum", "ergonomic"): ("no", "propagation is negative (-0.11): the clean continuation already fails (cont 0.34-0.54), so a corrupted step has nothing to break. The load-bearing state is the enumeration position (which candidate triple comes next); it shows up only as the step label (i,j,k), which is scaffolding here and which corrupt_step never touches, so the counted span (sums + verdict) is not what the model relies on. desk.json's 3.32 bits is one mod-10 digit, not this state"),
    ("threesum", "published"): ("no", "format for a model trained from scratch; the real state is the enumeration position, written only as the index label; no propagation measured"),
    ("cruxeval", "published"): ("no", "free prose/code-delta trace: no fixed state field; each step reports only the variable a line touches, so the program state is never written in full"),
    ("boolean_expressions", "published"): ("partial", "state is one True/False token per step (1 bit); knockout curve is flat, so it is unclear the value chain is read at all; no propagation measured"),
    ("entity_tracking_boxes", "published"): ("partial", "delta format: each step lists only the boxes the operation touched, so the full 94.6-bit state is never written in one step (bits per state token left blank); no propagation measured"),
    ("blocksworld", "published"): ("partial", "the step restates the previous state (Current State) and the action's preconditions in prose; only the Resulting State line counted; no propagation measured"),
    ("cellular_automaton", "cells"): ("yes", "the running row is rewritten after every cell (9 copies of the 8-cell row per generation); the full count is an upper bound, the canonical row: line alone is the floor"),
    ("hanoi", "execute"): ("yes", "each state is recomputable from the prompt's move list, so the model can route around a corrupted state (propagation 0.33); density reflects that redundancy, not a bad measurement"),
    ("turing_machine", "published"): ("yes", "desk.json has no state_bits (queue length unbounded), so no bits per state token"),
}

def apply_judgement(rows):
    for rec, notes in rows:
        w = WILD.get((rec["slug"], rec["format"]))
        if w:
            rec["estimable"] = w[0] if rec["estimable"] != "no" or w[0] == "no" else rec["estimable"]
            if w[0] == "no": rec["density_est"] = ""
            if rec["slug"] == "entity_tracking_boxes": rec["bits_per_state_token"] = ""
            notes.append(w[1])
        if rec["format"] == "published" and rec["slug"] in {"s5_composition", "random_lookup_table", "threesum", "addition"}:
            notes.append("published (SFT) format for side-by-side; batch 3 ran the ergonomic format, so no propagation")
        notes.append(rec["_spread"])
        rec["note"] = "; ".join(notes)

COLS = ["slug", "format", "depth", "n", "step_tokens_mean", "state_tokens_mean", "state_frac", "mean_propagation",
        "density_est", "bits_per_step", "bits_per_state_token", "estimable", "note"]

def write(rows, examples, tok_name):
    (RES / "tidy").mkdir(exist_ok=True)
    with open(RES / "tidy" / "density.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
        for rec, _ in rows: w.writerow(rec)
    def disp(rec):
        s = rec["slug"]
        from stateful.registry import display_name
        return display_name(s)
    L = ["# Load-bearing token density — ballpark from the trace format", "",
         "## Method", "",
         "Density = share of CoT tokens whose removal would change the answer. Not measured directly (needs attention masking);",
         "estimated from the format. For each gold step (split exactly as batch 3's `gold_prefix`), tokens are counted with the",
         f"`{tok_name}` tokenizer on `format_cot` as it sits in the assistant turn, each token assigned to the step its first character",
         "falls in. `state_tokens` = tokens overlapping the step's state field(s): a per-task regex (`density.py: SPANS`) taking the",
         "whole value of every field that reports the post-step state (labels such as `stack:` excluded). For the ten tasks with",
         "`corrupt_step`, every character the corruption rewrites (k = 1, d/2, d; 30 instances) must fall inside an extracted span:",
         "coverage column below. `state_frac` = mean state tokens / mean step tokens is an upper bound on density (all state",
         "tokens load-bearing, all scaffolding not). `density_est` = state_frac x mean propagation (BENCH3.md: share of answers",
         "a corrupted step changes). bits/step from `desk.json` (hanoi rescaled to the disks actually used). n = 30 instances per",
         "row, `instance_seed(0, depth, i)`, depths and knobs from `plans/batch3.json` (batch-2 plan for tasks not in batch 3).",
         "Where a format writes the state twice, the canonical-copy-only figure is given too.", "",
         "## Table", "",
         "| task | format | depth | step tok | state tok | state_frac | canonical copy (tok / frac) | mean prop. | density_est | bits/step | bits/state tok | estimable | corrupt edits in span | step-length spread |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rec, _ in rows:
        L.append("| " + " | ".join(str(x) if x != "" else "·" for x in [
            disp(rec), rec["format"], rec["depth"], rec["step_tokens_mean"], rec["state_tokens_mean"], rec["state_frac"],
            rec["_one"], rec["mean_propagation"], rec["density_est"], rec["bits_per_step"], rec["bits_per_state_token"],
            rec["estimable"], rec["_cov"], rec["_spread"].replace("step tokens: ", "")]) + " |")
    L += ["", "## What was counted as state", "", "Step 2 of instance 0 per row, counted state in `⟦ ⟧`; for multi-line steps, the first 2 lines that carry state.", ""]
    for rec, _ in rows:
        ex = examples.get((rec["slug"], rec["format"]))
        if not ex: continue
        s, sp = ex[min(1, len(ex) - 1)]
        lines = [l for l in bracket(s, sp).splitlines() if "⟦" in l][:2]
        L.append(f"- **{disp(rec)} ({rec['format']})**: " + " / ".join(f"`{l.strip()[:160]}`" for l in lines))
    L += ["", "## Too wild to estimate", ""]
    for (slug, fmt), (est, why) in WILD.items():
        if est in ("no", "partial"):
            L.append(f"- **{slug} ({fmt})**: {why}.")
    L.append("- **published † formats (s5_composition, random_lookup_table, threesum, addition)**: state_frac is real, but no propagation was measured in these formats (batch 3 ran the ergonomic ones), and batch 1 showed they do not transfer by prompting.")
    L += ["", "## Estimable, with caveats", ""]
    for (slug, fmt), (est, why) in WILD.items():
        if est == "yes":
            L.append(f"- **{slug} ({fmt})**: {why}.")
    L.append("- **s5_composition, random_lookup_table (ergonomic)**: the ergonomic step writes the new state twice (spaced and packed; mapping result and `Now at`); both copies are counted. The canonical-copy column gives the single-copy floor.")
    (RES / "DENSITY.md").write_text("\n".join(L) + "\n")

if __name__ == "__main__":
    rows, examples, tok_name = run()
    if "--examples" in sys.argv:
        for (slug, fmt), ex in examples.items():
            print(f"== {slug} ({fmt})")
            for s, sp in ex:
                for l in bracket(s, sp).splitlines()[:3]: print("   ", l[:170])
    write(rows, examples, tok_name)
    for rec, _ in rows:
        print(rec["slug"], rec["format"], rec["step_tokens_mean"], rec["state_tokens_mean"], rec["state_frac"],
              rec["density_est"], rec["_cov"], rec["estimable"])
