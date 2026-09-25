"""Tidy, chart-ready CSVs for the stateful-tasks post. Computes from the Inspect logs; the markdown tables are
only cross-checked (results/tidy/crosscheck.csv). Idempotent: rewrites results/tidy/*.csv every run.

    uv run python tidy.py

Outputs (results/tidy/): depth_sweep.csv, survival.csv, batch2.csv, batch3.csv, batch3_summary.csv, tasks.csv,
master.csv, crosscheck.csv. README.md there documents the format labels and judgement calls.
"""
from __future__ import annotations
import ast, csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inspect_ai.log import list_eval_logs, read_eval_log
from stateful.registry import display_name, implemented, DAGGER, ASTERISK, DROPPED, SWEEP_KNOBS
from stateful.traces import MODEL_LOGS

RES = HERE / "results"; LOGS = RES / "logs"; OUT = RES / "tidy"; ROOT = HERE.parent
GAP, P, ACC = 0.2, 0.05, 0.7
EPS = 1e-9
MODELS = ["qwen9b", "llama70b", "ds", "dspro"]
SUMMARY_MODELS = ["qwen9b", "llama70b", "ds"]          # the batch-1 roster SUMMARY.md reports
MODEL_STR = {tag: v[0] for tag, v in MODEL_LOGS.items()}
FORMATS = ["published", "ours", "ergonomic", "execute", "cells2", "cells3"]

# ---- reuse collect.py's exact McNemar without executing collect.py's argparse/script body -------------------
_src = ast.parse((HERE / "collect.py").read_text())
_fn = next(n for n in _src.body if isinstance(n, ast.FunctionDef) and n.name == "mcnemar_exact")
_ns = {"comb": math.comb}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), str(HERE / "collect.py"), "exec"), _ns)
mcnemar_exact = _ns["mcnemar_exact"]

# ---- which log dirs hold which (model, format) ---------------------------------------------------------------
# kind -> (slug -> format label). Sample ids do not encode format (hanoi/cot/d4/s... is the same id under the
# plan and execute formulations), so dedupe happens only WITHIN a group, never across groups.
def b1_format(slug): return "ours" if slug in ASTERISK else "published"
KIND_FORMAT = {
    "batch1":  b1_format,
    "second":  lambda s: {"hanoi": "execute", "cellular_automaton": "cells2"}[s],
    "second3": lambda s: {"cellular_automaton": "cells3"}[s],
    "erg":     lambda s: "ergonomic",
}
GROUPS = [  # (model tag, log dirs in override order, kind). Batch-1 dirs mirror stateful/traces.py MODEL_LOGS.
    ("qwen9b",   ["batch1", "batch1b-9b"],       "batch1"),
    ("llama70b", ["batch1-70b"],                 "batch1"),
    ("ds",       ["batch1-ds", "batch1-ds-cap"], "batch1"),
    ("dspro",    ["batch1-dspro"],               "batch1"),
    ("qwen9b",   ["batch1-2nd-9b"],  "second"),  ("ds", ["batch1-2nd-ds"],  "second"),
    ("qwen9b",   ["batch1-2nd3-9b"], "second3"), ("ds", ["batch1-2nd3-ds"], "second3"),
    ("qwen9b",   ["batch1-erg-9b"],  "erg"),     ("ds", ["batch1-erg-ds"],  "erg"),
]
for tag, dirs, kind in GROUPS:
    if kind == "batch1":
        assert [f"results/logs/{d}" for d in dirs] == MODEL_LOGS[tag][1], (tag, dirs)

def prompt_marker(sample) -> str:
    """Classify the trace format from the few-shot assistant turns (no trace text is printed)."""
    txt = " ".join(m.text for m in (sample.input if isinstance(sample.input, list) else []) if m.role == "assistant")
    if "row so far" in txt: return "cells3"
    if "cell 1:" in txt: return "cells2"
    if "move 1:" in txt: return "execute"
    if "moves = [" in txt: return "plan"
    if "generation 1:" in txt: return "rows"
    return "?"
EXPECTED_MARKER = {("hanoi", "published"): "plan", ("hanoi", "execute"): "execute",
                   ("cellular_automaton", "ours"): "rows", ("cellular_automaton", "cells2"): "cells2",
                   ("cellular_automaton", "cells3"): "cells3"}

checks = []   # crosscheck rows: (table, key, field, expected(markdown/csv), got(logs), class)
def note(table, key, field, expected, got, cls):
    checks.append([table, key, field, expected, got, cls])

# ---- load batch-1-style logs --------------------------------------------------------------------------------
def load_group(tag, dirs, kind):
    seen, unscored = {}, defaultdict(set)   # id -> (md, score, dir) ; later dirs override earlier (collect.py rule)
    superseded = defaultdict(set)
    for d in dirs:
        for info in sorted(list_eval_logs(str(LOGS / d)), key=lambda i: i.name):
            log = read_eval_log(info)
            if log.status != "success" or not log.samples: continue
            assert log.eval.model == MODEL_STR[tag], (d, log.eval.model)
            slug = log.eval.metadata["slug"]; fmt = KIND_FORMAT[kind](slug)
            knobs = (log.eval.task_args or {}).get("knobs") or {}
            knobs = json.loads(knobs) if isinstance(knobs, str) else knobs
            want = {"format": "ergonomic"} if kind == "erg" else (SWEEP_KNOBS.get(slug, {}) if kind == "batch1" else {})
            assert knobs == want, (d, slug, knobs, want)
            if (slug, fmt) in EXPECTED_MARKER and log.eval.metadata["condition"] == "cot":   # nocot shots carry no trace
                got = prompt_marker(log.samples[0])
                assert got == EXPECTED_MARKER[(slug, fmt)], (d, slug, fmt, got)
            for s in log.samples:
                sc = s.scores.get("task_check") if s.scores else None
                if sc is None: unscored[(slug, fmt)].add(s.id); continue
                if s.id in seen and seen[s.id][2] != d: superseded[s.id].add(seen[s.id][2])
                seen[s.id] = (s.metadata, sc, d, len(s.output.completion or "") if s.output else 0)
    for key in unscored: unscored[key] -= set(seen)
    return seen, superseded, unscored

def stats(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

sweep = {}   # (slug, fmt, model, depth) -> row dict
unscored_total = defaultdict(int)
for tag, dirs, kind in GROUPS:
    seen, superseded, unscored = load_group(tag, dirs, kind)
    for (slug, fmt), ids in unscored.items():
        for i in ids: unscored_total[(slug, fmt, tag, int(i.split("/d")[1].split("/")[0]))] += 1
    cells = defaultdict(lambda: {"cot": {}, "nocot": {}, "src": set(), "sup": set()})
    for sid, (md, sc, d, nchars) in seen.items():
        fmt = KIND_FORMAT[kind](md["slug"])
        c = cells[(md["slug"], fmt, tag, md["depth"])]
        c[md["condition"]][md["inst_seed"]] = (sc.value == "C", {**(sc.metadata or {}), "_chars": nchars, "_gold": md.get("gold_cot_chars")})
        c["src"].add(d); c["sup"] |= superseded.get(sid, set())
    for key, c in cells.items():
        cot, noc = c["cot"], c["nocot"]
        def acc(arm): return None if not arm else sum(v[0] for v in arm.values()) / len(arm)
        def se(arm, p): return None if not arm else math.sqrt(p * (1 - p) / len(arm))
        ac, an = acc(cot), acc(noc)
        common = set(cot) & set(noc)
        b = sum(cot[i][0] and not noc[i][0] for i in common); cc = sum(noc[i][0] and not cot[i][0] for i in common)
        pv = mcnemar_exact(b, cc) if (cot and noc) else None
        tok_c = stats([v[1].get("output_tokens") for v in cot.values()])
        allm = [v[1] for v in list(cot.values()) + list(noc.values())]
        sweep[key] = dict(
            slug=key[0], format=key[1], model=key[2], depth=key[3],
            n=max(len(cot), len(noc)), acc_cot=ac, se_cot=se(cot, ac), acc_nocot=an, se_nocot=se(noc, an),
            gap=None if ac is None or an is None else ac - an, p_mcnemar=pv, n_paired=len(common),
            tok_out_cot_mean=tok_c, tok_out_nocot_mean=stats([v[1].get("output_tokens") for v in noc.values()]),
            tok_per_step=None if tok_c is None else tok_c / key[3],
            truncated_rate_cot=None if not cot else sum(bool(v[1].get("truncated")) for v in cot.values()) / len(cot),
            hidden_reasoning_rate=sum(bool(m.get("hidden_reasoning")) for m in allm) / len(allm),
            n_cot=len(cot), n_nocot=len(noc), b_cot_only=b, c_nocot_only=cc,
            leak_rate_nocot=None if not noc else sum(bool(v[1].get("leaked_reasoning")) for v in noc.values()) / len(noc),
            no_answer_rate_cot=None if not cot else sum(not v[1].get("has_answer_line") for v in cot.values()) / len(cot),
            n_unscored=unscored_total.get(key, 0),
            cot_lines_per_step=None if not cot else stats([v[1].get("n_lines") for v in cot.values()]) / key[3],
            cot_len_ratio_vs_gold=None if not cot else stats([v[1]["_chars"] for v in cot.values()]) / stats([v[1]["_gold"] for v in cot.values()]),
            sources="+".join(d for d in dirs if d in c["src"]),
            superseded_sources="+".join(d for d in dirs if d in c["sup"]),
        )

def fnum(x, nd=4):
    if x is None: return ""
    if isinstance(x, bool): return str(x)
    if isinstance(x, float): return f"{x:.3e}" if 0 < abs(x) < 1e-3 else f"{x:.{nd}f}"
    return str(x)

def write(name, header, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow([fnum(r.get(h)) for h in header] if isinstance(r, dict) else [fnum(x) for x in r])
    return len(rows)

SW_COLS = ["slug", "format", "model", "depth", "n", "acc_cot", "se_cot", "acc_nocot", "se_nocot", "gap", "p_mcnemar",
           "n_paired", "tok_out_cot_mean", "tok_out_nocot_mean", "tok_per_step", "truncated_rate_cot",
           "hidden_reasoning_rate", "sources", "superseded_sources", "n_cot", "n_nocot", "b_cot_only", "c_nocot_only",
           "leak_rate_nocot", "no_answer_rate_cot", "n_unscored", "cot_lines_per_step", "cot_len_ratio_vs_gold"]
order = lambda k: (k[0], FORMATS.index(k[1]), MODELS.index(k[2]), k[3])
sweep_rows = [sweep[k] for k in sorted(sweep, key=order)]
counts = {"depth_sweep.csv": write("depth_sweep.csv", SW_COLS, sweep_rows)}

# ---- survival -------------------------------------------------------------------------------------------------
def passes(r): return (r["gap"] is not None and round(r["gap"], 9) >= GAP - EPS and r["p_mcnemar"] is not None
                       and r["p_mcnemar"] < P and round(r["acc_cot"], 9) >= ACC - EPS)
surv = {}
for (slug, fmt, model) in sorted({k[:3] for k in sweep}, key=lambda k: (k[0], FORMATS.index(k[1]), MODELS.index(k[2]))):
    cells = [sweep[k] for k in sweep if k[:3] == (slug, fmt, model) and sweep[k]["gap"] is not None]
    cells.sort(key=lambda r: r["depth"])
    ok = [r for r in cells if passes(r)]
    # ties on gap (to 2 dp, as displayed) -> DEEPEST depth, matching summarize2.py and plans/batch2.json
    pick = lambda rs: max(rs, key=lambda r: (round(r["gap"], 2), r["depth"]))
    s = dict(slug=slug, format=fmt, model=model, passes=bool(ok), n_depths=len(cells))
    if ok:
        w, b, dp = ok[0], pick(ok), ok[-1]
        s.update(working_depth=w["depth"], best_depth=b["depth"], best_gap=b["gap"], best_acc_cot=b["acc_cot"],
                 best_acc_nocot=b["acc_nocot"], best_p=b["p_mcnemar"], deepest_passing_depth=dp["depth"],
                 deepest_acc_cot=dp["acc_cot"], tok_per_step_at_best=b["tok_per_step"],
                 verdict_text=f"d{w['depth']} / d{b['depth']} ({b['acc_cot']:.2f}, {b['gap']:.2f})")
    elif cells:
        b = pick(cells)
        sig = [r for r in cells if round(r["gap"], 9) >= GAP - EPS]
        reason = ("acc<0.7" if any(r["p_mcnemar"] < P for r in sig) else "p≥0.05") if sig else "no gap"
        s.update(best_depth=b["depth"], best_gap=b["gap"], best_acc_cot=b["acc_cot"], best_acc_nocot=b["acc_nocot"],
                 best_p=b["p_mcnemar"], tok_per_step_at_best=b["tok_per_step"], fail_reason=reason,
                 verdict_text=f"— ({reason}; best gap {b['gap']:.2f} at d{b['depth']}, acc {b['acc_cot']:.2f})")
    else:
        s.update(verdict_text="· (no paired cot/nocot cell)")
    if "best_depth" in s:
        s["second_chance_text"] = (f"{'PASS' if ok else 'fail'} d{s['best_depth']}: {s['best_acc_cot']:.2f} / "
                                   f"{s['best_acc_nocot']:.2f} / {s['best_gap']:+.2f}")
    surv[(slug, fmt, model)] = s

def task_verdict(slug, fmt):
    roster = SUMMARY_MODELS if fmt in ("published", "ours") else ["qwen9b", "ds"]
    ran = [m for m in roster if (slug, fmt, m) in surv and surv[(slug, fmt, m)]["n_depths"]]
    alive = sum(surv[(slug, fmt, m)]["passes"] for m in ran)
    if slug in DROPPED: return "DROPPED: " + DROPPED[slug]
    if ran and alive == len(ran): return "survives (all models)"
    if alive: return f"survives ({alive} model{'s' if alive > 1 else ''})"
    if slug in DAGGER: return "passes through with † (" + DAGGER[slug] + ")"
    if slug in ASTERISK: return "passes through with * (" + ASTERISK[slug] + ")"
    return "fails batch 1" if fmt in ("published", "ours") else "fails"
for (slug, fmt, model), s in surv.items(): s["task_verdict"] = task_verdict(slug, fmt)

SV_COLS = ["slug", "format", "model", "passes", "working_depth", "best_depth", "best_gap", "best_acc_cot", "best_acc_nocot",
           "deepest_passing_depth", "deepest_acc_cot", "tok_per_step_at_best", "verdict_text", "best_p", "fail_reason",
           "second_chance_text", "task_verdict", "n_depths"]
counts["survival.csv"] = write("survival.csv", SV_COLS, list(surv.values()))

# ---- cross-check vs SUMMARY.md and SECOND_CHANCE.md ----------------------------------------------------------
def md_rows(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("| ") and not line.startswith("| task") and not line.startswith("|---"):
            out.append([x.strip() for x in line.strip().strip("|").split("|")])
    return out
for row in md_rows(RES / "SUMMARY.md"):
    name, cells, verdict = row[0], row[1:4], row[4]
    slug = name.rstrip("*†")
    if name != display_name(slug): note("SUMMARY.md", slug, "task marker", name, display_name(slug), "stale marker (pre-AMENDMENT 5 registry)")
    for m, exp in zip(SUMMARY_MODELS, cells):
        s = surv.get((slug, b1_format(slug), m)); got = s["verdict_text"] if s else "·"
        if exp == got: continue
        cls = "number/verdict"
        if s and s["passes"] and exp.startswith("d"):
            # same working depth and same best numbers but a different tied best depth?
            ok = [sweep[k] for k in sweep if k[:3] == (slug, b1_format(slug), m) and sweep[k]["gap"] is not None and passes(sweep[k])]
            m_ = re.match(r"d(\d+) / d(\d+) \(([\d.]+), ([\d.]+)\)", exp)
            if m_ and any(r["depth"] == int(m_.group(2)) and f"{r['gap']:.2f}" == m_.group(4) and f"{r['acc_cot']:.2f}" == m_.group(3) for r in ok) \
                    and int(m_.group(1)) == s["working_depth"]:
                cls = "tie-break only (SUMMARY takes shallowest of equal gaps; tidy takes deepest)"
        elif s and not s["passes"] and exp.startswith("—"):
            m_ = re.match(r"— \((.*?); best gap (-?[\d.]+) at d(\d+), acc ([\d.]+)\)", exp)
            if m_ and m_.group(1) == s.get("fail_reason") and f"{s['best_gap']:.2f}" == m_.group(2):
                cls = "tie-break only (SUMMARY takes shallowest of equal gaps; tidy takes deepest)"
        note("SUMMARY.md", f"{slug}/{m}", "cell", exp, got, cls)
    got_v = task_verdict(slug, b1_format(slug))
    if verdict != got_v:
        cls = "marker wording (pre-AMENDMENT 5)" if verdict.split("(")[0].strip()[:18] == got_v.split("(")[0].strip()[:18] else "number/verdict"
        note("SUMMARY.md", slug, "verdict", verdict, got_v, cls)
VARIANT_FMT = {"execution formulation": "execute", "per-cell format (rev 2)": "cells2",
               "per-cell + named neighbours + running row (rev 3)": "cells3", "ergonomic (SFT-free) format": "ergonomic"}
for row in md_rows(RES / "SECOND_CHANCE.md"):
    slug = row[0].rstrip("*†"); vf = VARIANT_FMT[row[1]]
    for m, fmt, exp in [("qwen9b", b1_format(slug), row[2]), ("qwen9b", vf, row[3]), ("ds", b1_format(slug), row[4]), ("ds", vf, row[5])]:
        s = surv.get((slug, fmt, m)); got = s.get("second_chance_text", "·") if s else "·"
        if exp != got: note("SECOND_CHANCE.md", f"{slug}/{fmt}/{m}", "cell", exp, got, "number/verdict")

# ---- cross-check vs per-dir batch1.csv / BENCH1.md ---------------------------------------------------------------
RESDIR = {"qwen35-9b": ("qwen9b", "batch1"), "llama33-70b": ("llama70b", "batch1"), "deepseek-v4-flash": ("ds", "batch1"),
          "deepseek-v4-pro": ("dspro", "batch1"), "second-chance-9b": ("qwen9b", "second"), "second-chance-ds": ("ds", "second"),
          "second-chance3-9b": ("qwen9b", "second3"), "second-chance3-ds": ("ds", "second3"),
          "ergonomic-9b": ("qwen9b", "erg"), "ergonomic-ds": ("ds", "erg")}
for rd, (tag, kind) in RESDIR.items():
    for r in csv.DictReader(open(RES / rd / "batch1.csv")):
        key = (r["slug"], KIND_FORMAT[kind](r["slug"]), tag, int(r["depth"])); s = sweep.get(key)
        arm = r["condition"]
        if not s or not s[f"n_{arm}"]: note(f"{rd}/batch1.csv", f"{key}", arm, r["n"], "missing", "cell missing in logs"); continue
        if int(r["n"]) != s[f"n_{arm}"] or abs(float(r["acc"]) - s[f"acc_{arm}"]) > 0.0006:
            note(f"{rd}/batch1.csv", "/".join(map(str, key)), f"{arm} n/acc", f"{r['n']}/{r['acc']}", f"{s[f'n_{arm}']}/{s[f'acc_{arm}']:.3f}", "number")
    for c in md_rows(RES / rd / "BENCH1.md"):
        if len(c) < 8: continue
        slug = c[0].rstrip("*†"); key = (slug, KIND_FORMAT[kind](slug), tag, int(c[1])); s = sweep.get(key)
        if not s: continue
        pv = s["p_mcnemar"]; mine = "·" if pv is None else (f"{pv:.3f}" if pv >= 0.001 else "<.001")
        if c[7] != mine: note(f"{rd}/BENCH1.md", "/".join(map(str, key)), "p", c[7], mine, "number")

# ---- batch 2 --------------------------------------------------------------------------------------------------
plan2 = json.load(open(HERE / "plans/batch2.json"))
b2_csv = list(csv.DictReader(open(RES / "batch2.csv")))
# recompute from logs with collect2.py's rule (batch2 then batch2c2 overriding) and cross-check the published csv
seen2 = {}
for d in ["batch2", "batch2c2"]:
    for info in sorted(list_eval_logs(str(LOGS / d)), key=lambda i: i.name):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples: continue
        for s in log.samples:
            sc = s.scores.get("b2_check") if s.scores else None
            if sc is not None: seen2[s.id] = (s.metadata, sc)
cells2 = defaultdict(lambda: [0, 0])
for md, sc in seen2.values():
    sub = md.get("fraction") if md["condition"] == "knockout" else (f"k{md['k']}{'r' if md['redacted'] else 'p'}" if md["condition"] == "continue" else "")
    cells2[(md["slug"], md["condition"], sub)][0] += 1; cells2[(md["slug"], md["condition"], sub)][1] += (sc.value == "C")
def a2(slug, cond, sub=""):
    n, c = cells2.get((slug, cond, sub), (0, 0)); return None if not n else c / n
MODEL_NORM = {"qwen9b": "qwen9b", "qwen35-9b": "qwen9b", "llama70b": "llama70b", "ds": "ds", "dspro": "dspro"}
b2_rows = []
for r in b2_csv:
    slug = r["slug"]
    ks = sorted({k[2] for k in cells2 if k[0] == slug and k[1] == "continue"})
    recomputed = {"ko0": a2(slug, "knockout", 0.0), "ko25": a2(slug, "knockout", 0.25), "ko50": a2(slug, "knockout", 0.5),
                  "ko75": a2(slug, "knockout", 0.75), "filler": a2(slug, "filler"),
                  "cont_redacted": next((a2(slug, "continue", k) for k in ks if k.endswith("r")), None),
                  "cont_plain": next((a2(slug, "continue", k) for k in ks if k.endswith("p")), None)}
    for f_, v in recomputed.items():
        if (r[f_] or None) is None and v is None: continue
        if r[f_] == "" or v is None or abs(float(r[f_]) - v) > 1e-6:
            note("batch2.csv", slug, f_, r[f_], fnum(v), "number")
    # the published batch-2 acc_cot/acc_nocot are batch-1 numbers; check them against the tidy sweep
    s = sweep.get((slug, b1_format(slug), MODEL_NORM[r["model"]], int(r["depth"])))
    for arm in ("cot", "nocot"):
        if s and abs(float(r[f"acc_{arm}"]) - s[f"acc_{arm}"]) > 1e-6:
            note("batch2.csv", slug, f"acc_{arm}", r[f"acc_{arm}"], f"{s[f'acc_{arm}']:.3f}", "number")
    out = dict(r); out["model"] = MODEL_NORM[r["model"]]; out["format"] = b1_format(slug)
    out["filler_recovery"] = f"{float(r['filler']) - float(r['acc_nocot']):.2f}" if r["filler"] else ""
    out["cont_k"] = next((k[1:-1] for k in ks if k.endswith("r")), "")
    b2_rows.append(out)
B2_COLS = ["slug", "format", "model", "depth", "n", "acc_cot", "acc_nocot", "ko0", "ko25", "ko50", "ko75", "filler",
           "filler_recovery", "cont_redacted", "cont_plain", "cont_k", "kept_reasoning_ko50"]
counts["batch2.csv"] = write("batch2.csv", B2_COLS, b2_rows)
b2 = {r["slug"]: r for r in b2_rows}

# knockout curve shape (batch 2, own trace, line-fraction cuts). Classification as written in the post notes
# (vault "2026/comparative review of stateful tasks post.md", Batch 2 bullet); entity_tracking_boxes was not
# classified there. Verified below against the numbers: decided_at_end = ko@.75 within 0.15 of no-CoT.
KO_SHAPE = {"nested_arithmetic": "decided_at_end", "turing_machine": "decided_at_end", "s5_composition": "decided_at_end",
            "synthetic_program_trace": "decided_at_end", "dyck": "decided_at_end", "random_lookup_table": "decided_at_end",
            "cup_shuffling": "decided_at_end", "cruxeval": "early", "blocksworld": "early", "addition": "early",
            "boolean_expressions": "flat", "entity_tracking_boxes": "unclassified"}
for slug, shape in KO_SHAPE.items():
    r = b2[slug]; near_floor = float(r["ko75"]) <= float(r["acc_nocot"]) + 0.15
    if (shape == "decided_at_end") != near_floor and shape != "unclassified" and slug != "boolean_expressions":
        note("knockout_shape", slug, "shape", shape, f"ko75={r['ko75']} nocot={r['acc_nocot']}", "classification vs numbers")

# ---- batch 3 --------------------------------------------------------------------------------------------------
plan3 = json.load(open(HERE / "plans/batch3.json"))
B3_FORMAT = {"hanoi": "execute", "cellular_automaton": "cells3"}   # task defaults at batch-3 run time (verified by prefill marker)
def b3_format(slug):
    kn = plan3[slug].get("knobs", {})
    return kn.get("format") or B3_FORMAT.get(slug) or b1_format(slug)
seen3 = {}
for info in sorted(list_eval_logs(str(LOGS / "batch3")), key=lambda i: i.name):
    log = read_eval_log(info)
    if log.status != "success" or not log.samples: continue
    slug = log.eval.metadata["slug"]
    assert log.eval.model == MODEL_STR[plan3[slug]["model"]]
    kn = log.samples[0].metadata.get("knobs") or {}
    assert kn == plan3[slug].get("knobs", {}), (slug, kn)
    if slug in B3_FORMAT:   # prefill is the gold trace; check it is in the format we label
        pre = [m for m in log.samples[0].input if m.role == "assistant"][-1].text
        mk = "cells3" if "row so far" in pre else ("execute" if "move 1:" in pre else "?")
        if log.eval.metadata["condition"] != "goldko" or log.samples[0].metadata.get("k"):
            assert mk == B3_FORMAT[slug], (slug, mk)
    for s in log.samples:
        sc = s.scores.get("b2_check") if s.scores else None
        if sc is not None: seen3[s.id] = (s.metadata, sc)
cells3 = defaultdict(lambda: [0, 0]); depth3 = {}
for md, sc in seen3.values():
    key = (md["slug"], md["condition"], md["k"]); cells3[key][0] += 1; cells3[key][1] += (sc.value == "C")
    depth3[md["slug"]] = md["depth"]
COND = {"goldko": "goldko", "mistake": "mistake", "continue": "cont"}
b3_rows = []
for (slug, cond, k), (n, c) in sorted(cells3.items(), key=lambda x: (list(plan3).index(x[0][0]), x[0][1], x[0][2])):
    d = depth3[slug]; p_ = c / n
    b3_rows.append(dict(slug=slug, format=b3_format(slug), model=plan3[slug]["model"], depth=d, condition=COND[cond],
                        k=k, k_frac=k / d, n=n, acc=p_, se=math.sqrt(p_ * (1 - p_) / n)))
counts["batch3.csv"] = write("batch3.csv", ["slug", "format", "model", "depth", "condition", "k", "k_frac", "n", "acc", "se"], b3_rows)
def a3(slug, cond, k):
    n, c = cells3.get((slug, cond, k), (0, 0)); return None if not n else c / n
b3s = {}
for slug, p in plan3.items():
    d = p["depth"]; assert depth3.get(slug) == d, (slug, depth3.get(slug), d)
    kq = [max(1, round(d * x)) for x in (0.25, 0.5, 0.75)]           # collect3.py's k columns (may repeat at small d)
    props = [a3(slug, "continue", k) - a3(slug, "mistake", k) for k in kq]
    conts = [a3(slug, "continue", k) for k in kq]
    b3s[slug] = dict(slug=slug, format=b3_format(slug), model=p["model"], depth=d, k_q1=kq[0], k_half=kq[1], k_q3=kq[2],
                     goldko_k0=a3(slug, "goldko", 0), goldko_q1=a3(slug, "goldko", kq[0]), goldko_half=a3(slug, "goldko", kq[1]),
                     goldko_q3=a3(slug, "goldko", kq[2]), goldko_dm1=a3(slug, "goldko", d - 1),
                     prop_q1=props[0], prop_half=props[1], prop_q3=props[2], mean_propagation=sum(props) / 3,
                     cont_mean=sum(conts) / 3,
                     mistake_q1=a3(slug, "mistake", kq[0]), mistake_half=a3(slug, "mistake", kq[1]), mistake_q3=a3(slug, "mistake", kq[2]),
                     cont_q1=conts[0], cont_half=conts[1], cont_q3=conts[2], n_per_cell=cells3[(slug, "goldko", 0)][0])
B3S_COLS = ["slug", "format", "model", "depth", "goldko_k0", "goldko_q1", "goldko_half", "goldko_q3", "goldko_dm1",
            "prop_q1", "prop_half", "prop_q3", "mean_propagation", "cont_mean", "k_q1", "k_half", "k_q3",
            "mistake_q1", "mistake_half", "mistake_q3", "cont_q1", "cont_half", "cont_q3", "n_per_cell"]
counts["batch3_summary.csv"] = write("batch3_summary.csv", B3S_COLS, list(b3s.values()))
for c in md_rows(RES / "BENCH3.md"):   # cross-check the markdown
    slug = c[0].rstrip("*†"); s = b3s[slug]
    mine = [s["goldko_k0"], s["goldko_q1"], s["goldko_half"], s["goldko_q3"], s["goldko_dm1"]]
    for lab, e, g in zip(["goldko_k0", "goldko_q1", "goldko_half", "goldko_q3", "goldko_dm1"], c[2:7], mine):
        if e != f"{g:.2f}": note("BENCH3.md", slug, lab, e, f"{g:.2f}", "number")
    if c[10] != f"{s['mean_propagation']:.2f}": note("BENCH3.md", slug, "mean_propagation", c[10], f"{s['mean_propagation']:.2f}", "number")
    # the 9B batch-3 depth is meant to be the deepest passing 9B depth in the batch-3 format
    sv = surv.get((slug, s["format"], "qwen9b"))
    if sv and sv.get("deepest_passing_depth") != s["depth"]:
        note("plans/batch3.json", slug, "depth vs deepest passing 9B depth", s["depth"], sv.get("deepest_passing_depth"), "plan choice")

# ---- desk metrics -----------------------------------------------------------------------------------------------
desk_md = {}; knobs_md = {}
lines = (ROOT / "DESK.md").read_text(encoding="utf-8").splitlines()
hdr = None
for ln in lines:
    if ln.startswith("| task"): hdr = [x.strip() for x in ln.strip().strip("|").split("|")]; continue
    if hdr and ln.startswith("| ") and not ln.startswith("|---"):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]; desk_md[c[0]] = dict(zip(hdr, c))
    m = re.match(r"- \*\*(\w+)\*\*: (.*?) / ", ln)
    if m and "tokens flat" in ln: knobs_md[m.group(1)] = m.group(2)
HARNESS_DEPTH = {  # what `depth` counts in tasks/<slug>/task.py (the swept knob; not always DESK.md's published knob)
    "addition": "digit columns", "blocksworld": "actions executed", "boolean_expressions": "named sub-expression resolutions",
    "cellular_automaton": "generations", "cruxeval": "executed statements", "cup_shuffling": "pairwise swaps",
    "dyck": "input symbols (stack updates)", "entity_tracking_boxes": "box operations", "hanoi": "moves",
    "multiplication": "numbered steps = digits_y*(digits_x+1)", "nested_arithmetic": "sub-expression reductions",
    "random_lookup_table": "table applications", "s5_composition": "permutations composed",
    "synthetic_program_trace": "executed statements", "threesum": "candidate-triple checks in the gold trace",
    "turing_machine": "tag-system rewrite steps"}
SOURCE = {  # slug -> (source_short, source_title, trace_source_short). Titles from tasks/<slug>/desk.json primary_source.
    "addition": ("Nye et al. 2021", "Show Your Work: Scratchpads for Intermediate Computation with Language Models", "Nye et al. 2021"),
    "blocksworld": ("Stechly et al. 2024", "Chain of Thoughtlessness? An Analysis of CoT in Planning", "Stechly et al. 2024"),
    "boolean_expressions": ("Suzgun et al. 2022", "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", "Suzgun et al. 2022"),
    "cellular_automaton": ("Neary & Woods 2006", "P-completeness of Cellular Automaton Rule 110", "none (format is ours)"),
    "cruxeval": ("Gu et al. 2024", "CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution", "Gu et al. 2024"),
    "cup_shuffling": ("Suzgun et al. 2022", "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", "Suzgun et al. 2022"),
    "dyck": ("Suzgun et al. 2022", "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", "Suzgun et al. 2022"),
    "entity_tracking_boxes": ("Kim & Schuster 2023", "Entity Tracking in Language Models", "none (format is ours)"),
    "hanoi": ("Shojaee et al. 2025", "The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity", "Shojaee et al. 2025"),
    "multiplication": ("Dziri et al. 2023", "Faith and Fate: Limits of Transformers on Compositionality", "Dziri et al. 2023"),
    "nested_arithmetic": ("Suzgun et al. 2022", "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them", "Suzgun et al. 2022"),
    "random_lookup_table": ("Ramesh et al. 2024", "Compositional Capabilities of Autoregressive Transformers: A Study on Synthetic, Interpretable Tasks", "Ramesh et al. 2024"),
    "s5_composition": ("Liu et al. 2022", "Transformers Learn Shortcuts to Automata", "Li et al. 2025, (How) Do Language Models Track State?"),
    "synthetic_program_trace": ("Nye et al. 2021", "Show Your Work: Scratchpads for Intermediate Computation with Language Models", "Nye et al. 2021"),
    "threesum": ("Pfau et al. 2024", "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models", "Pfau et al. 2024"),
    "turing_machine": ("Wu et al. 2025", "Computational Reasoning of Large Language Models (TMBench)", "Wu et al. 2025"),
}
STATE = {  # <= 8 words: what the trace carries from step to step
    "addition": "partial sum digits and carry", "blocksworld": "block stacking configuration",
    "boolean_expressions": "partially reduced boolean expression", "cellular_automaton": "row of 8 binary cells",
    "cruxeval": "program variable values", "cup_shuffling": "permutation of objects over people",
    "dyck": "stack contents", "entity_tracking_boxes": "contents of every box", "hanoi": "disks on three pegs",
    "multiplication": "partial products and running sum", "nested_arithmetic": "values of evaluated sub-expressions",
    "random_lookup_table": "current symbol", "s5_composition": "permutation of 5",
    "synthetic_program_trace": "integer variable assignments", "threesum": "enumeration position (implicit) and hit flag",
    "turing_machine": "tag-system symbol queue",
}
def desk_val(slug, md_col, json_key, conv):
    """DESK.md table value (the curated, later-corrected table) with desk.json as fallback; disagreements are logged."""
    dj = json.load(open(ROOT / "tasks" / slug / "desk.json"))
    mv = desk_md.get(slug, {}).get(md_col); mdv = None if mv in (None, "·") else mv
    jv = dj.get(json_key)
    a = None if mdv is None else conv(mdv)
    b = conv(jv) if (jv is not None or json_key == "answer_space") else None
    if a is not None and a != b:
        note("DESK.md vs desk.json", slug, json_key, mdv, "null" if jv is None else jv, "desk disagreement (DESK.md used)")
    return a if a is not None else b
yn = lambda v: v if isinstance(v, bool) else {"Y": True, "N": False}.get(v, v)
bits = lambda v: round(float(v), 1)
ans = lambda v: "∞" if v in ("∞", None) else str(int(float(v)))
max_depth = defaultdict(int)
for k in sweep: max_depth[k[0]] = max(max_depth[k[0]], k[3])
def primary_format(slug):
    passing = [f for f in FORMATS if any(surv.get((slug, f, m), {}).get("passes") for m in MODELS)]
    if slug in plan3 and b3_format(slug) in passing: return b3_format(slug), passing
    if passing:
        nm = lambda f: sum(bool(surv.get((slug, f, m), {}).get("passes")) for m in MODELS)
        return max(passing, key=lambda f: (nm(f), f == b1_format(slug))), passing
    return b1_format(slug), passing
task_rows = []
for slug in implemented(include_dropped=True):
    pf, _ = primary_format(slug)
    aspace = desk_val(slug, "ans space", "answer_space", ans)
    if aspace is None: aspace = "∞"
    task_rows.append(dict(
        slug=slug, display_name=display_name(slug), marker="†" if slug in DAGGER else ("*" if slug in ASTERISK else ""),
        source_short=SOURCE[slug][0], source_title=SOURCE[slug][1], surviving_format=pf, state_description=STATE[slug],
        bits_per_step=desk_val(slug, "bits", "state_bits", bits), bounded_state=desk_val(slug, "bnd", "state_bounded", yn),
        theory_class=desk_val(slug, "theory", "theory_class", str), answer_space=aspace,
        guess_floor=0.0 if aspace == "∞" else 1 / int(aspace),
        contamination_risk=desk_val(slug, "contam", "contamination_risk", str), depth_knob=knobs_md.get(slug, ""),
        max_depth_swept=max_depth[slug], trace_source_short=SOURCE[slug][2], dropped=slug in DROPPED,
        harness_depth=HARNESS_DEPTH[slug]))
T_COLS = ["slug", "display_name", "marker", "source_short", "source_title", "surviving_format", "state_description",
          "bits_per_step", "bounded_state", "theory_class", "answer_space", "guess_floor", "contamination_risk", "depth_knob",
          "max_depth_swept", "trace_source_short", "dropped", "harness_depth"]
counts["tasks.csv"] = write("tasks.csv", T_COLS, task_rows)
tasks = {r["slug"]: r for r in task_rows}

# ---- master -----------------------------------------------------------------------------------------------------
master = []
for slug in implemented(include_dropped=True):
    pf, passing = primary_format(slug)
    for fmt in (passing or [pf]):
        S = lambda m: surv.get((slug, fmt, m))
        def ps(m): s = S(m); return "" if not s or not s["n_depths"] else s["passes"]
        def tps(m): s = S(m); return s["tok_per_step_at_best"] if s and s["passes"] else None
        pas = [S(m) for m in MODELS if S(m) and S(m)["passes"]]
        b2r = b2.get(slug) if b2.get(slug, {}).get("format") == fmt else None
        b3r = b3s.get(slug) if slug in b3s and b3s[slug]["format"] == fmt else None
        if b2r: shape, shape_src = KO_SHAPE.get(slug, "n/a"), "batch2 line-fraction knockout"
        elif b3r:
            floor = sweep.get((slug, fmt, b3r["model"], b3r["depth"]), {}).get("acc_nocot")
            shape = "decided_at_end" if floor is not None and b3r["goldko_q3"] <= floor + 0.15 else "early"
            shape_src = "batch3 gold step-aligned knockout (goldko@3d/4 vs no-CoT)"
        else: shape, shape_src = "n/a", ""
        master.append(dict(
            slug=slug, display_name=display_name(slug), marker=tasks[slug]["marker"], source_short=SOURCE[slug][0], format=fmt,
            state_description=STATE[slug], bits_per_step=tasks[slug]["bits_per_step"], answer_space=tasks[slug]["answer_space"],
            tok_per_step_9b=tps("qwen9b"), tok_per_step_ds=tps("ds"),
            passes_qwen9b=ps("qwen9b"), passes_llama70b=ps("llama70b"), passes_ds=ps("ds"),
            models_passing=len(pas), deepest_passing_depth_any=max((s["deepest_passing_depth"] for s in pas), default=None),
            best_gap_any=max((s["best_gap"] for s in pas), default=None),
            filler_recovery=b2r["filler_recovery"] if b2r else None, knockout_shape=shape,
            cont_redacted=(b2r["cont_redacted"] or None) if b2r else None, cont_plain=(b2r["cont_plain"] or None) if b2r else None,
            goldko_dm1=b3r["goldko_dm1"] if b3r else None, mean_propagation=b3r["mean_propagation"] if b3r else None,
            self_checking=("n/a" if not b3r else {"hanoi": "yes", "dyck": "weak"}.get(slug, "no")),
            published_format_passes_anywhere=("" if fmt in ("published", "ours") else
                                              any(bool((surv.get((slug, b1_format(slug), m)) or {}).get("passes")) for m in MODELS)),
            passes_dspro=ps("dspro"), primary=(fmt == pf), knockout_shape_source=shape_src,
            batch2_model=b2r["model"] if b2r else None, batch2_depth=b2r["depth"] if b2r else None,
            batch3_depth=b3r["depth"] if b3r else None))
M_COLS = ["slug", "display_name", "marker", "source_short", "format", "state_description", "bits_per_step", "answer_space",
          "tok_per_step_9b", "tok_per_step_ds", "passes_qwen9b", "passes_llama70b", "passes_ds", "models_passing",
          "deepest_passing_depth_any", "best_gap_any", "filler_recovery", "knockout_shape", "cont_redacted", "cont_plain",
          "goldko_dm1", "mean_propagation", "self_checking", "published_format_passes_anywhere",
          "passes_dspro", "primary", "knockout_shape_source", "batch2_model", "batch2_depth", "batch3_depth"]
# cont_* / filler_recovery arrive as strings from batch2.csv; normalise to float
for r in master:
    for k in ("filler_recovery", "cont_redacted", "cont_plain"):
        if isinstance(r[k], str): r[k] = float(r[k])
counts["master.csv"] = write("master.csv", M_COLS, master)
counts["crosscheck.csv"] = write("crosscheck.csv", ["table", "key", "field", "expected_in_file", "got_from_logs", "class"], checks)

print("wrote", OUT)
for k, v in counts.items(): print(f"  {k}: {v} rows")
print(f"crosscheck: {len(checks)} disagreements")
for c in checks: print("  ", " | ".join(map(str, c)))
