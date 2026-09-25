"""Fill the <!-- TABLE:name --> slots in the blog post from results/tidy/*.csv.

    cd harness && uv run python post/tables.py [--post PATH]

Each slot becomes `<!-- TABLE:name -->…generated HTML…<!-- /TABLE:name -->`, so rerunning replaces
the previous block. Every number in the post's tables comes from the tidy CSVs; nothing is hand-typed.
If post/charts_alt.json exists, the alt text of each chart <img> is refreshed from it too.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIDY = HERE.parent / "results" / "tidy"
POST_DEFAULT = Path.home() / "markhenrypage" / "posts" / "2026-09-24-stateful-tasks.html"

NAMES = {
    "addition": "addition", "blocksworld": "blocksworld", "boolean_expressions": "boolean expressions",
    "cellular_automaton": "cellular automaton", "cruxeval": "CRUXEval", "cup_shuffling": "cup shuffling",
    "dyck": "Dyck", "entity_tracking_boxes": "entity tracking", "hanoi": "Tower of Hanoi",
    "multiplication": "multiplication", "nested_arithmetic": "nested arithmetic",
    "random_lookup_table": "random lookup table", "s5_composition": "S5 composition",
    "synthetic_program_trace": "program trace", "threesum": "3SUM", "turing_machine": "tag system",
}
FORMATS = {"published": "", "ours": "ours", "ergonomic": "ergonomic", "execute": "execute",
           "cells2": "cells rev 2", "cells3": "cells rev 3"}
MODELS = {"qwen9b": "Qwen3.5-9B", "llama70b": "Llama-3.3-70B", "ds": "DeepSeek-V4-Flash", "dspro": "DeepSeek-V4-Pro"}
MODEL_ORDER = ["qwen9b", "llama70b", "ds"]
CLEAN_FOUR = {"random_lookup_table", "nested_arithmetic", "s5_composition", "cellular_automaton"}
# bibliography: the paper whose trace format each task implements (verified against arXiv / Crossref titles)
SOURCE_URLS = {
    "addition": "https://arxiv.org/abs/2112.00114", "synthetic_program_trace": "https://arxiv.org/abs/2112.00114",
    "blocksworld": "https://arxiv.org/abs/2405.04776",
    "boolean_expressions": "https://arxiv.org/abs/2210.09261", "cup_shuffling": "https://arxiv.org/abs/2210.09261",
    "dyck": "https://arxiv.org/abs/2210.09261", "nested_arithmetic": "https://arxiv.org/abs/2210.09261",
    "cellular_automaton": "https://doi.org/10.1007/11786986_13",
    "cruxeval": "https://arxiv.org/abs/2401.03065",
    "entity_tracking_boxes": "https://arxiv.org/abs/2305.02363",
    "hanoi": "https://arxiv.org/abs/2506.06941",
    "multiplication": "https://arxiv.org/abs/2305.18654",
    "random_lookup_table": "https://arxiv.org/abs/2311.12997",
    "s5_composition": "https://arxiv.org/abs/2210.10749",
    "threesum": "https://arxiv.org/abs/2404.15758",
    "turing_machine": "https://arxiv.org/abs/2504.20771",
}


def source_link(slug, text):
    url = SOURCE_URLS.get(slug)
    return f'<a href="{url}">{html.escape(text)}</a>' if url else html.escape(text)


def read(name):
    with open(TIDY / f"{name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=2, signed=False):
    if x in ("", None):
        return "·"
    v = float(x)
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s


def dint(x):
    return "·" if x in ("", None) else f"d{int(float(x))}"


def marker(display_name):
    return "".join(ch for ch in display_name if ch in "†*")


def tname(slug, display_name=None, fmt=None):
    s = NAMES.get(slug, slug)
    if display_name:
        s += marker(display_name)
    if fmt and FORMATS.get(fmt, fmt):
        s += f' <span style="color:#777">({FORMATS.get(fmt, fmt)})</span>'
    return s


def table(caption_lead, caption_rest, head, rows, widths=None):
    out = ['<div class="table-wrap">', "<table>",
           f"    <caption><b>{caption_lead}</b> {caption_rest}</caption>", "    <thead>"]
    if isinstance(head[0], list):  # two header rows
        for hr in head:
            out.append("<tr>" + "".join(f"<th{attr}>{h}</th>" for h, attr in hr) + "</tr>")
    else:
        out.append("<tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>")
    out += ["    </thead>", "    <tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    out += ["    </tbody>", "</table>", "</div>"]
    return "\n".join(out)


# ---------------------------------------------------------------- data
SURV = read("survival")
SWEEP = read("depth_sweep")
B2 = read("batch2")
B3S = read("batch3_summary")
TASKS = read("tasks")
MASTER = read("master")
DENS = read("density")
for _d in DENS:
    if _d["slug"] == "cellular_automaton" and _d["format"] == "cells":
        _d["format"] = "cells3"

DISPLAY = {t["slug"]: t["display_name"] for t in TASKS}
SURV_BY = {(r["slug"], r["format"], r["model"]): r for r in SURV}
SWEEP_BY = {(r["slug"], r["format"], r["model"], int(r["depth"])): r for r in SWEEP}
B1_FORMAT = {t["slug"]: ("ours" if t["slug"] in ("cellular_automaton", "entity_tracking_boxes") else "published")
             for t in TASKS}


def surv_cell(r, short=False):
    if r is None:
        return "·"
    if r["passes"] == "True":
        rng = dint(r["working_depth"])
        if r["deepest_passing_depth"] and int(float(r["deepest_passing_depth"])) != int(float(r["working_depth"])):
            rng += f"–{dint(r['deepest_passing_depth'])}"
        return f"passes {rng}<br>gap {fnum(r['best_gap'], signed=True)} at {dint(r['best_depth'])} (acc {fnum(r['best_acc_cot'])})"
    reason = {"no gap": "no gap", "acc<0.7": "acc below 0.7"}.get(r["fail_reason"], html.escape(r["fail_reason"] or "fails"))
    return f"— {reason}<br>best gap {fnum(r['best_gap'], signed=True)} at {dint(r['best_depth'])} (acc {fnum(r['best_acc_cot'])})"


# ---------------------------------------------------------------- tables
def t_survival():
    verdict = {}
    for t in TASKS:
        s = t["slug"]
        fmt = B1_FORMAT[s]
        n = sum(1 for m in MODEL_ORDER if (SURV_BY.get((s, fmt, m)) or {}).get("passes") == "True")
        if t.get("dropped") == "True":
            v = "dropped: solved without a scratchpad"
        elif n == 3:
            v = "survives on all three"
        elif n == 2:
            v = "survives on two"
        elif n == 1:
            who = [MODELS[m] for m in MODEL_ORDER if (SURV_BY.get((s, fmt, m)) or {}).get("passes") == "True"][0]
            v = f"survives on {who} only"
        else:
            v = "fails as published"
        if s in ("hanoi", "cellular_automaton", "s5_composition", "random_lookup_table", "threesum", "addition"):
            v += "; second chance below"
        verdict[s] = (n, v)
    rows = []
    for t in sorted(TASKS, key=lambda t: (-verdict[t["slug"]][0], t["slug"])):
        s = t["slug"]
        fmt = B1_FORMAT[s]
        rows.append([tname(s, t["display_name"], fmt if fmt == "ours" else None)]
                    + [surv_cell(SURV_BY.get((s, fmt, m))) for m in MODEL_ORDER]
                    + [verdict[s][1]])
    return table(
        "Batch 1: survival by model, formats as published.",
        "A task passes on a model at a depth where acc(CoT) − acc(no-CoT) ≥ 0.2 with exact McNemar p &lt; 0.05 and "
        "acc(CoT) ≥ 0.7. Cells give the passing depth range, then the best gap and the depth it occurs at. "
        "n = 50 per cell (150 on the 9B's marginal cells). Multiplication was dropped after the 9B run and never "
        "ran on DeepSeek.",
        ["task", MODELS["qwen9b"], MODELS["llama70b"], MODELS["ds"], "verdict"], rows)


def t_deepest():
    rows = []
    for t in TASKS:
        s = t["slug"]
        fmt = B1_FORMAT[s]
        cells = []
        anyp = False
        for m in MODEL_ORDER:
            r = SURV_BY.get((s, fmt, m))
            if r and r["passes"] == "True":
                anyp = True
                cells.append(f"{dint(r['deepest_passing_depth'])} ({fnum(r['deepest_acc_cot'])})")
            else:
                cells.append("—")
        if anyp:
            rows.append([tname(s, t["display_name"])] + cells)
    return table(
        "Deepest passing depth per model, formats as published.",
        "The deepest depth at which the task still passes the survival rules, with acc(CoT) there in parentheses. "
        "Depth grids are task-specific, so compare across a row, not down a column.",
        ["task", MODELS["qwen9b"], MODELS["llama70b"], MODELS["ds"]], rows)


def t_second_chance():
    spec = [("hanoi", "published", "execute", "execution formulation"),
            ("cellular_automaton", "ours", "cells2", "per-cell format (rev 2)"),
            ("cellular_automaton", "ours", "cells3", "per-cell, named neighbours, running row (rev 3)"),
            ("s5_composition", "published", "ergonomic", "ergonomic (SFT-free) format"),
            ("random_lookup_table", "published", "ergonomic", "ergonomic (SFT-free) format"),
            ("threesum", "published", "ergonomic", "ergonomic (SFT-free) format"),
            ("addition", "published", "ergonomic", "ergonomic (SFT-free) format")]

    def cell(r):
        if r is None:
            return "·"
        tag = "PASS" if r["passes"] == "True" else "fail"
        return (f"{tag} {dint(r['best_depth'])}: {fnum(r['best_acc_cot'])} / {fnum(r['best_acc_nocot'])} / "
                f"{fnum(r['best_gap'], signed=True)}")

    rows = []
    for s, f0, f1, label in spec:
        rows.append([tname(s, DISPLAY[s]), label,
                     cell(SURV_BY.get((s, f0, "qwen9b"))), cell(SURV_BY.get((s, f1, "qwen9b"))),
                     cell(SURV_BY.get((s, f0, "ds"))), cell(SURV_BY.get((s, f1, "ds")))])
    return table(
        "Second chances: as published versus variant.",
        "Each cell is PASS or fail under the batch-1 rules at the best passing depth (else the best-gap depth): "
        "acc(CoT) / acc(no-CoT) / gap. Same instances and same answers in both formats. The 9B's cellular "
        "automaton rev-2 cell rests on 27 scored samples (a credit-limit error cut the run short).",
        ["task", "variant", "9B as published", "9B variant", "DeepSeek as published", "DeepSeek variant"], rows)


def t_filler():
    rows = []
    for r in sorted(B2, key=lambda r: -float(r["filler_recovery"])):
        rows.append([tname(r["slug"], DISPLAY[r["slug"]], r["format"] if r["format"] == "ours" else None),
                     MODELS[r["model"]], dint(r["depth"]), fnum(r["acc_cot"]), fnum(r["acc_nocot"]),
                     fnum(r["filler"]), fnum(r["filler_recovery"], signed=True)])
    return table(
        "Filler recovery.",
        "The model's own batch-1 trace is replaced by the same number of <code>&nbsp;.</code> tokens and "
        "<code>Answer:</code> is prefilled. Recovery = acc(filler) − acc(no-CoT). n = 50. Each survivor at its "
        "best passing (model, depth) from batch 1, so three rows are DeepSeek and the rest the 9B.",
        ["task", "model", "depth", "acc(CoT)", "no-CoT", "filler", "recovery"], rows)


def own_trace_acc(slug, fmt, depth):
    r = SWEEP_BY.get((slug, fmt, "qwen9b", int(depth)))
    return fnum(r["acc_cot"]) if r else "·"


def t_goldko():
    rows = []
    for r in sorted(B3S, key=lambda r: (r["slug"] not in CLEAN_FOUR, float(r["goldko_dm1"]))):
        rows.append([tname(r["slug"], DISPLAY[r["slug"]], r["format"]), dint(r["depth"]),
                     fnum(r["goldko_k0"]), fnum(r["goldko_q1"]), fnum(r["goldko_half"]), fnum(r["goldko_q3"]),
                     f"<b>{fnum(r['goldko_dm1'])}</b>", own_trace_acc(r["slug"], r["format"], r["depth"])])
    return table(
        "Step-aligned knockout on Qwen3.5-9B.",
        "The gold trace is prefilled through step k, then <code>Answer:</code> is forced. The last column is the "
        "model's accuracy writing its own full trace at the same depth. Each task at the depth batch 2 used, its "
        "best passing depth on the 9B, not its deepest. n = 50 per cell. The first four rows are the tasks with "
        "propagation near 1.",
        ["task", "depth", "k = 0", "d/4", "d/2", "3d/4", "d − 1", "own trace"], rows)


def t_propagation():
    notes = {"hanoi": "model detects the illegal move and recomputes",
             "dyck": "a wrong stack is often absorbed by later pops",
             "synthetic_program_trace": "corrupted values are often overwritten later",
             "threesum": "undefined: the clean continuation itself fails",
             "cup_shuffling": "", "turing_machine": "", "nested_arithmetic": "", "s5_composition": "",
             "random_lookup_table": "", "cellular_automaton": ""}
    rows = []
    for r in sorted(B3S, key=lambda r: (r["slug"] == "threesum", -float(r["mean_propagation"]))):
        cells = [f"{fnum(r[f'mistake_{q}'])} ({fnum(r[f'cont_{q}'])})" for q in ("q1", "half", "q3")]
        rows.append([tname(r["slug"], DISPLAY[r["slug"]], r["format"]), dint(r["depth"])] + cells
                    + [f"<b>{fnum(r['mean_propagation'])}</b>", notes.get(r["slug"], "")])
    return table(
        "Mistake propagation on Qwen3.5-9B.",
        "Step k's reported state is corrupted and the model continues; accuracy against the original answer, "
        "with the uncorrupted continuation from the same step in parentheses. Propagation = continuation − "
        "corrupted, averaged over k = d/4, d/2, 3d/4. n = 50 per cell.",
        ["task", "depth", "corrupted at d/4 (clean)", "d/2 (clean)", "3d/4 (clean)", "propagation", "what happens"],
        rows)


def t_redaction():
    rows = []
    for r in sorted([r for r in B2 if r["cont_plain"]],
                    key=lambda r: -(float(r["cont_redacted"]) - float(r["cont_plain"]))):
        d = float(r["cont_redacted"]) - float(r["cont_plain"])
        rows.append([tname(r["slug"], DISPLAY[r["slug"]], r["format"] if r["format"] == "ours" else None),
                     MODELS[r["model"]], dint(r["depth"]), r["cont_k"], fnum(r["cont_plain"]),
                     fnum(r["cont_redacted"]), fnum(d, signed=True)])
    return table(
        "Continuation from a gold prefix, with and without the prompt redacted.",
        "The gold trace is prefilled through step k = d/2 and the model continues. Redacted: the prompt's initial "
        "state and first k operators are replaced by <code>[…]</code>, so the state at step k is only in the trace. "
        "Δ = redacted − plain. Addition has no redaction row (its operands cannot be hidden). S5 as published is "
        "0.00 both ways: DeepSeek cannot continue the terse prefix even with the full prompt.",
        ["task", "model", "depth", "k", "plain", "redacted", "Δ"], rows)


def t_tokens():
    dens_by = {(d["slug"], d["format"]): d for d in DENS}
    rows = []
    prim = [m for m in MASTER if m["primary"] in ("T", "True") and (m["tok_per_step_9b"] or m["tok_per_step_ds"])]

    def key(m):
        return float(m["tok_per_step_9b"]) if m["tok_per_step_9b"] else 1e9

    for m in sorted(prim, key=key):
        s, f = m["slug"], m["format"]
        d = dens_by.get((s, f)) or dens_by.get((s, "published"))
        gold = fnum(d["step_tokens_mean"], 0) if d else "·"
        # how far the model's spend departs from the gold format at its best 9B depth
        note = ""
        sr = SURV_BY.get((s, f, "qwen9b")) or SURV_BY.get((s, f, "ds"))
        if sr and sr["best_depth"]:
            sw = SWEEP_BY.get((s, f, sr["model"], int(float(sr["best_depth"]))))
            if sw and sw.get("cot_len_ratio_vs_gold"):
                ratio = float(sw["cot_len_ratio_vs_gold"])
                if ratio >= 1.5:
                    note = f"{MODELS[sr['model']]} writes {ratio:.1f}× the gold trace"
                elif ratio <= 0.7:
                    note = f"{MODELS[sr['model']]} writes {ratio:.1f}× the gold trace"
        rows.append([tname(s, m["display_name"], f), fnum(m["tok_per_step_9b"], 0), fnum(m["tok_per_step_ds"], 0),
                     gold, note])
    return table(
        "Tokens per step, surviving formats.",
        "Output tokens per step of depth at the deepest passing depth, per model: what the model actually spends. "
        "The gold column is the format's own cost, the gold trace tokenized with the Qwen3.5 tokenizer. Where the "
        "two differ by more than half, the note says so: 3SUM's models keep enumerating past the first hit, and "
        "DeepSeek abandons the terse S5 format for prose.",
        ["task", "9B", "DeepSeek", "gold trace", "note"], rows)


def t_density():
    order = {"yes": 0, "partial": 1, "no": 2}
    rows = []
    for d in sorted(DENS, key=lambda d: (order.get(d["estimable"], 3),
                                          -float(d["density_est"]) if d["density_est"] else 1.0)):
        rows.append([tname(d["slug"], DISPLAY[d["slug"]], d["format"]), dint(d["depth"]),
                     fnum(d["step_tokens_mean"], 1), fnum(d["state_tokens_mean"], 1), fnum(d["state_frac"]),
                     fnum(d["mean_propagation"]), f"<b>{fnum(d['density_est'])}</b>",
                     fnum(d["bits_per_step"], 1), fnum(d["bits_per_state_token"]), d["estimable"]])
    return table(
        "Ballpark load-bearing density.",
        "Gold traces tokenized with the Qwen3.5-9B tokenizer, 30 instances per row at the batch-3 depth (batch-2 "
        "depth where there is no batch 3). State tokens = tokens inside the step's state field(s); where a format "
        "writes the state twice both copies are counted. Density = state fraction × measured propagation. Bits per "
        "step from the desk table. \"Partial\" rows have a fraction but no propagation to multiply it by.",
        ["task", "depth", "tokens / step", "state tokens", "state fraction", "propagation", "density",
         "bits / step", "bits / state token", "estimable"], rows)


def t_master():
    dens_by = {(d["slug"], d["format"]): d for d in DENS}
    src = {t["slug"]: t["source_short"] for t in TASKS}

    def yn(v):
        return {"True": "✓", "False": "✗"}.get(v, "·")

    rows = []
    prim = [m for m in MASTER if m["primary"] in ("T", "True")]
    prim.sort(key=lambda m: (-(int(m["models_passing"]) if m["models_passing"] else 0),
                             -(float(m["mean_propagation"]) if m["mean_propagation"] else -9), m["slug"]))
    for m in prim:
        s, f = m["slug"], m["format"]
        d = dens_by.get((s, f))
        rows.append([
            tname(s, m["display_name"], f), src.get(s, ""), m["state_description"], fnum(m["bits_per_step"], 1),
            m["answer_space"],
            " ".join(yn(m[c]) for c in ("passes_qwen9b", "passes_llama70b", "passes_ds")),
            dint(m["deepest_passing_depth_any"]), fnum(m["tok_per_step_9b"], 0) if m["tok_per_step_9b"] else fnum(m["tok_per_step_ds"], 0),
            fnum(m["filler_recovery"], signed=True), m["knockout_shape"].replace("_", " ") if m["knockout_shape"] else "·",
            (f"{fnum(m['cont_redacted'])} / {fnum(m['cont_plain'])}" if m["cont_plain"] else "·"),
            fnum(m["goldko_dm1"]), fnum(m["mean_propagation"]), m["self_checking"] or "·",
            fnum(d["density_est"]) if d and d["density_est"] else "·",
        ])
    head = ["task", "source", "state", "bits / step", "answers", "passes 9B · 70B · DS", "deepest depth",
            "tok / step", "filler rec.", "knockout shape", "redacted / plain", "knockout at d−1", "propagation",
            "self-checking", "density"]
    return table(
        "Everything, one row per task in its surviving format.",
        "Passes = survival rules on that model (· = that format never ran there). Deepest depth = deepest passing "
        "depth on any model. Tokens per step on the 9B where it passes, else DeepSeek. Filler recovery and the "
        "redacted/plain continuation are from batch 2 (published formats only, so the rewritten rows have none); "
        "knockout at d−1, propagation and self-checking from batch 3 (9B); density from the estimate above.",
        head, rows)


def t_tasks():
    rows = []
    for t in sorted(TASKS, key=lambda t: t["slug"]):
        knob = re.split(r"\s*\(", t["depth_knob"])[0]
        rows.append([tname(t["slug"], t["display_name"]), source_link(t["slug"], t["source_short"]), t["harness_depth"],
                     t["state_description"], knob, t["answer_space"], FORMATS.get(t["surviving_format"], t["surviving_format"]) or "published"])
    return table(
        "The sixteen tasks.",
        "Source is the paper whose trace format was implemented, linked; SOURCING.md in each task directory records "
        "the exact commit or figure. A step is one unit of depth in the harness. Format is the one the results "
        "tables use for that task.",
        ["task", "source", "one step is", "state carried", "depth knob", "answer space", "format"], rows)


GENERATORS = {
    "survival": t_survival, "deepest": t_deepest, "second_chance": t_second_chance, "filler": t_filler,
    "goldko": t_goldko, "propagation": t_propagation, "redaction": t_redaction, "tokens": t_tokens,
    "density": t_density, "master": t_master, "tasks": t_tasks,
}


def fill(post_path: Path):
    text = post_path.read_text(encoding="utf-8")
    for name, gen in GENERATORS.items():
        block = f"<!-- TABLE:{name} -->\n{gen()}\n<!-- /TABLE:{name} -->"
        pat = re.compile(rf"<!-- TABLE:{name} -->.*?(?:<!-- /TABLE:{name} -->|(?=\n\n)|\Z)", re.S)
        if not pat.search(text):
            print(f"[tables] no slot for {name}")
            continue
        text = pat.sub(lambda _m: block, text, count=1)
    alt_path = HERE / "charts_alt.json"
    if alt_path.exists():
        alts = json.loads(alt_path.read_text())
        for fname, alt in alts.items():
            pat = re.compile(rf'(<img[^>]*src="stateful-tasks/{re.escape(fname)}"[^>]*alt=")[^"]*(")')
            text, n = pat.subn(lambda m: m.group(1) + html.escape(alt, quote=True) + m.group(2), text)
            if not n:
                print(f"[tables] no <img> for {fname}")
    post_path.write_text(text, encoding="utf-8")
    print(f"[tables] wrote {post_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", type=Path, default=POST_DEFAULT)
    fill(ap.parse_args().post)
