"""Hand-rolled SVG charts for the stateful-tasks blog post.

    cd ~/stateful-tasks/harness && uv run python post/charts.py

Reads results/tidy/*.csv, writes five SVGs to ~/markhenrypage/assets/stateful-tasks/
and alt texts to post/charts_alt.json. No plotting library; the SVG text is written here.
"""
import csv, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIDY = HERE.parent / "results" / "tidy"
OUT = Path.home() / "markhenrypage" / "assets" / "stateful-tasks"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GRAY, TINT = "#1a5fb4", "#c0392b", "#999", "#9db9de"
INK, AXIS, GRID, INK2 = "#555", "#999", "#e5e5e5", "#333"
W = 640
MODELS = [("qwen9b", "Qwen3.5-9B"), ("llama70b", "Llama-3.3-70B"), ("ds", "DeepSeek-V4-Flash")]
FMT_LABEL = {"ergonomic": "ergonomic", "execute": "execute", "cells3": "cells rev 3",
             "cells2": "cells rev 2", "ours": "ours", "published": "published"}


def rd(name):
    with open(TIDY / name) as f:
        return list(csv.DictReader(f))


def num(s):
    return None if s is None or s.strip() == "" else float(s)


DS = rd("depth_sweep.csv")
SURV = rd("survival.csv")
B3 = rd("batch3.csv")
B3S = rd("batch3_summary.csv")
MASTER = rd("master.csv")
NAME = {"addition": "addition", "blocksworld": "blocksworld", "boolean_expressions": "boolean expressions",
        "cellular_automaton": "cellular automaton", "cruxeval": "CRUXEval", "cup_shuffling": "cup shuffling",
        "dyck": "Dyck", "entity_tracking_boxes": "entity tracking", "hanoi": "Tower of Hanoi",
        "multiplication": "multiplication", "nested_arithmetic": "nested arithmetic",
        "random_lookup_table": "random lookup table", "s5_composition": "S5 composition",
        "synthetic_program_trace": "program trace", "threesum": "3SUM", "turing_machine": "tag system"}
MARKER = {r["slug"]: r["marker"] for r in MASTER}
DISPLAY = {slug: NAME[slug] + MARKER.get(slug, "") for slug in NAME}
sweep = {(r["slug"], r["format"], r["model"], int(r["depth"])): r for r in DS}
surv = {(r["slug"], r["format"], r["model"]): r for r in SURV}


# ---------- text / svg helpers ----------
def tw(s, size=11):
    """Approximate Arial advance width in px."""
    w = 0.0
    for ch in s:
        if ch in "il.,:;'|!j":
            w += 0.24
        elif ch in "frt()-*[] ":
            w += 0.33
        elif ch in "mwMW":
            w += 0.85
        elif ch.isupper():
            w += 0.68
        else:
            w += 0.556
    return w * size


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def f(v):
    return f"{v:.1f}"


def T(x, y, s, anchor="start", fill=INK, size=11, weight=None):
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    sz = f' font-size="{size}"' if size != 11 else ""
    wt = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{f(x)}" y="{f(y)}"{a} fill="{fill}"{sz}{wt}>{esc(s)}</text>'


def L(x1, y1, x2, y2, stroke=GRID, w=1, extra=""):
    return f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{stroke}" stroke-width="{w}"{extra}/>'


def dot(x, y, color, r=4):
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="{color}" stroke="white" stroke-width="2"/>'


def ring(x, y, color, r=4.5):
    return f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="white" stroke="{color}" stroke-width="2"/>'


def path(pts, color, w=2):
    d = " ".join(("M" if i == 0 else "L") + f"{f(x)},{f(y)}" for i, (x, y) in enumerate(pts))
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round"/>'


def svg(h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" '
            f'font-family="Arial, sans-serif" font-size="11">\n<rect width="{W}" height="{h}" fill="white"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def legend(items, x, y):
    """items: list of (kind, color, label); kind in line|dot|ring|wash|tick|linedot|biglittle. Returns (svg parts, end x)."""
    out = []
    for kind, color, label in items:
        if kind == "line":
            out.append(L(x, y, x + 20, y, color, 2, ' stroke-linecap="round"')); x += 26
        elif kind == "linedot":
            out.append(L(x, y, x + 20, y, color, 2, ' stroke-linecap="round"')); out.append(dot(x + 10, y, color)); x += 26
        elif kind == "dot":
            out.append(dot(x + 5, y, color, 4.5)); x += 14
        elif kind == "bigdot":
            out.append(dot(x + 6, y, color, 6)); x += 16
        elif kind == "ring":
            out.append(ring(x + 5, y, color)); x += 14
        elif kind == "wash":
            out.append(f'<rect x="{f(x)}" y="{f(y - 6)}" width="16" height="12" fill="{BLUE}" fill-opacity="0.1"/>'); x += 22
        elif kind == "tick":
            out.append(L(x + 5, y - 6, x + 5, y + 6, GRAY, 2)); x += 14
        out.append(T(x, y + 4, label)); x += tw(label) + 18
    return out, x


def spread(targets, lo, hi, gap):
    """Place labels at positions near targets (sorted input order preserved), min spacing gap, within [lo, hi]."""
    order = sorted(range(len(targets)), key=lambda i: targets[i])
    pos = [targets[i] for i in order]
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap - 1e-6:
                c = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = c - gap / 2, c + gap / 2
                moved = True
        if pos[0] < lo:
            pos = [p + (lo - pos[0]) for p in pos]; moved = True
        if pos[-1] > hi:
            pos = [p - (pos[-1] - hi) for p in pos]; moved = True
        if not moved:
            break
    res = [0.0] * len(targets)
    for k, i in enumerate(order):
        res[i] = pos[k]
    return res


def fmt_depth(d):
    return str(int(d)) if float(d).is_integer() else f"{d:g}"


ALT = {}


# ---------- 1. depth sweep small multiples ----------
def chart_depth_sweep():
    rows = [r for r in MASTER if r["primary"] == "True"]
    rows.sort(key=lambda r: (-int(r["models_passing"]), -(num(r["deepest_passing_depth_any"]) or 0)))
    LG, PW, GAP, PH = 150, 150, 12, 64
    PADX = 8
    TOP = 66
    ROWH = PH + 32
    H = TOP + ROWH * len(rows) + 20
    b = []
    leg, _ = legend([("linedot", BLUE, "with chain of thought"), ("linedot", RED, "without chain of thought"),
                     ("wash", BLUE, "gap")], 8, 14)
    b += leg
    b.append(T(8, 36, "Accuracy (0 to 1) by depth, log scale; x ticks are each run's own depth grid", fill=INK))
    for c, (_, name) in enumerate(MODELS):
        x0 = LG + c * (PW + GAP)
        b.append(T(x0 + PW / 2, TOP - 12, name, "middle", INK2, weight="bold"))
    for i, r in enumerate(rows):
        slug, fmt = r["slug"], r["format"]
        y0 = TOP + i * ROWH
        # row label (wrap long names at an underscore)
        name = DISPLAY[slug]
        lines = [name]
        if tw(name) > LG - 30:
            cut = name.rfind(" ", 0, len(name) - 3)
            lines = [name[:cut], name[cut + 1:]]
        yy = y0 + 14
        for ln in lines:
            b.append(T(4, yy, ln, fill=INK2)); yy += 13
        if fmt != "published":
            b.append(T(4, yy, FMT_LABEL[fmt], fill=GRAY))
        grids = {m: sorted(d for (s, fo, mm, d) in sweep if s == slug and fo == fmt and mm == m) for m, _ in MODELS}
        alld = [d for g in grids.values() for d in g]
        lo, hi = math.log2(min(alld)), math.log2(max(alld))
        if hi == lo:
            hi = lo + 1
        yv = lambda v: y0 + PH - v * PH
        for t, lab in [(0, "0"), (0.5, ".5"), (1, "1")]:
            b.append(T(LG - 5, yv(t) + 4, lab, "end", INK, 10))
        for c, (m, _) in enumerate(MODELS):
            x0 = LG + c * (PW + GAP)
            xv = lambda d: x0 + PADX + (math.log2(d) - lo) / (hi - lo) * (PW - 2 * PADX)
            g = grids[m]
            if not g:
                b.append(L(x0, yv(0), x0 + PW, yv(0), GRID))
                b.append(T(x0 + PW / 2, yv(0.5) + 4, "not run", "middle", GRAY, 10))
                continue
            for t in (0.5, 1):
                b.append(L(x0, yv(t), x0 + PW, yv(t), GRID))
            b.append(L(x0, yv(0), x0 + PW, yv(0), AXIS))
            # x ticks + greedy non-colliding labels (first and last always)
            placed = []
            cand = [g[0], g[-1]] + g[1:-1]
            for d in cand:
                x, w = xv(d), tw(fmt_depth(d), 10)
                if all(abs(x - px) >= (w + pw) / 2 + 3 for px, pw in placed):
                    placed.append((x, w))
                    b.append(T(x, yv(0) + 13, fmt_depth(d), "middle", INK, 10))
            for d in g:
                b.append(L(xv(d), yv(0), xv(d), yv(0) + 3, AXIS))
            cot = [(xv(d), yv(float(sweep[(slug, fmt, m, d)]["acc_cot"]))) for d in g]
            noc = [(xv(d), yv(float(sweep[(slug, fmt, m, d)]["acc_nocot"]))) for d in g]
            poly = " ".join(f"{f(x)},{f(y)}" for x, y in cot + noc[::-1])
            b.append(f'<polygon points="{poly}" fill="{BLUE}" fill-opacity="0.1"/>')
            b.append(path(noc, RED)); b.append(path(cot, BLUE))
            b += [dot(x, y, RED) for x, y in noc]
            b += [dot(x, y, BLUE) for x, y in cot]
    open(OUT / "depth_sweep.svg", "w").write(svg(H, b))
    ALT["depth_sweep.svg"] = ("Small multiples of accuracy against depth for 16 stateful tasks (rows) on Qwen3.5-9B, "
                              "Llama-3.3-70B and DeepSeek-V4-Flash (columns), with chain of thought in blue and without "
                              "in red; the shaded band between them is the CoT gain, which is wide for many tasks, narrow for several, and "
                              "absent for multiplication.")
    return H


# ---------- 2. dumbbell: published vs rewritten ----------
def chart_dumbbell():
    pairs = [("hanoi", "published", "execute", "plan"), ("cellular_automaton", "ours", "cells3", "ours"),
             ("s5_composition", "published", "ergonomic", "published"),
             ("random_lookup_table", "published", "ergonomic", "published"),
             ("threesum", "published", "ergonomic", "published")]
    mods = [("qwen9b", "Qwen3.5-9B"), ("ds", "DeepSeek-V4-Flash")]
    X0, X1 = 130, 606
    TOP, GH, RH, GG = 58, 18, 22, 10
    xv = lambda a: X0 + a * (X1 - X0)
    H = TOP + len(pairs) * (GH + 2 * RH + GG) + 34
    b = []
    leg, _ = legend([("ring", BLUE, "published format (ours for cellular automaton)"), ("dot", BLUE, "rewritten variant")], 8, 14)
    b += leg
    b.append(T(8, 36, "Accuracy with chain of thought at the best depth; label = depth", fill=INK))
    bottom = H - 34
    for t in (0, 0.25, 0.5, 0.75, 1):
        b.append(T(xv(t), bottom + 14, f"{t:g}", "middle"))
    b.append(T((X0 + X1) / 2, bottom + 29, "accuracy with chain of thought", "middle", INK2))
    y = TOP
    for slug, pub, var, publab in pairs:
        b.append(T(4, y + 12, DISPLAY[slug], fill=INK2, weight="bold"))
        b.append(T(4 + 1.12 * tw(DISPLAY[slug], 11) + 10, y + 12, f"{publab} → {FMT_LABEL[var]}", fill=GRAY))
        y += GH
        for t in (0, 0.25, 0.5, 0.75, 1):   # grid only behind the data rows, not the task headers
            b.append(L(xv(t), y - 2, xv(t), y + 2 * RH + (GG if slug == pairs[-1][0] else 2), AXIS if t == 0 else GRID))
        for m, mname in mods:
            cy = y + RH / 2
            b.append(T(12, cy + 4, mname, fill=GRAY))
            p, v = surv[(slug, pub, m)], surv[(slug, var, m)]
            ap, av = float(p["best_acc_cot"]), float(v["best_acc_cot"])
            dp, dv = "d" + p["best_depth"], "d" + v["best_depth"]
            b.append(L(xv(ap), cy, xv(av), cy, BLUE, 2))
            b.append(ring(xv(ap), cy, BLUE, 6))
            b.append(dot(xv(av), cy, BLUE, 4))
            if av >= ap:   # published label left, variant right
                b.append(T(xv(ap) - 11, cy + 4, dp, "end", GRAY, 10))
                b.append(T(xv(av) + 11, cy + 4, dv, "start", GRAY, 10))
            else:
                b.append(T(xv(ap) + 11, cy + 4, dp, "start", GRAY, 10))
                b.append(T(xv(av) - 11, cy + 4, dv, "end", GRAY, 10))
            y += RH
        y += GG
    open(OUT / "dumbbell.svg", "w").write(svg(H, b))
    ALT["dumbbell.svg"] = ("Dumbbell chart of accuracy with chain of thought, published trace format versus a rewritten "
                           "variant, for five tasks on Qwen3.5-9B and DeepSeek-V4-Flash; the rewrite lifts Tower of Hanoi, "
                           "cellular automaton, S5 composition, random lookup table and 3SUM on the 9B model from "
                           "0.5 or below to 0.9 or above.")
    return H


# ---------- 3. gold knockout curves ----------
def chart_goldko():
    EMPH = ["random_lookup_table", "nested_arithmetic", "s5_composition", "cellular_automaton"]
    X0, X1, XO = 50, 360, 392
    Y0, Y1 = 64, 300
    H = 360
    xv = lambda k: X0 + k * (X1 - X0)
    yv = lambda a: Y1 - a * (Y1 - Y0)
    curves = {}
    for r in B3:
        if r["condition"] == "goldko":
            curves.setdefault((r["slug"], r["format"], int(r["depth"])), []).append((float(r["k_frac"]), float(r["acc"])))
    b = []
    leg, _ = legend([("line", BLUE, "gold trace through step k, then answer forced (four tasks highlighted in blue)")], 8, 14)
    b += leg
    leg2, _ = legend([("ring", BLUE, "own full trace at the same depth (with-CoT accuracy)")], 8, 32)
    b += leg2
    for t in (0, 0.25, 0.5, 0.75, 1):
        b.append(L(X0, yv(t), XO + 10, yv(t), AXIS if t == 0 else GRID))
        b.append(T(X0 - 6, yv(t) + 4, f"{t:g}", "end"))
    for t in (0, 0.25, 0.5, 0.75, 1):
        b.append(T(xv(t), Y1 + 15, f"{t:g}", "middle"))
    b.append(T((X0 + X1) / 2, Y1 + 32, "fraction of gold steps given (k / depth)", "middle", INK2))
    b.append(T(XO, Y0 - 12, "own trace", "middle", INK))
    b.append(f'<text transform="translate(13 {f((Y0 + Y1) / 2)}) rotate(-90)" text-anchor="middle" fill="{INK2}">accuracy</text>')
    keys = sorted(curves, key=lambda k: k[0] in EMPH)   # gray first, blue on top
    own = {}
    for key in keys:
        slug, fmt, d = key
        own[key] = float(sweep[(slug, fmt, "qwen9b", d)]["acc_cot"])
    for key in keys:
        slug = key[0]
        col = BLUE if slug in EMPH else GRAY
        pts = sorted(curves[key])
        xy = [(xv(k), yv(a)) for k, a in pts]
        # faint connector from last knockout point to own-trace marker: the jump
        b.append(L(xy[-1][0], xy[-1][1], XO, yv(own[key]), col, 1, ' stroke-opacity="0.45"'))
        b.append(path(xy, col, 2))
        if col == BLUE:
            b += [dot(x, y, BLUE) for x, y in xy]
    for key in keys:
        col = BLUE if key[0] in EMPH else GRAY
        b.append(ring(XO, yv(own[key]), col))
    # labels to the right of own-trace markers, spread with leader lines
    ks = list(own)
    ty = [yv(own[k]) for k in ks]
    ly = spread(ty, Y0 - 4, Y1 + 20, 14)
    LX = XO + 34
    for k, t, l in zip(ks, ty, ly):
        b.append(L(XO + 7, t, LX - 4, l, "#ccc", 1))
        b.append(T(LX, l + 4, f"{DISPLAY[k[0]]}  d{k[2]}", fill=INK2 if k[0] in EMPH else INK))
    open(OUT / "goldko.svg", "w").write(svg(H, b))
    ALT["goldko.svg"] = ("Line chart of Qwen3.5-9B accuracy when the gold trace is given through step k and the answer "
                         "is then forced, for ten tasks: nearly every curve stays near its no-CoT floor until the "
                         "last step, then jumps to high accuracy when the model writes its own full trace, with "
                         "program trace the exception that reaches 1.0 at k = d-1.")
    return H


# ---------- 4. propagation dot plot ----------
def chart_propagation():
    rows = sorted(B3S, key=lambda r: (r["slug"] == "threesum", -float(r["mean_propagation"])))
    X0, X1 = 168, 600
    lo, hi = -0.2, 1.0
    TOP, RH = 52, 22
    H = TOP + RH * len(rows) + 48
    xv = lambda v: X0 + (v - lo) / (hi - lo) * (X1 - X0)
    b = []
    leg, _ = legend([("bigdot", BLUE, "mean propagation"), ("dot", TINT, "at k = d/4, d/2, 3d/4")], 8, 14)
    b += leg
    top, bot = TOP - 6, TOP + RH * len(rows)
    for t in (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0):
        if t in (0, 1.0):
            b.append(L(xv(t), top, xv(t), bot, AXIS if t == 0 else GRID))
        b.append(T(xv(t), bot + 15, f"{t:g}".replace("-", "−"), "middle"))
    b.append(T((X0 + X1) / 2, bot + 34, "propagation = clean-continuation accuracy − accuracy after a corrupted step",
               "middle", INK2))
    for i, r in enumerate(rows):
        cy = TOP + i * RH + RH / 2
        slug = r["slug"]
        b.append(T(4, cy + 4, DISPLAY[slug], fill=INK2))
        qs = [float(r[c]) for c in ("prop_q1", "prop_half", "prop_q3")]
        mp = float(r["mean_propagation"])
        b.append(L(xv(min(qs + [mp])), cy, xv(max(qs + [mp])), cy, GRID, 1))
        b += [dot(xv(q), cy, TINT, 4) for q in qs]
        b.append(dot(xv(mp), cy, BLUE, 6))
        right = xv(max(qs + [mp])) + 12
        if slug == "threesum":
            b.append(T(right, cy + 4, f"{mp:.2f}".replace("-", "−") + "  undefined: clean continuation fails", fill=GRAY))
        elif slug == "hanoi":
            b.append(T(right, cy + 4, "model detects the corruption", fill=GRAY))
    open(OUT / "propagation.svg", "w").write(svg(H, b))
    ALT["propagation.svg"] = ("Dot plot of mistake propagation for ten tasks on Qwen3.5-9B: random lookup table and "
                              "nested arithmetic carry a corrupted step through to the answer almost always, Tower of "
                              "Hanoi much less because the model detects the corruption, and 3SUM is negative because "
                              "its clean continuation already fails.")
    return H


# ---------- 5. tokens per step ----------
def chart_tokens():
    rows = [r for r in MASTER if r["primary"] == "True" and r["slug"] != "addition"
            and (num(r["tok_per_step_9b"]) or num(r["tok_per_step_ds"]))]
    rows.sort(key=lambda r: (num(r["tok_per_step_9b"]) is None, num(r["tok_per_step_9b"]) or 0))
    X0, X1 = 196, 610
    lo, hi = math.log10(3), math.log10(1000)
    TOP, RH = 50, 20
    H = TOP + RH * len(rows) + 46
    xv = lambda v: X0 + (math.log10(v) - lo) / (hi - lo) * (X1 - X0)
    b = []
    leg, _ = legend([("dot", BLUE, "9B spend"), ("dot", RED, "DeepSeek spend"), ("tick", GRAY, "gold trace")], 8, 14)
    b += leg
    top, bot = TOP - 6, TOP + RH * len(rows)
    for t in (3, 10, 30, 100, 300, 1000):
        b.append(L(xv(t), top, xv(t), bot, AXIS if t == 3 else GRID))
        b.append(T(xv(t), bot + 15, f"{t:,}", "middle"))
    b.append(T((X0 + X1) / 2, bot + 34, "output tokens per step at the best passing depth (log scale)", "middle", INK2))
    notes = []
    for i, r in enumerate(rows):
        cy = TOP + i * RH + RH / 2
        slug, fmt = r["slug"], r["format"]
        b.append(T(4, cy + 4, DISPLAY[slug], fill=INK2))
        if fmt != "published":
            b.append(T(4 + tw(DISPLAY[slug]) + 5, cy + 4, f"({FMT_LABEL[fmt]})", fill=GRAY))
        gold = None
        for m in ("qwen9b", "ds"):
            s = surv.get((slug, fmt, m))
            if not s:
                continue
            row = sweep.get((slug, fmt, m, int(s["best_depth"])))
            tps, ratio = num(row["tok_per_step"]), num(row["cot_len_ratio_vs_gold"])
            if tps and ratio:
                gold = tps / ratio
                if m == "ds":
                    notes.append(f"{slug}/{fmt}: gold from ds row")
                break
        if gold:
            b.append(L(xv(gold), cy - 8, xv(gold), cy + 8, GRAY, 2))
        t9, tds = num(r["tok_per_step_9b"]), num(r["tok_per_step_ds"])
        if tds:
            b.append(dot(xv(tds), cy, RED, 4.5))
        if t9:
            b.append(dot(xv(t9), cy, BLUE, 4.5))
    open(OUT / "tokens.svg", "w").write(svg(H, b))
    ALT["tokens.svg"] = ("Dot plot of chain-of-thought tokens per step for each task's surviving format, on a log scale "
                         "from 3 to 1000, for 14 tasks, comparing Qwen3.5-9B and DeepSeek-V4-Flash spend with the gold trace's cost; "
                         "most tasks cost 10 to 100 tokens per step, while cellular automaton and 3SUM run to "
                         "several hundred.")
    return H, notes


if __name__ == "__main__":
    sizes = {"depth_sweep.svg": chart_depth_sweep(), "dumbbell.svg": chart_dumbbell(), "goldko.svg": chart_goldko(),
             "propagation.svg": chart_propagation()}
    h, notes = chart_tokens()
    sizes["tokens.svg"] = h
    json.dump(ALT, open(HERE / "charts_alt.json", "w"), indent=2, ensure_ascii=False)
    for k, v in sizes.items():
        print(f"{k}: {W}x{v}")
    for n in notes:
        print("note:", n)
