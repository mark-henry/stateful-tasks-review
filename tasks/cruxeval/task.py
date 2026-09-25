"""cruxeval — CRUXEval-O (output prediction) wrapped in the stateful-tasks interface.

Source: Gu, Roziere, Leather, Solar-Lezama, Synnaeve, Wang.
"CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution."
arXiv:2401.03065 (ICML 2024). Repo: facebookresearch/cruxeval (MIT).

The fixed 800-row dataset is read from vendor/; see README.md for the format
decision, depth semantics, and caveats.  Pure stdlib; subprocess is used to run
reference code (ground truth) and to trace it (gold scratchpad).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "vendor", "cruxeval_repo", "data", "cruxeval.jsonl")

_ROWS_CACHE: list[dict] | None = None


def _rows() -> list[dict]:
    """The 800 vendored CRUXEval rows: {code, input, output, id}."""
    global _ROWS_CACHE
    if _ROWS_CACHE is None:
        with open(DATA_PATH, "r", encoding="utf-8") as fh:
            _ROWS_CACHE = [json.loads(ln) for ln in fh if ln.strip()]
    return _ROWS_CACHE


# --------------------------------------------------------------------------
# interface constants
# --------------------------------------------------------------------------

ANSWER_FORMAT = "a Python literal, the value returned by the function"

# AMENDMENT 4 (prompt redaction).  The input argument is the only per-instance
# "operator" a CRUXEval prompt carries — the code is static material — so the
# redaction is all-or-nothing at k >= 1.  See README.md :: Redaction.
REDACTION_MEANINGFUL = True
REDACTION_PLACEHOLDER = "[…]"

# Depth = number of executed statements (sys.settrace 'line' events) inside f.
# These are the exact executed-line buckets of the fixed 800-row dataset that
# hold >= 16 instances, spread roughly geometrically.  See README.md.
DEPTHS = [1, 2, 4, 6, 9, 12, 15]

KNOBS: dict = {
    "tolerance": (
        0,
        "widen the executed-line bucket to [depth-tolerance, depth+tolerance]; "
        "lets the harness reach sparsely-populated deep buckets, but then "
        "len(steps) != depth",
    ),
    "state_in_step": (
        "delta",
        "how the state is rendered inside a step line: 'delta' (only the "
        "variables this line changed), 'full' (all locals), 'none'",
    ),
    "max_state_chars": (
        120,
        "truncation length for the rendered state string in steps/states",
    ),
    "max_repr_chars": (
        80,
        "truncation length for a single variable's repr inside the tracer",
    ),
    "timeout": (
        10.0,
        "seconds allowed for each reference-execution / tracing subprocess",
    ),
}

DEFAULT_TIMEOUT = 10.0
_TRACE_FILENAME = "<cruxeval>"


@dataclass
class Instance:
    prompt: str
    steps: list[str]
    states: list[str]
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# subprocess workers
# --------------------------------------------------------------------------

# Worker A: execute f(input) and report repr(result).  This is the GROUND TRUTH
# path used by solve(); it shares no code with the generator, which reads the
# answer out of the dataset's `output` column.
_EXEC_SRC = r'''
import json, sys
p = json.loads(sys.stdin.read())
out = []
for item in p["items"]:
    try:
        g = {}
        src = item["code"] + "\n__cruxeval_result__ = f(" + item["input"] + ")\n"
        exec(compile(src, "<cruxeval-exec>", "exec"), g)
        out.append({"ok": True, "repr": repr(g["__cruxeval_result__"])})
    except BaseException as e:
        out.append({"ok": False, "err": type(e).__name__ + ": " + str(e)[:200]})
sys.stdout.write(json.dumps(out))
'''

# Worker B: sys.settrace the reference execution and report the event stream.
_TRACE_SRC = r'''
import json, sys
p = json.loads(sys.stdin.read())
FN = "<cruxeval>"
MAXR = p.get("max_repr", 80)
MAXS = p.get("max_steps", 20000)
COUNT_ONLY = p.get("count_only", False)

def _r(v):
    try:
        s = repr(v)
    except BaseException:
        return "<unrepr>"
    if len(s) > MAXR:
        s = s[: MAXR - 3] + "..."
    return s

def run(code, inp):
    src = code + "\n__cruxeval_result__ = f(" + inp + ")\n"
    events = []
    order = [0]

    def snap(fr):
        try:
            return {k: _r(v) for k, v in fr.f_locals.items()}
        except BaseException:
            return {}

    def gt(fr, ev, arg):
        if ev == "call" and fr.f_code.co_name == "f" and fr.f_code.co_filename == FN:
            order[0] += 1
            o = order[0]
            if not COUNT_ONLY:
                events.append({"t": "call", "line": fr.f_lineno, "o": o, "v": snap(fr)})

            def lt(f2, e2, a2, _o=o):
                if len(events) >= MAXS:
                    raise RuntimeError("cruxeval: step limit exceeded")
                if e2 == "line":
                    if COUNT_ONLY:
                        events.append(0)
                    else:
                        events.append({"t": "line", "line": f2.f_lineno,
                                       "o": _o, "v": snap(f2)})
                elif e2 == "return" and not COUNT_ONLY:
                    events.append({"t": "return", "line": f2.f_lineno, "o": _o,
                                   "v": snap(f2), "r": _r(a2)})
                return lt

            return lt
        return None

    g = {}
    sys.settrace(gt)
    try:
        exec(compile(src, FN, "exec"), g)
    finally:
        sys.settrace(None)
    res = g.get("__cruxeval_result__")
    if COUNT_ONLY:
        return {"ok": True, "n": len(events)}
    return {"ok": True, "events": events, "result": repr(res),
            "calls": order[0],
            "n": sum(1 for e in events if e["t"] == "line")}

out = []
for item in p["items"]:
    try:
        out.append(run(item["code"], item["input"]))
    except BaseException as e:
        out.append({"ok": False, "err": type(e).__name__ + ": " + str(e)[:200]})
sys.stdout.write(json.dumps(out))
'''


def _run_worker(src: str, payload: dict, timeout: float) -> list:
    proc = subprocess.run(
        [sys.executable, "-I", "-c", src],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "cruxeval worker failed (rc=%d): %s"
            % (proc.returncode, proc.stderr.decode("utf-8", "replace")[-500:])
        )
    return json.loads(proc.stdout.decode("utf-8"))


# --------------------------------------------------------------------------
# depth index (executed-line count per dataset row)
# --------------------------------------------------------------------------

_INDEX_CACHE: dict | None = None


def _index(timeout: float = 120.0) -> dict:
    """{'depth_of': {id: n}, 'buckets': {n: [id, ...]}, 'skipped': [...]}.

    One subprocess traces all 800 rows in count-only mode (~0.1 s).
    """
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    rows = _rows()
    payload = {
        "count_only": True,
        "items": [{"code": r["code"], "input": r["input"]} for r in rows],
    }
    results = _run_worker(_TRACE_SRC, payload, timeout)
    depth_of: dict[str, int] = {}
    buckets: dict[int, list[str]] = {}
    skipped = []
    for row, res in zip(rows, results):
        if not res.get("ok"):
            skipped.append((row["id"], res.get("err", "?")))
            continue
        n = res["n"]
        depth_of[row["id"]] = n
        buckets.setdefault(n, []).append(row["id"])
    for ids in buckets.values():
        ids.sort()
    _INDEX_CACHE = {
        "depth_of": depth_of,
        "buckets": buckets,
        "skipped": skipped,
        "python": ".".join(str(x) for x in sys.version_info[:3]),
    }
    return _INDEX_CACHE


def bucket_sizes() -> dict[int, int]:
    """Executed-line count -> number of dataset instances with that count."""
    return {k: len(v) for k, v in sorted(_index()["buckets"].items())}


# --------------------------------------------------------------------------
# trace -> steps / states
# --------------------------------------------------------------------------

_TRACE_CACHE: dict[tuple[str, str], dict] = {}


def _trace(code: str, inp: str, max_repr: int, timeout: float) -> dict:
    key = (code, inp)
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    payload = {"max_repr": max_repr, "items": [{"code": code, "input": inp}]}
    res = _run_worker(_TRACE_SRC, payload, timeout)[0]
    if not res.get("ok"):
        raise RuntimeError("cruxeval: could not trace instance: %s" % res.get("err"))
    _TRACE_CACHE[key] = res
    return res


def _trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def _fmt_bindings(d: dict, max_chars: int) -> str:
    if not d:
        return "{}"
    return _trunc(", ".join("%s = %s" % (k, v) for k, v in d.items()), max_chars)


def _delta(before: dict, after: dict) -> dict:
    return {k: v for k, v in after.items() if before.get(k) != v}


def _build_steps(code: str, trace: dict, answer: str, state_in_step: str,
                 max_state_chars: int) -> tuple[list[str], list[str]]:
    """Render the settrace event stream as numbered published-style step lines."""
    src_lines = code.splitlines()
    events = trace["events"]
    multi = trace.get("calls", 1) > 1

    line_idx = [i for i, e in enumerate(events) if e["t"] == "line"]
    steps: list[str] = []
    states: list[str] = []

    for k, i in enumerate(line_idx):
        ev = events[i]
        lineno, ordinal = ev["line"], ev["o"]
        text = src_lines[lineno - 1].strip() if 0 < lineno <= len(src_lines) else "?"

        # the next event belonging to the same frame tells us the state after
        # this statement finished (settrace fires 'line' BEFORE the line runs)
        nxt = None
        for j in range(i + 1, len(events)):
            if events[j]["o"] == ordinal:
                nxt = events[j]
                break

        if k == len(line_idx) - 1:
            # last executed statement overall: the function hands back its value
            after = nxt["v"] if nxt else ev["v"]
            effect = "returns %s" % answer
            state = _trunc("return = %s" % answer, max_state_chars)
        elif nxt is not None and nxt["t"] == "return":
            after = nxt["v"]
            effect = "returns %s" % nxt["r"]
            state = _trunc("return = %s" % nxt["r"], max_state_chars)
        else:
            after = nxt["v"] if nxt else ev["v"]
            d = _delta(ev["v"], after)
            effect = _fmt_bindings(d, max_state_chars) if d else "no change"
            state = _fmt_bindings(after, max_state_chars)

        where = "line %d" % lineno
        if multi:
            where += " (call %d)" % ordinal

        if state_in_step == "none":
            body = "%s: %s" % (where, text)
        elif state_in_step == "full":
            body = "%s: %s -> %s" % (
                where, text, _fmt_bindings(after, max_state_chars))
        else:  # "delta"
            body = "%s: %s -> %s" % (where, text, effect)

        steps.append("%d. %s" % (k + 1, body))
        states.append(state)

    return steps, states


# --------------------------------------------------------------------------
# generate / solve / check / format_cot / exemplars
# --------------------------------------------------------------------------

def make_prompt(code: str, inp: str) -> str:
    """The published CRUXEval-O problem statement (prompts.py, no instruction)."""
    return "[PYTHON]\n%s\nassert f(%s) == ??\n[/PYTHON]" % (code, inp)


# matches the published prompt's final assertion line, capturing the input
# argument between `assert f(` and `) == ??`
_PROMPT_ASSERT_RE = re.compile(
    r"(?m)^(?P<head>[ \t]*assert[ \t]+f\()(?P<arg>.*)(?P<tail>\)[ \t]*==[ \t]*\?\?[ \t]*)$"
)


def redact_prompt(inst: Instance, k: int) -> str:
    """AMENDMENT 4: hide what steps 1..k consumed, keep what k+1..depth need.

    A CRUXEval prompt is `[PYTHON] <code> assert f(<input>) == ?? [/PYTHON]`.
    The code is *static material* (the rule table of this task) and every
    remaining step needs it, so it is kept; the input argument value is the
    only thing the model could read the initial state back out of, so for
    k >= 1 it is replaced by a single `[…]` placeholder.  This is the closest
    available analogue of operator-wise redaction: the inputs are not split
    per step, so redaction is all-or-nothing (see README.md :: Redaction).
    """
    if k < 0:
        raise ValueError("cruxeval: k must be >= 0")
    if k == 0:
        return inst.prompt

    code, inp = inst.meta.get("code"), inst.meta.get("input")
    if code is not None and inp is not None and inst.prompt == make_prompt(code, inp):
        return make_prompt(code, REDACTION_PLACEHOLDER)

    # fallback: rewrite the last `assert f(...) == ??` line in place
    matches = list(_PROMPT_ASSERT_RE.finditer(inst.prompt))
    if not matches:
        raise ValueError("cruxeval: prompt has no `assert f(...) == ??` line")
    m = matches[-1]
    return (
        inst.prompt[: m.start()]
        + m.group("head")
        + REDACTION_PLACEHOLDER
        + m.group("tail")
        + inst.prompt[m.end():]
    )


def _pick(depth: int, seed: int, tolerance: int) -> dict:
    idx = _index()
    buckets = idx["buckets"]
    lo, hi = depth - tolerance, depth + tolerance
    pool = sorted(
        i for n, ids in buckets.items() if lo <= n <= hi for i in ids
    )
    if not pool:
        avail = ", ".join(
            "%d(%d)" % (n, len(ids)) for n, ids in sorted(buckets.items())
        )
        raise ValueError(
            "cruxeval: no dataset instance with executed-line count in "
            "[%d, %d]. Populated buckets: %s" % (lo, hi, avail)
        )
    # deterministic shuffle of the bucket, indexed by seed -> distinct
    # instances for seeds 0..len(pool)-1, wrapping afterwards
    order = list(pool)
    random.Random("cruxeval|bucket|%d|%d" % (depth, tolerance)).shuffle(order)
    chosen = order[seed % len(order)]
    by_id = {r["id"]: r for r in _rows()}
    return by_id[chosen]


def generate(depth: int, seed: int, **knobs) -> Instance:
    tolerance = int(knobs.get("tolerance", KNOBS["tolerance"][0]))
    state_in_step = knobs.get("state_in_step", KNOBS["state_in_step"][0])
    max_state_chars = int(knobs.get("max_state_chars", KNOBS["max_state_chars"][0]))
    max_repr_chars = int(knobs.get("max_repr_chars", KNOBS["max_repr_chars"][0]))
    timeout = float(knobs.get("timeout", KNOBS["timeout"][0]))
    if state_in_step not in ("delta", "full", "none"):
        raise ValueError("state_in_step must be 'delta', 'full' or 'none'")

    row = _pick(depth, seed, tolerance)
    answer = row["output"]
    trace = _trace(row["code"], row["input"], max_repr_chars, timeout)
    steps, states = _build_steps(
        row["code"], trace, answer, state_in_step, max_state_chars
    )

    n = len(steps)
    inst = Instance(
        prompt=make_prompt(row["code"], row["input"]),
        steps=steps,
        states=states,
        answer=answer,
        depth=n,
        meta={
            "id": row["id"],
            "code": row["code"],
            "input": row["input"],
            "requested_depth": depth,
            "executed_lines": n,
            "calls": trace.get("calls", 1),
            "seed": seed,
            "tolerance": tolerance,
            "state_in_step": state_in_step,
            "max_state_chars": max_state_chars,
            "timeout": timeout,
            "python": _index()["python"],
            "closing_line": "%d. The return value of the function is therefore %s."
            % (n + 1, answer),
        },
    )
    return inst


def solve(inst: Instance) -> str:
    """Reference solver: run the function in a subprocess, return repr(result).

    Independent of generate(), which copies the answer out of the dataset's
    `output` column; this path never looks at that column.
    """
    code = inst.meta.get("code")
    inp = inst.meta.get("input")
    if code is None or inp is None:
        raise ValueError("cruxeval: instance.meta lacks 'code'/'input'")
    timeout = float(inst.meta.get("timeout", DEFAULT_TIMEOUT))
    key = (code, inp)
    cached = _SOLVE_CACHE.get(key)
    if cached is None:
        res = _run_worker(
            _EXEC_SRC, {"items": [{"code": code, "input": inp}]}, timeout
        )[0]
        if not res.get("ok"):
            raise RuntimeError("cruxeval: reference execution failed: %s"
                               % res.get("err"))
        cached = res["repr"]
        _SOLVE_CACHE[key] = cached
    return cached


_SOLVE_CACHE: dict[tuple[str, str], str] = {}


def format_cot(inst: Instance) -> str:
    """Gold trace in the published [THOUGHT]/[ANSWER] format + 'Answer: X'."""
    parts = ["[THOUGHT]", "Let's execute the code step by step:", ""]
    parts.extend(inst.steps)
    closing = inst.meta.get("closing_line")
    if closing:
        parts.append(closing)
    parts.append("[/THOUGHT]")
    parts.append("[ANSWER]")
    parts.append("assert f(%s) == %s" % (inst.meta.get("input", "..."), inst.answer))
    parts.append("[/ANSWER]")
    parts.append("Answer: %s" % inst.answer)
    return "\n".join(parts)


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    """Char offsets of each step within format_cot(inst)."""
    text = format_cot(inst)
    spans = []
    cur = 0
    for s in inst.steps:
        i = text.index(s, cur)
        spans.append((i, i + len(s)))
        cur = i + len(s)
    return spans


# --- answer extraction + comparison (mirrors the official cruxeval scorer) ---

_ASSERT_RE = re.compile(r"^\s*assert\s+f\s*\(")
_SET_CALL_RE = re.compile(r"^(set|frozenset)\((.*)\)$", re.S)
_SPECIAL = {
    "set()": set(),
    "frozenset()": frozenset(),
    "dict()": {},
    "list()": [],
    "tuple()": (),
    "float('inf')": float("inf"),
    'float("inf")': float("inf"),
    "-float('inf')": float("-inf"),
    '-float("inf")': float("-inf"),
    "float('nan')": float("nan"),
    'float("nan")': float("nan"),
}
_FAILED = object()


def _literal(s: str):
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    if s in _SPECIAL:
        return _SPECIAL[s]
    m = _SET_CALL_RE.match(s)
    if m:
        try:
            inner = ast.literal_eval(m.group(2)) if m.group(2).strip() else []
            return set(inner) if m.group(1) == "set" else frozenset(inner)
        except Exception:
            pass
    return _FAILED


def extract_answer(completion: str) -> str:
    """Last 'Answer:' line; else the [ANSWER] block; else last non-empty line."""
    lines = completion.splitlines()
    for ln in reversed(lines):
        t = ln.strip()
        if t.lower().startswith("answer:"):
            return _strip_assertion(t[len("answer:"):].strip())
    if "[ANSWER]" in completion:
        seg = completion.rsplit("[ANSWER]", 1)[1]
        seg = seg.split("[/ANSWER]")[0]
        for ln in seg.splitlines():
            if ln.strip():
                return _strip_assertion(ln.strip())
    for ln in reversed(lines):
        if ln.strip():
            return _strip_assertion(ln.strip())
    return ""


def _strip_assertion(s: str) -> str:
    s = s.strip()
    if _ASSERT_RE.match(s) and "==" in s:
        s = s.split("==", 1)[1].strip()
    # drop a trailing comment only when it cannot be inside a string literal
    if "#" in s and _literal(s) is _FAILED:
        cand = s.split("#", 1)[0].strip()
        if cand and _literal(cand) is not _FAILED:
            s = cand
    return s.strip().rstrip(";").strip()


def answers_equal(pred: str, gold: str) -> bool:
    a, b = _literal(pred), _literal(gold)
    if a is not _FAILED and b is not _FAILED:
        try:
            if a == b:
                return True
        except Exception:
            pass
        try:
            return repr(a) == repr(b)
        except Exception:
            return False
    return " ".join(pred.split()) == " ".join(gold.split())


def check(inst: Instance, completion: str) -> bool:
    pred = extract_answer(completion)
    if not pred:
        return False
    # official scorer's anti-cheat guard: refuse a "prediction" that just
    # re-states the call instead of a literal
    inp = inst.meta.get("input")
    if inp is not None and ("f(%s)" % inp) in pred:
        return False
    return answers_equal(pred, inst.answer)


# --- exemplars -------------------------------------------------------------

# Verbatim from vendor/cruxeval_repo/prompts.py :: make_cot_output_prompt()
# (commit 190faf16d175b5847b0af05d937872b1fb395942, MIT). See published_trace.txt.
PUBLISHED_CODE = 'def f(s):\n    s = s + s\n    return "b" + s + "a"'
PUBLISHED_INPUT = '"hi"'
PUBLISHED_ANSWER = '"bhihia"'
PUBLISHED_STEPS = [
    "1. The function f is defined, which takes a single argument s.",
    '2. The function is called with the argument "hi", so within the function, '
    's is initially "hi".',
    "3. Inside the function, s is concatenated with itself, so s becomes "
    '"hihi".',
    '4. The function then returns a new string that starts with "b", followed '
    "by the value of s (which is now \"hihi\"), and ends with \"a\".",
    '5. The return value of the function is therefore "bhihia".',
]
PUBLISHED_STATES = [
    "f defined",
    "s = 'hi'",
    "s = 'hihi'",
    "s = 'hihi'",
    "return = 'bhihia'",
]


def published_exemplar() -> Instance:
    return Instance(
        prompt=make_prompt(PUBLISHED_CODE, PUBLISHED_INPUT),
        steps=list(PUBLISHED_STEPS),
        states=list(PUBLISHED_STATES),
        answer=PUBLISHED_ANSWER,
        depth=len(PUBLISHED_STEPS),
        meta={
            "id": "published_exemplar",
            "code": PUBLISHED_CODE,
            "input": PUBLISHED_INPUT,
            "published_exemplar": True,
            "executed_lines": 2,
            "closing_line": None,
            "note": "verbatim hand-written narration from prompts.py; its 5 "
                    "sentences are NOT the mechanical executed-line trace "
                    "(that would be 2 steps)",
        },
    )


_EXEMPLAR_DEPTHS = [3, 5, 2, 4, 6, 3, 5, 2]


def exemplars(k: int, seed: int) -> list[Instance]:
    out = [published_exemplar()]
    i = 0
    while len(out) < k:
        d = _EXEMPLAR_DEPTHS[i % len(_EXEMPLAR_DEPTHS)]
        out.append(generate(d, 10_000 + seed * 97 + i))
        i += 1
    return out[:k]


# --------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------

def _wrong_answer(answer: str) -> str:
    v = _literal(answer)
    if v is not _FAILED:
        try:
            if isinstance(v, bool):
                return repr(not v)
            if isinstance(v, int):
                return repr(v + 1)
            if isinstance(v, float):
                return repr(v + 1.0)
            if isinstance(v, str):
                return repr(v + "Z")
            if isinstance(v, list):
                return repr(v + ["__cruxeval_wrong__"])
            if isinstance(v, tuple):
                return repr(v + ("__cruxeval_wrong__",))
        except Exception:
            pass
    return repr(("__cruxeval_wrong__",))


def selftest(n: int = 200) -> int:
    idx = _index()
    sizes = bucket_sizes()
    print("python %s; %d rows; %d executed-line buckets; skipped %d"
          % (idx["python"], len(_rows()), len(sizes), len(idx["skipped"])))
    print("DEPTHS %s -> bucket sizes %s"
          % (DEPTHS, {d: sizes.get(d, 0) for d in DEPTHS}))
    for d in DEPTHS:
        if sizes.get(d, 0) == 0:
            print("FAIL: DEPTHS contains empty bucket %d" % d)
            return 1

    rng = random.Random(20260916)
    failures = 0
    seen_ids = set()
    for t in range(n):
        depth = rng.choice(DEPTHS)
        seed = rng.randrange(10_000)
        inst = generate(depth, seed)
        seen_ids.add(inst.meta["id"])

        if not (len(inst.steps) == len(inst.states) == inst.depth == depth):
            print("FAIL[%d] depth/steps/states mismatch: d=%d steps=%d states=%d "
                  "inst.depth=%d id=%s"
                  % (t, depth, len(inst.steps), len(inst.states), inst.depth,
                     inst.meta["id"]))
            failures += 1

        got = solve(inst)
        if got != inst.answer:
            print("FAIL[%d] solve()!=answer for %s: %r vs %r"
                  % (t, inst.meta["id"], got, inst.answer))
            failures += 1

        if not check(inst, format_cot(inst)):
            print("FAIL[%d] check(gold) False for %s" % (t, inst.meta["id"]))
            failures += 1

        wrong = _wrong_answer(inst.answer)
        if wrong != inst.answer and check(inst, "Answer: %s" % wrong):
            print("FAIL[%d] check(wrong) True for %s (wrong=%s)"
                  % (t, inst.meta["id"], wrong))
            failures += 1

        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer, again.depth) != (
            inst.prompt, inst.steps, inst.states, inst.answer, inst.depth
        ):
            print("FAIL[%d] generate not deterministic (d=%d seed=%d)"
                  % (t, depth, seed))
            failures += 1

        # bare literal (no 'Answer:' line) must still be accepted
        if not check(inst, "[ANSWER]\nassert f(%s) == %s\n[/ANSWER]"
                     % (inst.meta["input"], inst.answer)):
            print("FAIL[%d] check([ANSWER] block) False for %s" % (t, inst.meta["id"]))
            failures += 1

    # redaction (AMENDMENT 4)
    rrng = random.Random(20260918)
    for t in range(20):
        depth = rrng.choice(DEPTHS)
        inst = generate(depth, rrng.randrange(10_000))
        inp = inst.meta["input"]
        code = inst.meta["code"]

        if redact_prompt(inst, 0) != inst.prompt:
            print("FAIL[r%d] redact_prompt(inst, 0) != prompt for %s"
                  % (t, inst.meta["id"]))
            failures += 1

        for k in (1, inst.depth // 2, inst.depth):
            if k < 1:
                continue
            red = redact_prompt(inst, k)
            if red == inst.prompt:
                print("FAIL[r%d] redaction at k=%d is a no-op for %s"
                      % (t, k, inst.meta["id"]))
                failures += 1
            if REDACTION_PLACEHOLDER not in red:
                print("FAIL[r%d] redaction at k=%d lacks the placeholder for %s"
                      % (t, k, inst.meta["id"]))
                failures += 1
            # all-or-nothing: every k >= 1 redacts the same thing
            if red != redact_prompt(inst, 1):
                print("FAIL[r%d] redaction at k=%d differs from k=1 for %s"
                      % (t, k, inst.meta["id"]))
                failures += 1

        # task-specific: at k == depth the input argument is gone and only the
        # static code + the question remain
        full = redact_prompt(inst, inst.depth)
        if full != make_prompt(code, REDACTION_PLACEHOLDER):
            print("FAIL[r%d] k=depth redaction is not `assert f(%s) == ??` for %s"
                  % (t, REDACTION_PLACEHOLDER, inst.meta["id"]))
            failures += 1
        if code not in full or "== ??" not in full:
            print("FAIL[r%d] k=depth redaction dropped static material for %s"
                  % (t, inst.meta["id"]))
            failures += 1
        # the input tokens survive only if the code itself contains them
        if inp not in code and inp in full:
            print("FAIL[r%d] k=depth redaction still shows the input %r for %s"
                  % (t, inp, inst.meta["id"]))
            failures += 1
        if inp.strip() and inp in full.splitlines()[-2]:
            print("FAIL[r%d] k=depth redaction leaves the input on the assert "
                  "line for %s" % (t, inst.meta["id"]))
            failures += 1

    pe = published_exemplar()
    if redact_prompt(pe, 0) != pe.prompt or PUBLISHED_INPUT in redact_prompt(pe, 1):
        print("FAIL redaction of the published exemplar")
        failures += 1

    if not REDACTION_MEANINGFUL:
        print("FAIL REDACTION_MEANINGFUL should be True for cruxeval")
        failures += 1

    # knobs
    for sis in ("delta", "full", "none"):
        i2 = generate(4, 3, state_in_step=sis)
        if len(i2.steps) != 4:
            print("FAIL knob state_in_step=%s" % sis)
            failures += 1
    i3 = generate(20, 0, tolerance=4)
    if not (16 <= i3.meta["executed_lines"] <= 24):
        print("FAIL knob tolerance: got %d" % i3.meta["executed_lines"])
        failures += 1

    # exemplars
    ex = exemplars(3, 0)
    if len(ex) != 3 or not ex[0].meta.get("published_exemplar"):
        print("FAIL exemplars(3, 0) shape")
        failures += 1
    for e in ex:
        if len(e.steps) != len(e.states) or len(e.steps) != e.depth:
            print("FAIL exemplar steps/states/depth mismatch: %s" % e.meta["id"])
            failures += 1
        if not check(e, format_cot(e)):
            print("FAIL exemplar check(gold) False: %s" % e.meta["id"])
            failures += 1
        if not format_cot(e).strip():
            print("FAIL exemplar renders empty")
            failures += 1
    # the published exemplar's answer uses double quotes; compare as literals
    if not answers_equal(solve(ex[0]), ex[0].answer):
        print("FAIL published exemplar solve(): %r vs %r"
              % (solve(ex[0]), ex[0].answer))
        failures += 1

    # unpopulated depth must be a clean error
    try:
        generate(999, 0)
        print("FAIL generate(999, 0) should raise")
        failures += 1
    except ValueError:
        pass

    print("selftest: %d instances, %d distinct dataset rows, %d failures"
          % (n, len(seen_ids), failures))
    print("PASS" if failures == 0 else "FAIL")
    return 0 if failures == 0 else 1


def demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for seed in range(3):
            inst = generate(depth, seed)
            print("=" * 78)
            print("depth=%d seed=%d id=%s executed_lines=%d"
                  % (depth, seed, inst.meta["id"], inst.meta["executed_lines"]))
            print("-" * 78)
            print(inst.prompt)
            print("-" * 78)
            print(format_cot(inst))
            print()


def _main() -> int:
    ap = argparse.ArgumentParser(description="cruxeval (CRUXEval-O) task module")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--buckets", action="store_true",
                    help="print the executed-line bucket histogram")
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()
    if args.buckets:
        for n, c in bucket_sizes().items():
            print("%4d  %d" % (n, c))
        return 0
    if args.selftest:
        return selftest(args.n)
    if args.demo:
        demo()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
