"""
synthetic_program_trace -- tiny integer Python programs traced line-by-line with the
Nye et al. (2021) scratchpad format.

Published format (Nye, Andreassen, Gur-Ari, Michalewski, Austin, Bieber, Dohan, Lewkowycz,
Bosma, Luan, Sutton, Odena, "Show Your Work: Scratchpads for Intermediate Computation with
Language Models", arXiv:2112.00114, Appendix C, "Example few-shot prompt for synthetic Python
experiments"), reproduced character-for-character by this module:

    [BEGIN]

    state: {}
    line: def f(v0):
    state: {"f": "<callable_object f>"}
    line: output = f(6)
    state: {"v0": 6}
    line:   v0 += 0
    state: {"v0": 6}
    ...
    line:   return v0
    state: {"f": "<callable_object f>", "output": 24}

    [DONE]

One STEP is one `line:` / `state:` pair, i.e. one executed statement, with loops fully
unrolled (Nye et al. Figure 1: "all loops are unrolled across time").  The leading
`state: {}` is a prologue, not a step.  depth == number of executed statements ==
len(steps) == len(states), exactly.

The answer is the final value of `output` (the published trace's own last state line is
`{"f": "<callable_object f>", "output": 24}`).  See README.md for the format decision, depth
semantics, ANSWER_FORMAT rationale, sourcing and caveats.  Pure python + stdlib.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------

@dataclass
class Instance:
    prompt: str          # problem statement only, published wording
    steps: list[str]     # gold trace, one element per executed statement ("line: ...\nstate: {...}")
    states: list[str]    # state after each step: the JSON state dict, e.g. '{"v0": 12, "v4": 1}'
    answer: str          # exact-match target: the final value of `output`, a plain integer
    depth: int           # number of executed statements == len(steps)
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a single integer, the final value of output (e.g. 24)"

# Justified in README.md.  depth is executed-statement count, of which 3 are fixed overhead
# (def / call / return), so depth 6 is three body statements (no loop fits under 4 free
# statements) and depth 12 is exactly the published Appendix C example's length.
DEPTHS = [6, 12, 20, 32, 48, 72]

KNOBS = {
    "n_vars": (
        3,
        "number of distinct variables, including the parameter v0 and any loop counters "
        "(>= 2; the published example uses 2: v0 and the counter v4)",
    ),
    "const_max": (
        2,
        "constants for `=`, `+=` and `-=` are drawn from 0..const_max ('These programs "
        "include small integers (0, 1, and 2)' -- Nye et al. Section 5.1)",
    ),
    "mul_max": (
        2,
        "multipliers for `*=` are drawn from 2..mul_max (the published example only ever "
        "uses `v0 *= 2`); raise for a faster-growing state",
    ),
    "max_trips": (4, "maximum trip count of a while loop (its counter is initialised to 1..max_trips)"),
    "max_body": (3, "maximum number of statements inside a while-loop body, including the counter decrement"),
    "p_loop": (0.5, "probability of opening a while loop at each block position, when one still fits"),
    "max_loops": (6, "maximum number of while loops in one program"),
    "p_target_v0": (
        0.5,
        "probability that a statement targets v0, the returned variable; keeps most of the "
        "trace on the critical path to the answer instead of computing dead values",
    ),
    "p_var_operand": (
        0.35,
        "probability that the right-hand side of a `+=`/`-=` is another variable rather than "
        "a constant (e.g. `v0 += v4`); 0.0 restricts the program to constant operands only",
    ),
    "value_cap": (
        1000,
        "generation-time gate: any statement that would push |value| of some variable above "
        "this is re-drawn, so state stays printable as depth grows (see README caveats)",
    ),
    "allow_if": (
        False,
        "also emit `if <var> <cmp> <const>:` blocks (Nye et al. Section 5.1 mentions if "
        "statements; off by default because the assigned operation vocabulary is "
        "assign/+/-/small */while)",
    ),
    "p_if": (0.3, "probability of opening an if block at each block position, when allow_if is on"),
    "arg_max": (9, "the call argument in `output = f(K)` is drawn from 0..arg_max (published example: 6)"),
}

_INDENT = "  "
_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------------------
# program representation
#
# A program is (arg, body) where body is a list of nodes at indent level 1:
#   {"kind": "assign", "var": v, "val": c,                    "src": <indented source line>}
#   {"kind": "aug",    "var": v, "op": "+"|"-"|"*", "operand": "<const or var name>", "src": ...}
#   {"kind": "while",  "cvar": v, "cop": ">", "cval": 0, "body": [...], "src": ...}
#   {"kind": "if",     "cvar": v, "cop": ">"|"=="|"<", "cval": c, "body": [...], "src": ...}
# --------------------------------------------------------------------------------------

class _CapExceeded(Exception):
    pass


def _ind(level):
    return _INDENT * level


def _stmt_src(kind, var, op=None, operand=None, val=None, level=1):
    if kind == "assign":
        return "%s%s = %d" % (_ind(level), var, val)
    return "%s%s %s= %s" % (_ind(level), var, op, operand)


def _apply(node, env, cap):
    """Execute one simple statement against env (an ordered dict of live variables)."""
    if node["kind"] == "assign":
        env[node["var"]] = node["val"]
    else:
        operand = node["operand"]
        rhs = env[operand] if operand in env else int(operand)
        cur = env[node["var"]]
        if node["op"] == "+":
            cur = cur + rhs
        elif node["op"] == "-":
            cur = cur - rhs
        else:
            cur = cur * rhs
        env[node["var"]] = cur
    if cap is not None and abs(env[node["var"]]) > cap:
        raise _CapExceeded()


def _cmp(env, node):
    left, op, right = env[node["cvar"]], node["cop"], node["cval"]
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    return left == right


def _exec_nodes(nodes, env, cap, emit):
    """Walk the structured program in execution order; emit(node, env) once per executed line."""
    for node in nodes:
        kind = node["kind"]
        if kind == "while":
            while True:
                emit(node, env)
                if not _cmp(env, node):
                    break
                _exec_nodes(node["body"], env, cap, emit)
        elif kind == "if":
            emit(node, env)
            if _cmp(env, node):
                _exec_nodes(node["body"], env, cap, emit)
        else:
            _apply(node, env, cap)
            emit(node, env)


def _cost_and_env(nodes, env, cap):
    """(number of executed lines, resulting env) for a candidate block, or (None, None) if
    it would push a variable past the value cap."""
    probe = dict(env)
    count = [0]

    def _tick(_node, _env):
        count[0] += 1

    try:
        _exec_nodes(nodes, probe, cap, _tick)
    except _CapExceeded:
        return None, None
    return count[0], probe


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------

def _rng(depth, seed, knobs, attempt=0):
    key = "synthetic_program_trace|%d|%d|%d|%s" % (
        depth, seed, attempt, json.dumps(sorted(knobs.items()), default=str),
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _simple(rng, env, var, k, level, allow_mul=True):
    """Draw one augmented assignment on `var` (already live), legal in the current env.

    Plain `var = const` is deliberately NOT offered here: re-binding a live variable would
    discard everything the trace had accumulated in it, letting a model skip to the last
    write instead of carrying state.  Plain assignment is reserved for introducing a new
    variable and for loop-counter initialisation (both of which the published example does).
    """
    choices, weights = ["+", "-"], [0.42, 0.38]
    if allow_mul and k["mul_max"] >= 2:
        choices.append("*")
        weights.append(0.20)
    op = rng.choices(choices, weights=weights, k=1)[0]
    if op == "*":
        operand = str(rng.randint(2, max(2, k["mul_max"])))
    else:
        others = [n for n in env if n != var]
        if others and rng.random() < k["p_var_operand"]:
            operand = rng.choice(others)
        else:
            operand = str(rng.randint(0, k["const_max"]))
    return {"kind": "aug", "var": var, "op": op, "operand": operand,
            "src": _stmt_src("aug", var, op=op, operand=operand, level=level)}


def _fresh_assign(rng, var, k, level):
    val = rng.randint(0, k["const_max"])
    return {"kind": "assign", "var": var, "val": val,
            "src": _stmt_src("assign", var, val=val, level=level)}


def _draw_simple(rng, env, names, k, level, exclude_target=None, allow_mul=True):
    targets = [n for n in names if n != exclude_target]
    if "v0" in targets and rng.random() < k["p_target_v0"]:
        var = "v0"                       # keep the returned variable on the critical path
    else:
        var = rng.choice(targets)
    if var not in env:
        return _fresh_assign(rng, var, k, level)
    return _simple(rng, env, var, k, level, allow_mul=allow_mul)


def _build_loop(rng, env, names, k, trips, body_len, allow_mul=True):
    """`vC = T` + `while vC > 0:` with a body whose first statement is `vC -= 1`."""
    counters = [n for n in names if n != "v0"]
    unused = [n for n in counters if n not in env]
    counter = rng.choice(unused if unused else counters)
    init = {"kind": "assign", "var": counter, "val": trips,
            "src": _stmt_src("assign", counter, val=trips, level=1)}
    probe = dict(env)
    probe[counter] = trips
    body = [{"kind": "aug", "var": counter, "op": "-", "operand": "1",
             "src": _stmt_src("aug", counter, op="-", operand="1", level=2)}]
    probe[counter] = trips - 1
    for _ in range(body_len - 1):
        st = _draw_simple(rng, probe, names, k, 2, exclude_target=counter, allow_mul=allow_mul)
        body.append(st)
        _apply(st, probe, None)
    header = {"kind": "while", "cvar": counter, "cop": ">", "cval": 0, "body": body,
              "src": "%swhile %s > 0:" % (_ind(1), counter)}
    return [init, header]


def _build_if(rng, env, names, k, body_len, allow_mul=True):
    cvar = rng.choice(sorted(env)) if env else "v0"
    cop = rng.choice([">", "==", "<"])
    cval = rng.randint(0, k["const_max"])
    probe = dict(env)
    body = []
    for _ in range(body_len):
        st = _draw_simple(rng, probe, names, k, 2, allow_mul=allow_mul)
        body.append(st)
        _apply(st, probe, None)
    return [{"kind": "if", "cvar": cvar, "cop": cop, "cval": cval, "body": body,
             "src": "%sif %s %s %d:" % (_ind(1), cvar, cop, cval)}]


def _gen_body(rng, k, names, env, budget):
    """Build a function body whose execution costs exactly `budget` executed statements."""
    nodes, remaining, n_loops = [], budget, 0

    def _commit(cand, cost, new_env):
        nonlocal remaining
        nodes.extend(cand)
        env.clear()
        env.update(new_env)
        remaining -= cost

    while remaining > 0:
        placed = False

        # while loop: cost = 1 (init) + (T + 1) (header checks) + T * B (body statements).
        # B >= 2 so the body always does work besides decrementing its own counter.
        if remaining >= 5 and n_loops < k["max_loops"] and rng.random() < k["p_loop"]:
            feasible = [
                (t, b)
                for t in range(1, k["max_trips"] + 1)
                for b in range(2, max(2, k["max_body"]) + 1)
                if t * (b + 1) + 2 <= remaining
            ]
            if feasible:
                trips, body_len = rng.choice(feasible)
                for attempt in range(6):
                    cand = _build_loop(rng, env, names, k, trips, body_len, allow_mul=attempt < 3)
                    cost, new_env = _cost_and_env(cand, env, k["value_cap"])
                    if cost is not None and cost <= remaining:
                        _commit(cand, cost, new_env)
                        n_loops += 1
                        placed = True
                        break

        # if block: cost = 1 (header) + B if the condition holds, else 1
        if not placed and k["allow_if"] and remaining >= 2 and rng.random() < k["p_if"]:
            for attempt in range(6):
                body_len = rng.randint(1, min(k["max_body"], remaining - 1))
                cand = _build_if(rng, env, names, k, body_len, allow_mul=attempt < 3)
                cost, new_env = _cost_and_env(cand, env, k["value_cap"])
                if cost is not None and cost <= remaining:
                    _commit(cand, cost, new_env)
                    placed = True
                    break

        if not placed:
            for attempt in range(6):
                cand = [_draw_simple(rng, env, names, k, 1, allow_mul=attempt < 3)]
                cost, new_env = _cost_and_env(cand, env, k["value_cap"])
                if cost is not None and cost <= remaining:
                    _commit(cand, cost, new_env)
                    placed = True
                    break

        if not placed:  # always-safe fallback: introduce a variable, else `vX += 0`
            unused = [n for n in names if n not in env]
            if unused:
                cand = [_fresh_assign(rng, rng.choice(unused), k, 1)]
            else:
                var = "v0" if rng.random() < k["p_target_v0"] else rng.choice(names)
                cand = [{"kind": "aug", "var": var, "op": "+", "operand": "0",
                         "src": _stmt_src("aug", var, op="+", operand="0", level=1)}]
            cost, new_env = _cost_and_env(cand, env, None)
            _commit(cand, cost, new_env)

    return nodes


def _render_source(arg, body):
    lines = ["def f(v0):"]
    lines.extend(_flatten_src(body))
    lines.append("%sreturn v0" % _ind(1))
    lines.append("")
    lines.append("output = f(%d)" % arg)
    return "\n".join(lines)


def _flatten_src(nodes):
    out = []
    for node in nodes:
        out.append(node["src"])
        if node["kind"] in ("while", "if"):
            out.extend(_flatten_src(node["body"]))
    return out


def _state(env):
    return json.dumps(env)


def _trace(arg, body):
    """The published line:/state: trace. Returns (steps, states, final output value)."""
    steps, states = [], []

    def _emit(src, env):
        rendered = _state(env)
        steps.append("line: %s\nstate: %s" % (src, rendered))
        states.append(rendered)

    glob = {}
    glob["f"] = "<callable_object f>"
    _emit("def f(v0):", glob)

    local = {"v0": arg}
    _emit("output = f(%d)" % arg, local)

    _exec_nodes(body, local, None, lambda node, env: _emit(node["src"], env))

    glob["output"] = local["v0"]
    _emit("%sreturn v0" % _ind(1), glob)
    return steps, states, local["v0"]


PROMPT_TEMPLATE = (
    "Consider the following Python function:\n"
    "\n"
    "%s\n"
    "\n"
    "What is the value of output?"
)


def _make_instance(arg, body, depth, meta):
    steps, states, out = _trace(arg, body)
    source = _render_source(arg, body)
    return Instance(
        prompt=PROMPT_TEMPLATE % source,
        steps=steps,
        states=states,
        answer=str(out),
        depth=depth,
        meta=dict(meta, source=source, arg=arg, n_steps=len(steps)),
    )


def _resolve_knobs(knobs):
    k = {name: default for name, (default, _desc) in KNOBS.items()}
    unknown = set(knobs) - set(k)
    if unknown:
        raise ValueError("unknown knob(s): %s" % ", ".join(sorted(unknown)))
    k.update(knobs)
    if k["n_vars"] < 2:
        raise ValueError("n_vars must be >= 2 (v0 plus at least one loop counter)")
    if k["n_vars"] > 10:
        raise ValueError("n_vars must be <= 10 (variable pool is v0..v9)")
    return k


def generate(depth: int, seed: int, **knobs) -> Instance:
    """A tiny integer program whose execution trace is exactly `depth` statements long."""
    k = _resolve_knobs(knobs)
    if depth < 4:
        raise ValueError(
            "depth must be >= 4: `def f(v0):`, `output = f(K)` and `return v0` are 3 "
            "executed statements of fixed overhead, plus at least one body statement"
        )
    budget = depth - 3
    want = max(1, int(round(0.3 * budget)))
    best = None
    for attempt in range(8):
        rng = _rng(depth, seed, k, attempt)
        pool = sorted(rng.sample(range(1, 10), k["n_vars"] - 1))
        names = ["v0"] + ["v%d" % i for i in pool]
        arg = rng.randint(0, k["arg_max"])
        env = {"v0": arg}
        body = _gen_body(rng, k, names, env, budget)
        hits = _v0_writes(body, {"v0": arg})
        if best is None or hits > best[0]:
            best = (hits, arg, body, names, attempt)
        if hits >= want:
            break
    hits, arg, body, names, attempt = best
    return _make_instance(arg, body, depth, {
        "seed": seed, "knobs": k, "names": names, "attempt": attempt, "v0_writes": hits,
    })


def _v0_writes(body, env):
    """How many executed statements write to v0 (the returned variable)?"""
    count = [0]

    def _tick(node, _env):
        if node["kind"] in ("assign", "aug") and node["var"] == "v0":
            count[0] += 1

    _exec_nodes(body, dict(env), None, _tick)
    return count[0]


# --------------------------------------------------------------------------------------
# reference solver -- an independent AST interpreter, written without reference to the
# generator's own simulation: it reads the rendered program text back out of the prompt.
# --------------------------------------------------------------------------------------

class _Return(Exception):
    def __init__(self, value):
        super().__init__(value)
        self.value = value


def _extract_source(prompt):
    lines = prompt.split("\n")
    start = next(i for i, ln in enumerate(lines) if ln.startswith("def f("))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("output = f("))
    return "\n".join(lines[start:end + 1])


def _ev(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_ev(node.operand, env)
    if isinstance(node, ast.BinOp):
        left, right = _ev(node.left, env), _ev(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        raise ValueError("unsupported binary operator %r" % node.op)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _ev(node.left, env), _ev(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.Eq):
            return left == right
        raise ValueError("unsupported comparison %r" % op)
    raise ValueError("unsupported expression node %r" % node)


_AUG = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b}


def _run(stmts, env, fuel):
    for stmt in stmts:
        fuel[0] -= 1
        if fuel[0] < 0:
            raise RuntimeError("program did not terminate within the interpreter's fuel budget")
        if isinstance(stmt, ast.Assign):
            env[stmt.targets[0].id] = _ev(stmt.value, env)
        elif isinstance(stmt, ast.AugAssign):
            fn = _AUG[type(stmt.op)]
            env[stmt.target.id] = fn(env[stmt.target.id], _ev(stmt.value, env))
        elif isinstance(stmt, ast.While):
            while _ev(stmt.test, env):
                fuel[0] -= 1
                if fuel[0] < 0:
                    raise RuntimeError("program did not terminate within the interpreter's fuel budget")
                _run(stmt.body, env, fuel)
        elif isinstance(stmt, ast.If):
            _run(stmt.body if _ev(stmt.test, env) else stmt.orelse, env, fuel)
        elif isinstance(stmt, ast.Return):
            raise _Return(_ev(stmt.value, env))
        elif isinstance(stmt, ast.Pass):
            pass
        else:
            raise ValueError("unsupported statement node %r" % stmt)


def solve(inst: Instance) -> str:
    """Independently re-execute the program printed in the prompt and return `output`."""
    tree = ast.parse(_extract_source(inst.prompt))
    func, arg = None, None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func = node
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            arg = _ev(node.value.args[0], {})
    if func is None or arg is None:
        raise ValueError("prompt does not contain `def f(...)` plus `output = f(K)`")
    env = {func.args.args[0].arg: arg}
    try:
        _run(func.body, env, [2 * 10 ** 6])
    except _Return as ret:
        return str(ret.value)
    raise ValueError("function fell off the end without returning")


# --------------------------------------------------------------------------------------
# rendering and checking
# --------------------------------------------------------------------------------------

_PREFIX = "[BEGIN]\n\nstate: {}\n"
_SUFFIX = "\n\n[DONE]"


def format_cot(inst: Instance) -> str:
    return _PREFIX + "\n".join(inst.steps) + _SUFFIX + "\nAnswer: " + inst.answer


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    """Char offsets (start, end) of each step inside format_cot(inst)."""
    spans, pos = [], len(_PREFIX)
    for step in inst.steps:
        spans.append((pos, pos + len(step)))
        pos += len(step) + 1  # the joining newline
    return spans


_ANSWER_LINE = re.compile(r"^\s*(?:[*_`#>\-\s]*)answer\s*[:=]", re.IGNORECASE)
_INT = re.compile(r"-?\d+")


def _normalize(text):
    text = text.strip().strip("*_`\"' \t")
    text = text.rstrip(".").strip()
    text = text.replace(",", "")
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))
    found = _INT.findall(text)
    return str(int(found[-1])) if found else None


def check(inst: Instance, completion: str) -> bool:
    lines = completion.split("\n")
    payload = None
    for line in reversed(lines):
        if _ANSWER_LINE.match(line):
            payload = line.split(":", 1)[-1] if ":" in line else line.split("=", 1)[-1]
            break
    if payload is None:
        nonempty = [ln for ln in lines if ln.strip()]
        if not nonempty:
            return False
        payload = nonempty[-1]
    got = _normalize(payload)
    return got is not None and got == inst.answer


# --------------------------------------------------------------------------------------
# prompt redaction (SPEC AMENDMENT 4)
# --------------------------------------------------------------------------------------

REDACTION_MEANINGFUL = True

_REDACTED = "[…]"
_CALL_LINE = re.compile(r"^output = f\(-?\d+\)[ \t]*$", re.MULTILINE)


def redact_prompt(inst: Instance, k: int) -> str:
    """The prompt with everything that would let a model *recompute* the state after step k
    removed, and everything needed to *continue* from step k+1 left intact.

    What is removed: the call argument `K` in `output = f(K)`, replaced by one `[...]`
    placeholder (one removed item, one placeholder).

    What is kept, deliberately: the program text.  AMENDMENT 4 -- "where the 'initial state' is
    a static object needed for every step (a rule table, a lookup table, the program text, the
    machine's production rules, the full expression), keep it -- it is not state".  The program
    here is the rule table, not the state: it is the same text at every step, its statements are
    not consumed one per step (loops unroll, so a single source line drives many steps, and the
    number of steps a `while` contributes is itself a function of the runtime state), and steps
    k+1..depth are unexecutable without it.  The state a model could recompute is the trace so
    far, i.e. the program applied to K -- so K is the whole of the recomputable input, and
    removing it is what stops re-execution from scratch.  A model holding the trace prefix can
    still continue, because the last `state:` line of that prefix carries the full live variable
    dict; a model that was reading the initial state back out of the prompt cannot.

    Consequently the redaction is binary in k: `k == 0` returns `inst.prompt` unchanged and every
    `k >= 1` returns the same argument-redacted prompt (this includes `k == inst.depth`, where
    the initial state is gone and only the static program text and the question remain).  There
    is no finer gradation to offer, since no prefix of the source text corresponds to steps
    1..k.  Note that step 2 of the published trace is `line: output = f(K)`, so the argument is
    restated by the trace prefix itself for every k >= 2 -- that is expected (AMENDMENT 4: "the
    redaction is of the PROMPT only"); k == 1 is the one degenerate point, where the prefix has
    not reached the call line yet and the argument is genuinely unavailable.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return inst.prompt
    redacted, n = _CALL_LINE.subn("output = f(%s)" % _REDACTED, inst.prompt, count=1)
    if n != 1:
        raise ValueError("prompt has no `output = f(K)` call line to redact")
    return redacted


# --------------------------------------------------------------------------------------
# step corruption (SPEC AMENDMENT 6)
# --------------------------------------------------------------------------------------

_STATE_MARKER = "\nstate: "

# `  v0 += v4`, `  v4 = 1`, `  v0 *= 2` -- the statement's target variable.  Condition lines
# (`while ...:`, `if ...:`) and `def f(v0):` write nothing; `output = f(K)` binds the parameter,
# so the state it reports is the callee's `v0`; `return v0` publishes the global `output`.
_WRITE_RE = re.compile(r"^\s*(v\d+)\s*[-+*]?=\s*\S")


def _written_var(line_src):
    """The variable whose value this source line reports in its `state:` dict, or None."""
    stripped = line_src.strip()
    if stripped.startswith("def "):
        return None
    if stripped.startswith(("while ", "if ")):
        return None
    if stripped.startswith("return "):
        return "output"
    if stripped.startswith("output = f("):
        return "v0"
    m = _WRITE_RE.match(line_src)
    return m.group(1) if m else None


def _corrupt_rng(inst, k, seed):
    key = "synthetic_program_trace|corrupt|%d|%d|%s|%s" % (
        seed, k, inst.meta.get("source", inst.prompt), inst.states[k - 1],
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """Step `k` (1-based) rewritten to report a plausible WRONG state, plus that wrong state as a
    canonical string in the same form as `inst.states`.

    A step here is two lines -- `line: <source>` and `state: {...}` -- so, per AMENDMENT 6 ("rewrite
    exactly the lines that report the state"), the `line:` half is returned byte-identical and only
    the `state:` half changes.  The edit is minimal: exactly one variable's value moves by +/-1, so
    the wrong state has the same keys, in the same published order, with the same shapes; it is
    therefore never equal to `inst.states[k - 1]`, and it is a state the program could plausibly
    have been in.  The variable chosen is the one the step just wrote (the assignment's target; the
    callee's `v0` for `output = f(K)`; `output` for `return v0`) whenever that variable is an
    integer in the reported state, which is the corruption that actually propagates; on a condition
    line (`while`/`if`), which writes nothing, one live integer variable is picked deterministically
    instead.

    Degenerate case: step 1 is `line: def f(v0):` with `state: {"f": "<callable_object f>"}`, which
    holds no integer to shift.  It is corrupted into the premature-binding error a model plausibly
    makes -- the parameter already bound, `{"f": "<callable_object f>", "v0": K}` -- which is the
    same shape as the final step's state (a callable plus an integer), so it is still a legal state.

    Deterministic in `(inst, k, seed)` and independent of the object identity of `inst`.
    """
    if not 1 <= k <= inst.depth:
        raise ValueError("k must be in 1..%d (got %d)" % (inst.depth, k))

    step = inst.steps[k - 1]
    head, marker, _tail = step.partition(_STATE_MARKER)
    if not marker:
        raise ValueError("step %d has no `state:` line to corrupt" % k)
    line_src = head[len("line: "):] if head.startswith("line: ") else head

    env = json.loads(inst.states[k - 1])
    ints = [name for name, val in env.items() if isinstance(val, int) and not isinstance(val, bool)]
    rng = _corrupt_rng(inst, k, seed)

    if ints:
        written = _written_var(line_src)
        target = written if written in ints else rng.choice(ints)
        bad = dict(env)
        bad[target] = env[target] + rng.choice((1, -1))
    else:
        # `def f(v0):` -- nothing numeric yet; bind the parameter one step too early.
        arg = inst.meta.get("arg")
        if arg is None:
            m = re.search(r"output = f\((-?\d+)\)", inst.prompt)
            arg = int(m.group(1)) if m else 0
        bad = dict(env)
        bad["v0"] = arg

    corrupted = _state(bad)
    if corrupted == inst.states[k - 1]:  # unreachable; a +/-1 shift always moves the state
        raise AssertionError("corruption did not change the state at k=%d" % k)
    return head + _STATE_MARKER + corrupted, corrupted


# --------------------------------------------------------------------------------------
# exemplars -- [0] is the verbatim published one (Nye et al. 2021, Appendix C, pp. 13-14)
# --------------------------------------------------------------------------------------

def _published_instance():
    body = [
        {"kind": "aug", "var": "v0", "op": "+", "operand": "0",
         "src": _stmt_src("aug", "v0", op="+", operand="0", level=1)},
        {"kind": "assign", "var": "v4", "val": 2,
         "src": _stmt_src("assign", "v4", val=2, level=1)},
        {"kind": "while", "cvar": "v4", "cop": ">", "cval": 0,
         "src": "%swhile v4 > 0:" % _ind(1),
         "body": [
             {"kind": "aug", "var": "v4", "op": "-", "operand": "1",
              "src": _stmt_src("aug", "v4", op="-", operand="1", level=2)},
             {"kind": "aug", "var": "v0", "op": "*", "operand": "2",
              "src": _stmt_src("aug", "v0", op="*", operand="2", level=2)},
         ]},
    ]
    return _make_instance(6, body, 12, {"published": True, "source_ref":
                                        "Nye et al. 2021, arXiv:2112.00114, Appendix C, first example"})


def exemplars(k: int, seed: int) -> list[Instance]:
    if k <= 0:
        return []
    out = [_published_instance()]
    i = 0
    while len(out) < k:
        out.append(generate(10, (seed * 1009 + i) % (2 ** 31)))
        i += 1
    return out[:k]


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------

def _published_trace_body():
    path = os.path.join(_HERE, "published_trace.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    marker = text.index("=== Model/gold trace")
    idx = text.index("[BEGIN]", text.index("\n", marker))
    return text[idx:].strip()


def _selftest():
    rng = random.Random(20260916)
    failures = []

    # 1. the published exemplar is reproduced character-for-character
    pub = _published_instance()
    body = _published_trace_body()
    if body is None:
        print("  ! published_trace.txt missing; skipped the verbatim-fidelity check")
    else:
        mine = format_cot(pub)
        mine_trace = mine[: mine.rindex("\nAnswer: ")]
        if mine_trace != body:
            failures.append("published trace mismatch:\n--- mine ---\n%s\n--- published ---\n%s"
                            % (mine_trace, body))
        if pub.meta["source"] not in open(os.path.join(_HERE, "published_trace.txt"),
                                          encoding="utf-8").read():
            failures.append("published program source is not a substring of published_trace.txt")
    if pub.answer != "24" or pub.depth != 12 or len(pub.steps) != 12:
        failures.append("published exemplar is not 12 steps answering 24: %r" % pub.answer)
    if solve(pub) != pub.answer:
        failures.append("solve() disagrees with the published exemplar")

    # 2. 200 random (depth, seed) pairs
    cases = []
    for _ in range(200):
        depth = rng.choice([4, 5, 6, 8, 10, 12, 16, 20, 26, 32, 40, 48, 60, 72, 96])
        seed = rng.randrange(10 ** 6)
        knobs = {}
        if rng.random() < 0.35:
            knobs["n_vars"] = rng.choice([2, 3, 4, 5])
        if rng.random() < 0.25:
            knobs["allow_if"] = True
        if rng.random() < 0.2:
            knobs["mul_max"] = rng.choice([2, 3])
        if rng.random() < 0.2:
            knobs["p_var_operand"] = rng.choice([0.0, 0.6])
        if rng.random() < 0.15:
            knobs["max_trips"] = rng.choice([1, 2, 6])
        cases.append((depth, seed, knobs))

    for depth, seed, knobs in cases:
        inst = generate(depth, seed, **knobs)
        tag = "depth=%d seed=%d knobs=%s" % (depth, seed, knobs)
        if solve(inst) != inst.answer:
            failures.append("solve != answer (%s): %s vs %s" % (tag, solve(inst), inst.answer))
        if not (len(inst.steps) == len(inst.states) == inst.depth == depth):
            failures.append("length mismatch (%s): %d steps, %d states, depth %d"
                            % (tag, len(inst.steps), len(inst.states), inst.depth))
        if not check(inst, format_cot(inst)):
            failures.append("check(format_cot) false (%s)" % tag)
        wrong = str(int(inst.answer) + 1)
        if check(inst, "Answer: %s" % wrong):
            failures.append("check accepted a wrong answer (%s)" % tag)
        again = generate(depth, seed, **knobs)
        if (again.prompt, again.steps, again.answer) != (inst.prompt, inst.steps, inst.answer):
            failures.append("generate is not deterministic (%s)" % tag)
        # the final state line must carry the answer
        if ('"output": %s' % inst.answer) not in inst.states[-1]:
            failures.append("final state does not report output (%s): %s" % (tag, inst.states[-1]))
        # step spans must index format_cot correctly
        cot = format_cot(inst)
        for (lo, hi), step in zip(step_spans(inst), inst.steps):
            if cot[lo:hi] != step:
                failures.append("step_spans misaligned (%s)" % tag)
                break
        # every step is a line:/state: pair in the published shape
        for step in inst.steps:
            head, _, tail = step.partition("\n")
            if not head.startswith("line: ") or not tail.startswith("state: {"):
                failures.append("step is not a published-format line:/state: pair (%s): %r" % (tag, step))
                break

    # 3. tolerant parsing and exemplar rendering
    probe = generate(12, 1)
    for variant in ["Answer: %s" % probe.answer,
                    "blah blah\n**Answer:** %s.**" % probe.answer,
                    "%s" % probe.answer,
                    "The value of output is %s\n" % probe.answer]:
        if not check(probe, variant):
            failures.append("tolerant parse failed on %r" % variant)
    if check(probe, "Answer: I don't know"):
        failures.append("check accepted a non-numeric answer")

    ex = exemplars(3, 0)
    if len(ex) != 3 or not ex[0].meta.get("published"):
        failures.append("exemplars(3, 0) did not render with the published exemplar first")
    for e in ex:
        if solve(e) != e.answer or not check(e, format_cot(e)):
            failures.append("exemplar is inconsistent")

    # 4. prompt redaction (AMENDMENT 4)
    if REDACTION_MEANINGFUL is not True:
        failures.append("REDACTION_MEANINGFUL must be True for this task")
    for _ in range(20):
        depth = rng.choice([4, 6, 8, 12, 20, 32, 48, 72])
        seed = rng.randrange(10 ** 6)
        inst = generate(depth, seed)
        tag = "redaction depth=%d seed=%d" % (depth, seed)
        call = "output = f(%d)" % inst.meta["arg"]
        blank = "output = f(%s)" % _REDACTED
        if redact_prompt(inst, 0) != inst.prompt:
            failures.append("redact_prompt(inst, 0) != inst.prompt (%s)" % tag)
        for k in sorted({1, depth // 2, depth}):
            red = redact_prompt(inst, k)
            if red == inst.prompt:
                failures.append("redact_prompt k=%d left the prompt unchanged (%s)" % (k, tag))
            if red.count(_REDACTED) != 1:
                failures.append("redact_prompt k=%d: %d placeholders, want 1 (%s)"
                                % (k, red.count(_REDACTED), tag))
            if call in red or blank not in red:
                failures.append("redact_prompt k=%d did not blank the call argument (%s)" % (k, tag))
            if red != inst.prompt.replace(call, blank):
                failures.append("redact_prompt k=%d touched more than the argument (%s)" % (k, tag))
        # k == depth: no initial-state token survives; the static program text and the question do
        full = redact_prompt(inst, depth)
        for line in inst.meta["source"].split("\n"):
            if line.startswith("output = f(") or not line.strip():
                continue
            if line not in full:
                failures.append("redact_prompt(depth) dropped a program line (%s): %r" % (tag, line))
                break
        if "What is the value of output?" not in full:
            failures.append("redact_prompt(depth) dropped the question (%s)" % tag)
        # what a model must continue from instead: the last state line of the trace prefix
        if inst.states[depth // 2] not in format_cot(inst):
            failures.append("state line missing from the rendered trace (%s)" % tag)

    # 5. step corruption (AMENDMENT 6)
    for _ in range(20):
        depth = rng.choice([4, 6, 8, 12, 20, 32, 48, 72])
        seed = rng.randrange(10 ** 6)
        inst = generate(depth, seed)
        cseed = rng.randrange(10 ** 6)
        for k in sorted({1, depth // 2, depth}):
            tag = "corruption depth=%d seed=%d k=%d" % (depth, seed, k)
            text, bad = corrupt_step(inst, k, cseed)
            true_state = inst.states[k - 1]
            if bad == true_state:
                failures.append("%s: corrupted_state equals the true state %r" % (tag, bad))
            if text == inst.steps[k - 1]:
                failures.append("%s: corrupted step text is unchanged" % tag)
            # structure: still a `line: ...` / `state: {...}` pair, with the action half untouched
            head, _, tail = text.partition("\n")
            true_head, _, _true_tail = inst.steps[k - 1].partition("\n")
            if head != true_head:
                failures.append("%s: corruption altered the `line:` half: %r" % (tag, head))
            if not tail.startswith("state: {") or not tail.endswith("}") or "\n" in tail:
                failures.append("%s: corrupted step is not a published-format pair: %r" % (tag, text))
                continue
            shown = tail[len("state: "):]
            if shown != bad:
                failures.append("%s: reported state %r != corrupted_state %r" % (tag, shown, bad))
            try:
                parsed = json.loads(bad)
            except ValueError:
                failures.append("%s: corrupted_state is not JSON: %r" % (tag, bad))
                continue
            if _state(parsed) != bad:
                failures.append("%s: corrupted_state is not canonical: %r" % (tag, bad))
            truth = json.loads(true_state)
            # legal shape: variable names of this program (or `f`/`output`), values ints or the
            # published callable placeholder
            legal_names = set(inst.meta["names"]) | {"f", "output"}
            for name, val in parsed.items():
                if name not in legal_names:
                    failures.append("%s: corrupted_state invents a variable %r" % (tag, name))
                if not isinstance(val, int) and val != "<callable_object f>":
                    failures.append("%s: corrupted_state has an illegal value %r" % (tag, val))
            ints = [n for n, v in truth.items() if isinstance(v, int)]
            if ints:
                # minimal edit: same keys, same order, exactly one value off by exactly one
                if list(parsed) != list(truth):
                    failures.append("%s: corruption changed the state's keys: %r vs %r"
                                    % (tag, bad, true_state))
                    continue
                diffs = [n for n in truth if parsed[n] != truth[n]]
                if len(diffs) != 1 or abs(parsed[diffs[0]] - truth[diffs[0]]) != 1:
                    failures.append("%s: not a single +/-1 edit: %r vs %r" % (tag, bad, true_state))
                # preference: the variable the step just wrote, when it is live and numeric
                written = _written_var(true_head[len("line: "):])
                if written in ints and diffs != [written]:
                    failures.append("%s: corrupted %r, not the variable the step wrote (%r)"
                                    % (tag, diffs[0], written))
            else:
                # the `def f(v0):` prologue: premature parameter binding, one extra integer
                if k != 1 or list(parsed) != ["f", "v0"] or parsed["v0"] != inst.meta["arg"]:
                    failures.append("%s: unexpected corruption of a non-numeric state: %r" % (tag, bad))
            # determinism, and purity in (inst, k, seed)
            if corrupt_step(inst, k, cseed) != (text, bad):
                failures.append("%s: corrupt_step is not deterministic" % tag)
            if corrupt_step(generate(depth, seed), k, cseed) != (text, bad):
                failures.append("%s: corrupt_step is not a pure function of (inst, k, seed)" % tag)
            # the corrupted step must splice into the trace where the real one sat
            if format_cot(inst).count(inst.steps[k - 1]) < 1:
                failures.append("%s: the true step is not locatable in the rendered trace" % tag)
    for bad_k in (0, -1, 10 ** 6):
        try:
            corrupt_step(_published_instance(), bad_k, 0)
            failures.append("corrupt_step accepted k=%r" % bad_k)
        except ValueError:
            pass

    if failures:
        print("FAIL (%d)" % len(failures))
        for f in failures[:10]:
            print("  -", f)
        return 1
    print("ok: published trace reproduced verbatim (12 steps, answer 24)")
    print("ok: 200 random instances -- solve()==answer, check(format_cot)==True, "
          "check(wrong)==False, deterministic, len(steps)==len(states)==depth")
    print("ok: tolerant answer parsing, step_spans, exemplars(3, 0)")
    print("ok: 20 instances x k in {0, 1, depth//2, depth} -- redact_prompt(0) is the prompt, "
          "k>0 blanks the call argument with exactly one […] and nothing else, "
          "program text and question survive k=depth")
    print("ok: 20 instances x k in {1, depth//2, depth} -- corrupt_step leaves the `line:` half "
          "byte-identical and shifts exactly one variable of the `state:` dict by +/-1 "
          "(the one the step wrote, where it is live), deterministically")
    print("PASS")
    return 0


def _demo():
    out = []
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for seed in range(3):
            inst = generate(depth, seed)
            out.append("=" * 88)
            out.append("depth=%d  seed=%d  answer=%s" % (inst.depth, seed, inst.answer))
            out.append("=" * 88)
            out.append("--- prompt ---")
            out.append(inst.prompt)
            out.append("")
            out.append("--- gold CoT (format_cot) ---")
            out.append(format_cot(inst))
            out.append("")
    print("\n".join(out))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if args.demo:
        sys.exit(_demo())
    ap.print_help()


if __name__ == "__main__":
    main()
