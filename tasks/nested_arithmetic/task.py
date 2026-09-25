"""
nested_arithmetic -- BIG-Bench-Hard `multistep_arithmetic_two`, in the published
3-shot chain-of-thought format of Suzgun et al. 2022 (arXiv:2210.09261).

Published format (see published_trace.txt, verbatim from
vendor/bbh-suzgun/cot-prompts/multistep_arithmetic_two.txt @ 9ee07bd, MIT):

    Q: ((-5 + 9 * -4 - 0) * (4 + -7 + 0 * -5)) =
    A: Let's think step by step.
    Let's recall that the order of operations ... compute the expressions inside parentheses first.
    This equation can be written as "A * B", where A = (-5 + 9 * -4 - 0) and B = (4 + -7 + 0 * -5).
    Let's calculate A = (-5 + 9 * -4 - 0) = (-5 + (9 * -4) - 0) = ... = -41.
    Let's calculate B = (4 + -7 + 0 * -5) = ... = -3.
    Then, the final equation is A * B = -41 * -3 = 123. So the answer is 123.

i.e. the expression is decomposed into named parenthesised sub-expressions, each
sub-expression gets one "Let's calculate <letter> = ..." line that chains `=` rewrites
(parenthesise the next operation under standard precedence, then substitute its value)
until a single integer remains, and a final "Then, the final equation is ..." line
combines the named values and closes with "So the answer is X."

The fixed 250-example BBH set is vendored (vendor/bbh-suzgun/bbh/multistep_arithmetic_two.json)
but is canary-tagged and contamination-risked, so it is NOT used as eval data; instances are
generated fresh here, reusing the expression grammar of the pre-BBH BIG-bench generator
(vendor/bigbench-google/.../multistep_arithmetic/task.py, Apache 2.0): operators {+,-,*},
single-digit operands in [-9, 9], flat chains of `chain_len` operands inside each innermost
parenthesis, nested via parenthesis groups.

Pure python stdlib. See README.md for the format decision, depth semantics and caveats.
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass, field


# --------------------------------------------------------------------------------------
# interface types
# --------------------------------------------------------------------------------------


@dataclass
class Instance:
    prompt: str          # problem statement only, published wording ("<expr> =")
    steps: list[str]     # gold trace, one element per sub-expression reduction
    states: list[str]    # value of the sub-expression computed by each step, "A=-41"
    answer: str          # the integer value of the whole expression
    depth: int           # number of sub-expression reductions == len(steps)
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a single integer, possibly negative, with no commas or units"

# the expression is the operator list, so blanking evaluated sub-expressions is meaningful
REDACTION_MEANINGFUL = True

# depth = number of parenthesised sub-expression groups = number of "Let's calculate ..."
# lines plus the final combination line. BBH's own fixed set is depth 3. See README.
DEPTHS = [2, 3, 5, 7, 10, 14]

KNOBS = {
    "chain_len": (
        4,
        "number of operands in an innermost (leaf) parenthesis group; BBH's fixed "
        "multistep_arithmetic_two set uses 4, BIG-bench samples from range(2, 5)",
    ),
    "operators": (
        "+-*",
        "operator alphabet sampled uniformly; BIG-bench default ['+','-','*']",
    ),
    "min_operand": (
        -9,
        "smallest integer literal; BIG-bench default numbers=list(range(-9, 10))",
    ),
    "max_operand": (
        9,
        "largest integer literal; BIG-bench default numbers=list(range(-9, 10))",
    ),
    "max_children": (
        3,
        "maximum number of sub-expression groups directly nested inside one group; "
        "BBH's fixed set uses 2",
    ),
    "max_abs": (
        1000000,
        "soft cap on |intermediate value|; a group whose operators blow past it is "
        "resampled, so deep instances stay about arithmetic-with-state rather than "
        "big-number multiplication",
    ),
}

_RESAMPLE_TRIES = 64

# The two fixed preamble lines of the published CoT, verbatim (note: the published text
# says "addition and multiplication (from left to right)" where it means addition and
# subtraction, and uses a U+2019 apostrophe in "Let's recall" -- both reproduced as-is).
_PREAMBLE_THINK = "Let's think step by step."
_PREAMBLE_ORDER = (
    "Let\u2019s recall that the order of operations in mathematics is as follows: "
    "(1) Parentheses, (2) exponents, (3) multiplication and division (from left to "
    "right), (4) addition and multiplication (from left to right). So, remember to "
    "always compute the expressions inside parentheses or brackets first."
)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _letter(i: int) -> str:
    """A, B, ... Z, A1, B1, ... for sub-expression names, in evaluation order."""
    if i < 26:
        return chr(ord("A") + i)
    return chr(ord("A") + i % 26) + str(i // 26)


def _apply(a: int, op: str, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    raise ValueError("unknown operator %r" % (op,))


def _join(terms: list[str], ops: list[str], wrap: bool) -> str:
    """Render a flat chain; wrap in parentheses iff `wrap` and more than one term."""
    s = terms[0]
    for op, t in zip(ops, terms[1:]):
        s += " " + op + " " + t
    return "(" + s + ")" if (wrap and ops) else s


def _reduce_chain(vals: list[int], ops: list[str], wrap: bool) -> list[str]:
    """The published '=' rewrite chain for one flat chain of numbers.

    Repeatedly: pick the next operation under precedence (leftmost '*', else leftmost
    '+'/'-'), show it parenthesised, then show it substituted by its value. The
    parenthesise-then-substitute pair is the published move; it is skipped when only one
    operation is left, where it would add nothing.
    """
    terms = [str(v) for v in vals]
    cur_vals = list(vals)
    cur_ops = list(ops)
    pieces = [_join(terms, cur_ops, wrap)]
    while cur_ops:
        i = cur_ops.index("*") if "*" in cur_ops else 0
        rest_ops = cur_ops[:i] + cur_ops[i + 1:]
        if len(cur_ops) > 1:
            hl = (
                terms[:i]
                + ["(" + terms[i] + " " + cur_ops[i] + " " + terms[i + 1] + ")"]
                + terms[i + 2:]
            )
            pieces.append(_join(hl, rest_ops, wrap))
        v = _apply(cur_vals[i], cur_ops[i], cur_vals[i + 1])
        cur_vals = cur_vals[:i] + [v] + cur_vals[i + 2:]
        terms = terms[:i] + [str(v)] + terms[i + 2:]
        cur_ops = rest_ops
        pieces.append(_join(terms, cur_ops, wrap))
    out = []
    for p in pieces:
        if not out or out[-1] != p:
            out.append(p)
    return out


# --------------------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------------------


def _knobs(knobs: dict) -> dict:
    cfg = {k: v[0] for k, v in KNOBS.items()}
    for k, v in knobs.items():
        if k not in cfg:
            raise TypeError("unknown knob %r (known: %s)" % (k, sorted(cfg)))
        cfg[k] = v
    if cfg["chain_len"] < 2:
        raise ValueError("chain_len must be >= 2")
    if cfg["min_operand"] > cfg["max_operand"]:
        raise ValueError("min_operand > max_operand")
    if not cfg["operators"] or any(o not in "+-*" for o in cfg["operators"]):
        raise ValueError("operators must be a non-empty subset of '+-*'")
    if cfg["max_children"] < 1:
        raise ValueError("max_children must be >= 1")
    return cfg


def generate(depth: int, seed: int, **knobs) -> Instance:
    """Build a fresh nested-arithmetic instance with exactly `depth` sub-expression groups.

    Pure function of (depth, seed, knobs).
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    cfg = _knobs(knobs)
    sig = "|".join("%s=%s" % (k, cfg[k]) for k in sorted(cfg))
    rng = random.Random("nested_arithmetic|%d|%d|%s" % (depth, seed, sig))

    d = depth
    root = d - 1

    # --- tree shape: every group i < root gets a parent with a higher index, so the
    # evaluation order 0, 1, ..., root is automatically a valid bottom-up order.
    children: list[list[int]] = [[] for _ in range(d)]
    for i in range(d - 2, -1, -1):
        cands = [j for j in range(i + 1, d) if len(children[j]) < cfg["max_children"]]
        children[rng.choice(cands)].append(i)
    for j in range(d):
        rng.shuffle(children[j])

    # --- contents of each group, bottom-up, with a local resample when a group's value
    # runs away (see max_abs).
    lo, hi = cfg["min_operand"], cfg["max_operand"]
    ops_alpha = list(cfg["operators"])
    value: list[int] = [0] * d
    slots: list[list] = []      # per group: list of ("lit", n) / ("grp", idx)
    ops: list[list[str]] = []

    for i in range(d):
        kids = children[i]
        c = len(kids)
        n_terms = cfg["chain_len"] if c == 0 else max(c, 2)
        best = None
        for attempt in range(_RESAMPLE_TRIES):
            lits = [rng.randint(lo, hi) for _ in range(n_terms - c)]
            my_slots = [("grp", k) for k in kids] + [("lit", n) for n in lits]
            rng.shuffle(my_slots)
            alpha = ops_alpha
            if attempt >= _RESAMPLE_TRIES // 2:
                # last resort: additive-only combination keeps magnitudes linear
                add_only = [o for o in ops_alpha if o != "*"]
                alpha = add_only or ops_alpha
            my_ops = [rng.choice(alpha) for _ in range(n_terms - 1)]
            vals = [value[s[1]] if s[0] == "grp" else s[1] for s in my_slots]
            v, peak = _eval_chain(vals, my_ops)
            if best is None or peak < best[0]:
                best = (peak, my_slots, my_ops, v)
            if peak <= cfg["max_abs"]:
                break
        peak, my_slots, my_ops, v = best
        slots.append(my_slots)
        ops.append(my_ops)
        value[i] = v

    # --- rendering
    def literal_expr(i: int) -> str:
        terms = [literal_expr(s[1]) if s[0] == "grp" else str(s[1]) for s in slots[i]]
        return _join(terms, ops[i], True)

    def letter_terms(i: int) -> list[str]:
        return [_letter(s[1]) if s[0] == "grp" else str(s[1]) for s in slots[i]]

    def value_terms(i: int) -> list[str]:
        return [str(value[s[1]]) if s[0] == "grp" else str(s[1]) for s in slots[i]]

    def term_vals(i: int) -> list[int]:
        return [value[s[1]] if s[0] == "grp" else s[1] for s in slots[i]]

    prompt = literal_expr(root) + " ="

    steps: list[str] = []
    states: list[str] = []
    for i in range(d):
        is_root = i == root
        wrap = (not is_root) or d == 1
        disp = _join(letter_terms(i), ops[i], wrap)
        pieces = []
        chain = _reduce_chain(term_vals(i), ops[i], wrap)
        if disp != chain[0]:
            pieces.append(disp)
        pieces.extend(chain)
        body = " = ".join(pieces)
        if is_root:
            steps.append(
                "Then, the final equation is %s. So the answer is %d." % (body, value[i])
            )
        else:
            steps.append("Let's calculate %s = %s." % (_letter(i), body))
        states.append("%s=%d" % (_letter(i), value[i]))

    # notation-setup preamble line (omitted when there is nothing to name)
    preamble = [_PREAMBLE_THINK, _PREAMBLE_ORDER]
    if d > 1:
        names = [
            "%s = %s" % (_letter(i), _join(letter_terms(i), ops[i], True))
            for i in range(d - 1)
        ]
        if len(names) == 1:
            where = names[0]
        elif len(names) == 2:
            where = names[0] + " and " + names[1]
        else:
            where = ", ".join(names[:-1]) + ", and " + names[-1]
        preamble.append(
            'This equation can be written as "%s", where %s.'
            % (_join(letter_terms(root), ops[root], False), where)
        )

    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=str(value[root]),
        depth=d,
        meta={
            "seed": seed,
            "preamble": preamble,
            "knobs": cfg,
            "n_groups": d,
            "children": [list(c) for c in children],
            "peak_abs": max(abs(v) for v in value),
            "source": "generated",
        },
    )


def _eval_chain(vals: list[int], chain_ops: list[str]) -> tuple[int, int]:
    """Value of a flat chain under standard precedence, plus the peak |intermediate|."""
    cur = list(vals)
    cur_ops = list(chain_ops)
    peak = max(abs(v) for v in cur)
    while cur_ops:
        i = cur_ops.index("*") if "*" in cur_ops else 0
        v = _apply(cur[i], cur_ops[i], cur[i + 1])
        peak = max(peak, abs(v))
        cur = cur[:i] + [v] + cur[i + 2:]
        cur_ops = cur_ops[:i] + cur_ops[i + 1:]
    return cur[0], peak


# --------------------------------------------------------------------------------------
# reference solver -- written against the prompt string only, no generator internals
# --------------------------------------------------------------------------------------


_TOK = re.compile(r"\s*(\d+|[-+*()])")


def _tokenize(src: str) -> list[str]:
    toks: list[str] = []
    pos = 0
    while pos < len(src):
        m = _TOK.match(src, pos)
        if not m:
            if src[pos].isspace():
                pos += 1
                continue
            raise ValueError("bad character %r at %d in %r" % (src[pos], pos, src))
        toks.append(m.group(1))
        pos = m.end()
    return toks


def solve(inst: Instance) -> str:
    """Independent recursive-descent evaluator over inst.prompt. No eval()."""
    src = inst.prompt.strip()
    if src.endswith("="):
        src = src[:-1]
    toks = _tokenize(src)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def parse_expr() -> int:
        v = parse_term()
        while peek() in ("+", "-"):
            op = take()
            r = parse_term()
            v = v + r if op == "+" else v - r
        return v

    def parse_term() -> int:
        v = parse_atom()
        while peek() == "*":
            take()
            v = v * parse_atom()
        return v

    def parse_atom() -> int:
        t = peek()
        if t == "-":
            take()
            return -parse_atom()
        if t == "+":
            take()
            return parse_atom()
        if t == "(":
            take()
            v = parse_expr()
            if peek() != ")":
                raise ValueError("expected ) at token %d in %r" % (pos, inst.prompt))
            take()
            return v
        if t is None or not t.isdigit():
            raise ValueError("expected a number at token %d in %r" % (pos, inst.prompt))
        return int(take())

    val = parse_expr()
    if pos != len(toks):
        raise ValueError("trailing tokens in %r" % (inst.prompt,))
    return str(val)


# --------------------------------------------------------------------------------------
# gold trace / answer checking
# --------------------------------------------------------------------------------------


def format_cot(inst: Instance) -> str:
    """The gold completion: published preamble + reduction lines + the uniform Answer line."""
    lines = list(inst.meta.get("preamble", [])) + list(inst.steps)
    return "\n".join(lines) + "\nAnswer: " + inst.answer


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    """Char offsets of each step within format_cot(inst)."""
    text = format_cot(inst)
    spans = []
    cursor = 0
    for s in inst.steps:
        idx = text.index(s, cursor)
        spans.append((idx, idx + len(s)))
        cursor = idx + len(s)
    return spans


_INT = re.compile(r"[-+]?\d+")


def _extract_int(text: str):
    text = text.replace(",", "")
    text = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("*", "").replace("`", "")
    ms = _INT.findall(text)
    if not ms:
        return None
    try:
        return int(ms[-1])
    except ValueError:
        return None


def check(inst: Instance, completion: str) -> bool:
    """Take the last 'Answer:' line, else the last non-empty line; compare as integers."""
    lines = [ln for ln in completion.splitlines() if ln.strip()]
    if not lines:
        return False
    payload = None
    for ln in reversed(lines):
        m = re.match(r"\s*answer\s*[:\-]\s*(.*)$", ln, re.IGNORECASE)
        if m:
            payload = m.group(1)
            break
    if payload is None:
        payload = lines[-1]
    got = _extract_int(payload)
    if got is None:
        return False
    return got == int(inst.answer)


# --------------------------------------------------------------------------------------
# prompt redaction (batch 2 / prompt blinding)
# --------------------------------------------------------------------------------------

# For this task the expression IS the operator list: step i consumes the parenthesised
# sub-expression named by letter i, so "the operators consumed by steps 1..k" are exactly
# the sub-expressions of groups 0..k-1. They are replaced in place by the placeholder,
# one per removed span (a group nested inside another removed group disappears with its
# parent, so only maximal removed groups leave a visible placeholder). Everything else --
# the outer parenthesis structure, the unevaluated groups, the trailing "=" question --
# is left byte-identical.
_PLACEHOLDER = "[…]"

_NAME_DEF = re.compile(r"([A-Z]\d*) = (?=\()")
_LETTER_TOK = re.compile(r"[A-Z]\d*")


def _letter_index(name: str) -> int:
    """Inverse of _letter(): 'A' -> 0, 'Z' -> 25, 'A1' -> 26."""
    return (ord(name[0]) - ord("A")) + 26 * int(name[1:] or 0)


def _matching_paren(s: str, start: int) -> int:
    """Index just past the ')' matching the '(' at s[start]."""
    level = 0
    for j in range(start, len(s)):
        if s[j] == "(":
            level += 1
        elif s[j] == ")":
            level -= 1
            if level == 0:
                return j + 1
    raise ValueError("unbalanced parentheses in %r" % (s,))


def _group_templates(inst: Instance) -> list[str]:
    """Per group, in evaluation order, its expression with child groups left as letters.

    Read out of the published notation-setup sentence, which names every group but the
    root (`where A = (-5 + 9 * -4 - 0) and B = (4 + -7 + 0 * -5)`) and gives the root's
    own term list inside the quotes (`"A * B"`). Every letter in template i belongs to a
    group with a smaller index, so expanding bottom-up reconstructs the prompt exactly --
    which redact_prompt() asserts before touching anything.
    """
    d = len(inst.steps) or inst.depth
    sentence = None
    for ln in inst.meta.get("preamble", []):
        if ln.startswith("This equation can be written as "):
            sentence = ln
    if sentence is None:
        if d <= 1:
            return [_prompt_expr(inst)[0]]
        raise ValueError("no notation-setup line in meta['preamble']; cannot redact")
    q0 = sentence.index('"')
    q1 = sentence.index('"', q0 + 1)
    defs = {}
    rest = sentence[q1 + 1:]
    for m in _NAME_DEF.finditer(rest):
        defs[m.group(1)] = rest[m.end():_matching_paren(rest, m.end())]
    out = []
    for i in range(d - 1):
        name = _letter(i)
        if name not in defs:
            raise ValueError("group %s not named in %r" % (name, sentence))
        out.append(defs[name])
    out.append(sentence[q0 + 1:q1])
    return out


def _prompt_expr(inst: Instance) -> tuple[str, str]:
    """Split the prompt into (expression, question suffix), e.g. ('(1 + 2)', ' =')."""
    m = re.match(r"^(.*?)(\s*=\s*)$", inst.prompt, re.S)
    if not m:
        return inst.prompt, ""
    return m.group(1), m.group(2)


def _render(i: int, k: int, tpls: list[str]) -> str:
    if i < k:
        return _PLACEHOLDER
    return _LETTER_TOK.sub(
        lambda m: _render(_letter_index(m.group(0)), k, tpls), tpls[i]
    )


def redact_prompt(inst: Instance, k: int) -> str:
    """The prompt with the sub-expressions consumed by steps 1..k blanked out.

    k == 0 returns the prompt unchanged; k == depth removes the whole expression and
    leaves only the question ("[...] ="). What survives for 0 < k < depth is enough to
    continue from step k+1 given the trace so far, and nothing more: the values of the
    removed groups exist only in the trace.
    """
    d = len(inst.steps) or inst.depth
    if k < 0 or k > d:
        raise ValueError("k must be in [0, %d], got %d" % (d, k))
    if k == 0:
        return inst.prompt
    expr, suffix = _prompt_expr(inst)
    tpls = _group_templates(inst)
    whole = _render(d - 1, 0, tpls)
    if whole == expr:
        wrap = False
    elif "(" + whole + ")" == expr:
        wrap = True          # literal_expr() parenthesises the root; the "written as"
    else:                    # sentence does not
        raise ValueError(
            "reconstructed expression %r does not match prompt %r" % (whole, expr)
        )
    body = _render(d - 1, k, tpls)
    if wrap and k < d:
        body = "(" + body + ")"
    return body + suffix


# --------------------------------------------------------------------------------------
# step corruption (batch 3 / mistake propagation)
# --------------------------------------------------------------------------------------

# The state after step k is the value of the named group computed by that step, and the
# step line reports it exactly once as its final "= <value>" (the root line reports it
# twice: once closing the "=" chain and once in "So the answer is <value>."). Corrupting
# the step therefore means substituting a plausible wrong integer at that final position
# -- and at both positions on the root line, so the line stays internally consistent.
#
# DOCUMENTED DEVIATION: the intermediate "=" rewrites earlier in the same line are left
# untouched, so a reader who re-does the arithmetic of that one line can see the last
# step disagrees with its own penultimate rewrite. Making the whole chain consistent with
# the wrong value would mean fabricating a different (wrong) operand somewhere upstream,
# which changes the step's action/operator text -- forbidden by AMENDMENT 6 (d). The
# harness's question is whether a wrong *state* propagates forward, not whether the model
# audits the line it was handed, so the minimal edit is the right one here.
#
# This task has no `format` knob (one published format only), so AMENDMENT 6 (e) is
# satisfied trivially; the rewrite is driven off the step text, not generator internals,
# so it also works on the verbatim published exemplar.

# "Let's calculate A = (...) = ... = -41."
_STEP_LEAF = re.compile(r"^(Let's calculate ([A-Z]\d*) = .* = )(-?\d+)\.$")
# "Then, the final equation is A * B = -41 * -3 = 123. So the answer is 123."
_STEP_ROOT = re.compile(
    r"^(Then, the final equation is .* = )(-?\d+)\. So the answer is (-?\d+)\.$"
)


def _wrong_values(v: int) -> tuple[list[int], list[int]]:
    """Two pools of plausible wrong values for `v`: small offsets, and one-digit edits.

    Both keep the sign and (for the digit edits) the digit count of `v`, so the result
    still looks like a value this expression could have produced.
    """
    near = [v + d for d in (-3, -2, -1, 1, 2, 3)]
    digits = str(abs(v))
    edits = set()
    for i, ch in enumerate(digits):
        for repl in "0123456789":
            if repl == ch:
                continue
            if i == 0 and repl == "0" and len(digits) > 1:
                continue            # no leading zero: keep the magnitude plausible
            m = int(digits[:i] + repl + digits[i + 1:])
            edits.add(-m if v < 0 else m)
    near = [w for w in near if w != v]
    edit_pool = sorted(w for w in edits if w != v)
    return near, edit_pool


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """Step k rewritten to report a plausible wrong value, plus that state string.

    Returns `(step_text, corrupted_state)` where `step_text` is `inst.steps[k-1]` with
    only its final reported value replaced, and `corrupted_state` is the `"A=-41"`-form
    string matching `inst.states[k-1]`. Deterministic in `(inst, k, seed)`.
    """
    d = len(inst.steps) or inst.depth
    if k < 1 or k > d:
        raise ValueError("k must be in [1, %d], got %d" % (d, k))
    step = inst.steps[k - 1]
    name, _, true_str = inst.states[k - 1].partition("=")
    v = int(true_str)

    rng = random.Random(
        "nested_arithmetic|corrupt|%s|%d|%d|%d" % (inst.prompt, k, seed, v)
    )
    near, edits = _wrong_values(v)
    pool = near if (rng.random() < 0.5 or not edits) else edits
    w = rng.choice(pool)
    if w == v:                                  # unreachable; belt and braces
        w = v + 1

    m = _STEP_ROOT.match(step)
    if m:
        if int(m.group(2)) != v or int(m.group(3)) != v:
            raise ValueError("step %d does not report state %r: %r" % (k, v, step))
        text = "%s%d. So the answer is %d." % (m.group(1), w, w)
    else:
        m = _STEP_LEAF.match(step)
        if not m:
            raise ValueError("unrecognised step line %r" % (step,))
        if m.group(2) != name or int(m.group(3)) != v:
            raise ValueError("step %d does not report state %r: %r"
                             % (k, inst.states[k - 1], step))
        text = "%s%d." % (m.group(1), w)
    return text, "%s=%d" % (name, w)


# --------------------------------------------------------------------------------------
# exemplars -- [0] is the verbatim published BBH exemplar
# --------------------------------------------------------------------------------------


_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACE_PATHS = [
    os.path.join(_HERE, "published_trace.txt"),
    os.path.join(_HERE, "vendor", "bbh-suzgun", "cot-prompts",
                 "multistep_arithmetic_two.txt"),
]


def _published_exemplar() -> Instance:
    """Reconstruct the first of the paper's three 3-shot CoT exemplars, verbatim."""
    text = None
    for p in _TRACE_PATHS:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
            break
    if text is None:
        raise FileNotFoundError(
            "published_trace.txt not found; expected one of %s" % (_TRACE_PATHS,)
        )
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    lines = [ln for ln in lines if not ln.startswith("#")]
    qi = next(i for i, ln in enumerate(lines) if ln.startswith("Q: "))
    prompt = lines[qi][3:].strip()
    body = []
    for ln in lines[qi + 1:]:
        if ln.startswith("Q: "):
            break
        if not ln.strip():
            if body:
                break
            continue
        body.append(ln[3:] if ln.startswith("A: ") else ln)
    steps = [ln for ln in body if ln.startswith(("Let's calculate ", "Then, the final"))]
    preamble = [ln for ln in body if ln not in steps]
    states = []
    for i, s in enumerate(steps):
        m = re.search(r"=\s*(-?\d+)\.", s)
        states.append("%s=%s" % (_letter(i), m.group(1)))
    answer = re.search(r"So the answer is (-?\d+)\.", steps[-1]).group(1)
    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=answer,
        depth=len(steps),
        meta={
            "preamble": preamble,
            "source": "published",
            "citation": "Suzgun et al. 2022 (arXiv:2210.09261), cot-prompts/"
                        "multistep_arithmetic_two.txt, exemplar 1 of 3",
        },
    )


def exemplars(k: int, seed: int) -> list[Instance]:
    """k few-shot exemplars; [0] is the verbatim published one, the rest generated."""
    if k < 1:
        return []
    out = [_published_exemplar()]
    i = 0
    while len(out) < k:
        out.append(generate(3, 10_000_000 + seed * 1_000 + i))
        i += 1
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _unredact(inst: Instance, red: str, k: int) -> str | None:
    """Independent inverse of redact_prompt(), for the selftest.

    Walk the prompt and the redacted string in lockstep; wherever the redaction shows a
    placeholder, put back the balanced parenthesised span it must stand for. Returns the
    rebuilt prompt, or None if the redaction was not a clean in-place removal of whole
    parenthesised groups (at most k of them).
    """
    src = inst.prompt
    out: list[str] = []
    i = j = n = 0
    while j < len(red):
        if red.startswith(_PLACEHOLDER, j):
            if i >= len(src) or src[i] != "(":
                return None
            end = _matching_paren(src, i)
            out.append(src[i:end])
            i = end
            j += len(_PLACEHOLDER)
            n += 1
        else:
            if i >= len(src) or src[i] != red[j]:
                return None
            out.append(src[i])
            i += 1
            j += 1
    if i != len(src) or n > k:
        return None
    return "".join(out)


def _selftest() -> int:
    rng = random.Random(20260916)
    fails = 0
    peaks = []
    for trial in range(200):
        depth = rng.choice(DEPTHS)
        seed = rng.randrange(1 << 30)
        inst = generate(depth, seed)
        peaks.append(inst.meta["peak_abs"])

        got = solve(inst)
        if got != inst.answer:
            print("FAIL solve: depth=%d seed=%d %s -> %s != %s"
                  % (depth, seed, inst.prompt, got, inst.answer))
            fails += 1

        if not check(inst, format_cot(inst)):
            print("FAIL check(gold): depth=%d seed=%d" % (depth, seed))
            fails += 1

        wrong = str(int(inst.answer) + 1)
        if check(inst, "Answer: " + wrong):
            print("FAIL check(wrong): depth=%d seed=%d" % (depth, seed))
            fails += 1

        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer) != (
            inst.prompt, inst.steps, inst.states, inst.answer
        ):
            print("FAIL determinism: depth=%d seed=%d" % (depth, seed))
            fails += 1

        if not (len(inst.steps) == len(inst.states) == depth):
            print("FAIL step count: depth=%d seed=%d steps=%d states=%d"
                  % (depth, seed, len(inst.steps), len(inst.states)))
            fails += 1

        spans = step_spans(inst)
        text = format_cot(inst)
        if [text[a:b] for a, b in spans] != inst.steps:
            print("FAIL step_spans: depth=%d seed=%d" % (depth, seed))
            fails += 1

    # --- redaction (AMENDMENT 4): 20 instances, k in {0, 1, depth//2, depth}
    if not REDACTION_MEANINGFUL:
        print("FAIL REDACTION_MEANINGFUL must be True for this task")
        fails += 1
    for trial in range(20):
        depth = rng.choice(DEPTHS)
        seed = rng.randrange(1 << 30)
        inst = generate(depth, seed)
        if redact_prompt(inst, 0) != inst.prompt:
            print("FAIL redact k=0: depth=%d seed=%d" % (depth, seed))
            fails += 1
        prev_digits = sum(c.isdigit() for c in inst.prompt)
        for k in sorted({1, depth // 2, depth}):
            if k < 1:
                continue
            red = redact_prompt(inst, k)
            if red == inst.prompt or _PLACEHOLDER not in red:
                print("FAIL redact k=%d: depth=%d seed=%d %r"
                      % (k, depth, seed, red))
                fails += 1
            # only whole groups go; the surviving text stays byte-identical, so putting
            # the placeholders back where the groups were must rebuild the prompt
            if _unredact(inst, red, k) != inst.prompt:
                print("FAIL redact not in-place: k=%d depth=%d seed=%d %r"
                      % (k, depth, seed, red))
                fails += 1
            n_digits = sum(c.isdigit() for c in red)
            if n_digits > prev_digits:
                print("FAIL redaction not monotone: k=%d depth=%d seed=%d"
                      % (k, depth, seed))
                fails += 1
            prev_digits = n_digits
        # k == depth: no operand and no operator of the expression survives -- only the
        # placeholder and the published "=" question.
        full = redact_prompt(inst, depth)
        if full != _PLACEHOLDER + " =":
            print("FAIL redact k=depth: depth=%d seed=%d %r" % (depth, seed, full))
            fails += 1
        if any(c.isdigit() for c in full) or any(c in "+-*" for c in full):
            print("FAIL redact k=depth leaks tokens: depth=%d seed=%d %r"
                  % (depth, seed, full))
            fails += 1
        for bad_k in (-1, depth + 1):
            try:
                redact_prompt(inst, bad_k)
            except ValueError:
                pass
            else:
                print("FAIL redact k=%d not rejected" % bad_k)
                fails += 1

    # --- step corruption (AMENDMENT 6): 20 instances, k in {1, depth//2, depth}
    for trial in range(20):
        depth = rng.choice(DEPTHS)
        seed = rng.randrange(1 << 30)
        inst = generate(depth, seed)
        for k in sorted({1, depth // 2, depth}):
            if k < 1:
                continue
            text, state = corrupt_step(inst, k, 7)
            true_state = inst.states[k - 1]
            name, _, true_str = true_state.partition("=")
            if state == true_state or text == inst.steps[k - 1]:
                print("FAIL corrupt not a change: k=%d depth=%d seed=%d %r"
                      % (k, depth, seed, text))
                fails += 1
            # same shape: "<name>=<int>", same name as the true state
            m_state = re.match(r"^([A-Z]\d*)=(-?\d+)$", state)
            if not m_state or m_state.group(1) != name:
                print("FAIL corrupt state shape: k=%d depth=%d seed=%d %r"
                      % (k, depth, seed, state))
                fails += 1
                continue
            w = int(m_state.group(2))
            # the line still parses as a step line of the right kind, the action text is
            # byte-identical, and every reported value is the corrupted one
            old, new = inst.steps[k - 1], text
            if k == depth:
                mo, mn = _STEP_ROOT.match(old), _STEP_ROOT.match(new)
                ok = (
                    mo and mn and mn.group(1) == mo.group(1)
                    and int(mn.group(2)) == int(mn.group(3)) == w
                    and int(mo.group(2)) == int(mo.group(3)) == int(true_str)
                )
            else:
                mo, mn = _STEP_LEAF.match(old), _STEP_LEAF.match(new)
                ok = (
                    mo and mn and mn.group(1) == mo.group(1)
                    and mn.group(2) == name and int(mn.group(3)) == w
                    and int(mo.group(3)) == int(true_str)
                )
            if not ok:
                print("FAIL corrupt step structure: k=%d depth=%d seed=%d %r"
                      % (k, depth, seed, new))
                fails += 1
            # plausible magnitude: a small offset or a one-digit edit of the true value
            near, edits = _wrong_values(int(true_str))
            if w not in near and w not in edits:
                print("FAIL corrupt implausible: k=%d depth=%d seed=%d %d -> %d"
                      % (k, depth, seed, int(true_str), w))
                fails += 1
            # determinism, and sensitivity to the corruption seed
            if corrupt_step(inst, k, 7) != (text, state):
                print("FAIL corrupt determinism: k=%d depth=%d seed=%d" % (k, depth, seed))
                fails += 1
            if corrupt_step(generate(depth, seed), k, 7) != (text, state):
                print("FAIL corrupt not a function of the instance: k=%d depth=%d seed=%d"
                      % (k, depth, seed))
                fails += 1
        for bad_k in (0, depth + 1):
            try:
                corrupt_step(inst, bad_k, 0)
            except ValueError:
                pass
            else:
                print("FAIL corrupt k=%d not rejected" % bad_k)
                fails += 1
    # the corruption must actually vary with the seed (not a constant +1)
    probe_inst = generate(7, 4242)
    variants = {corrupt_step(probe_inst, 3, s)[1] for s in range(20)}
    if len(variants) < 3:
        print("FAIL corrupt seed sensitivity: only %r" % (variants,))
        fails += 1
    # and it must work on the verbatim published exemplar, at a leaf and at the root
    pub_probe = _published_exemplar()
    for k in (1, pub_probe.depth):
        ctext, cstate = corrupt_step(pub_probe, k, 1)
        if cstate == pub_probe.states[k - 1] or ctext == pub_probe.steps[k - 1]:
            print("FAIL corrupt on published exemplar: k=%d %r" % (k, ctext))
            fails += 1

    ex = exemplars(3, 0)
    if len(ex) != 3:
        print("FAIL exemplars: got %d" % len(ex))
        fails += 1
    pub = ex[0]
    if pub.meta.get("source") != "published" or pub.answer != "123" or pub.depth != 3:
        print("FAIL published exemplar: %r %r" % (pub.answer, pub.depth))
        fails += 1
    if solve(pub) != pub.answer:
        print("FAIL published exemplar solve: %s != %s" % (solve(pub), pub.answer))
        fails += 1
    for e in ex:
        format_cot(e)
        if not check(e, format_cot(e)):
            print("FAIL exemplar render/check")
            fails += 1
        if redact_prompt(e, 0) != e.prompt or _PLACEHOLDER not in redact_prompt(e, 1):
            print("FAIL exemplar redaction")
            fails += 1
    if redact_prompt(pub, 1) != "(" + _PLACEHOLDER + " * (4 + -7 + 0 * -5)) =":
        print("FAIL published exemplar redaction: %r" % redact_prompt(pub, 1))
        fails += 1

    # tolerant-parser spot checks
    probe = generate(3, 1)
    cases = [
        ("Answer: " + probe.answer, True),
        ("So the answer is %s.\nAnswer: %s" % (probe.answer, probe.answer), True),
        ("blah blah\nSo the answer is %s." % probe.answer, True),
        ("Answer: **%s**" % probe.answer, True),
        ("Answer: \\boxed{%s}" % probe.answer, True),
        ("Answer: %d" % (int(probe.answer) + 7), False),
        ("Answer: nope", False),
        ("", False),
    ]
    for text, want in cases:
        if check(probe, text) != want:
            print("FAIL parser case %r (wanted %s)" % (text, want))
            fails += 1

    # knobs must be honoured
    k = generate(4, 3, chain_len=2, operators="+-", min_operand=0, max_operand=3)
    if "*" in k.prompt or solve(k) != k.answer:
        print("FAIL knobs")
        fails += 1
    try:
        generate(3, 0, nonsense=1)
    except TypeError:
        pass
    else:
        print("FAIL unknown knob not rejected")
        fails += 1

    # differential check of the solver against the vendored fixed BBH targets (250
    # examples, ground truth produced by the paper's own pipeline).
    fixed = os.path.join(_HERE, "vendor", "bbh-suzgun", "bbh",
                         "multistep_arithmetic_two.json")
    if os.path.exists(fixed):
        import json
        with open(fixed, encoding="utf-8") as fh:
            ex_fixed = json.load(fh)["examples"]
        bad = 0
        for e in ex_fixed:
            probe_inst = Instance(prompt=e["input"], steps=[], states=[],
                                  answer=e["target"], depth=0)
            if solve(probe_inst) != e["target"]:
                bad += 1
        if bad:
            print("FAIL vendored BBH cross-check: %d/%d mismatches" % (bad, len(ex_fixed)))
            fails += 1
        else:
            print("vendored BBH cross-check: solve() matches all %d published targets"
                  % len(ex_fixed))
    else:
        print("note: vendored BBH fixed set absent, cross-check skipped")

    peaks.sort()
    print("peak |intermediate|: median %d, p95 %d, max %d"
          % (peaks[len(peaks) // 2], peaks[int(len(peaks) * 0.95)], peaks[-1]))
    print("depths swept: %s" % (DEPTHS,))
    if fails:
        print("SELFTEST FAILED: %d failures" % fails)
        return 1
    print("SELFTEST PASSED: 200 instances, solve/check/determinism/spans/exemplars/"
          "knobs/redaction/corruption OK")
    return 0


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for j in range(3):
            inst = generate(depth, 100 + j)
            print("=" * 88)
            print("depth=%d  seed=%d  groups=%d  answer=%s"
                  % (inst.depth, inst.meta["seed"], inst.meta["n_groups"], inst.answer))
            print("-" * 88)
            print("PROMPT:")
            print(inst.prompt)
            print("-" * 88)
            print("GOLD CoT:")
            print(format_cot(inst))
            print("-" * 88)
            print("STATES: " + " | ".join(inst.states))
            print()


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    if "--demo" in argv:
        _demo()
        return 0
    print(__doc__)
    print("usage: python3 task.py [--selftest | --demo]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
