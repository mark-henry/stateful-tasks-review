#!/usr/bin/env python3
"""boolean_expressions -- BBH-style Boolean formula evaluation with a serial-depth knob.

Primary source: Suzgun et al. 2022, "Challenging BIG-Bench Tasks and Whether
Chain-of-Thought Can Solve Them" (arXiv:2210.09261), BBH task `boolean_expressions`.
Repo: github.com/suzgunmirac/BIG-Bench-Hard @ 9ee07bd481feebf959a6b59d61ea57bdcf30964d (MIT).

This module implements the AMENDMENT-3 contract of ~/stateful-tasks/SPEC.md:

  * the gold trace reproduces BBH's published CoT format (name sub-expressions,
    evaluate innermost-first, "Plugging in ..., we get: ... So the answer is X.");
  * `exemplars(k, seed)[0]` is the verbatim first exemplar of BBH's
    `cot-prompts/boolean_expressions.txt`;
  * the fixed 250-example BBH set is vendored and exposed (`bbh_instances`), AND a
    generator produces fresh non-contaminated instances at arbitrary serial depth.

Pure python stdlib.  No network.  See README.md for the format decision, the depth
semantics and the caveats.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field

SLUG = "boolean_expressions"
_HERE = os.path.dirname(os.path.abspath(__file__))

# BBH's own one-line task description.  Kept out of Instance.prompt (AMENDMENT 3:
# prompt is the problem statement only); the harness may use it as an instruction.
TASK_DESCRIPTION = "Evaluate the result of a random Boolean expression."

# The two boilerplate lines that open every published BBH boolean_expressions CoT.
_PREAMBLE_1 = "Let's think step by step."
_PREAMBLE_2 = (
    "Remember that (i) expressions inside brackets are always evaluated first and "
    'that (ii) the order of operations from highest priority to lowest priority is '
    '"not", "and", "or", respectively.'
)

ANSWER_FORMAT = "exactly one word, either True or False"

# Depth = number of named sub-expression resolutions in the gold trace (== len(steps)).
# depth 2 is canonical BBH difficulty; 16 is far past where a 7B holds a 1-bit state
# across that many dependent reductions.  See README "Depth semantics".
DEPTHS = [2, 3, 4, 6, 8, 12, 16]

KNOBS = {
    "leaf_size": (2, "number of binary/unary operators in the innermost named sub-expression A"),
    "wrap_ops": (1, "number of operators added at each subsequent named level"),
    "operand_ops": (1, "max operators in the literal operand introduced alongside the inner level"),
    "extra_parens": (0.30, "probability of adding a redundant pair of parentheses at a node"),
    "require_dependence": (True, "reject a level whose value does not depend on the inner level's value"),
    "ops": (("and", "or"), "binary operators the generator may use"),
}


@dataclass
class Instance:
    prompt: str
    steps: list
    states: list
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AST used by the GENERATOR only.  Parentheses are explicit nodes, so rendering is
# purely structural and round-trips to the exact surface string.
#   ("const", bool) | ("name", str) | ("hole",) | ("paren", x) | ("not", x)
#   | ("and", l, r) | ("or", l, r)
# ---------------------------------------------------------------------------

_PREC = {"or": 1, "and": 2, "not": 3, "paren": 4, "const": 4, "name": 4}


def _prec(node, hole_prec=4):
    if node[0] == "hole":
        return hole_prec
    return _PREC[node[0]]


def _render(node):
    k = node[0]
    if k == "const":
        return "True" if node[1] else "False"
    if k == "name":
        return node[1]
    if k == "hole":
        return "?"
    if k == "paren":
        return "( " + _render(node[1]) + " )"
    if k == "not":
        return "not " + _render(node[1])
    return _render(node[1]) + " " + k + " " + _render(node[2])


def _mk_not(x, hole_prec=4):
    if _prec(x, hole_prec) < _PREC["not"]:
        x = ("paren", x)
    return ("not", x)


def _mk_bin(op, l, r, hole_prec=4):
    p = _PREC[op]
    if _prec(l, hole_prec) < p:
        l = ("paren", l)
    if _prec(r, hole_prec) <= p:
        r = ("paren", r)
    return (op, l, r)


def _ast_eval(node):
    k = node[0]
    if k == "const":
        return node[1]
    if k == "paren":
        return _ast_eval(node[1])
    if k == "not":
        return not _ast_eval(node[1])
    if k == "and":
        return _ast_eval(node[1]) and _ast_eval(node[2])
    if k == "or":
        return _ast_eval(node[1]) or _ast_eval(node[2])
    raise ValueError("cannot evaluate node " + repr(node))


def _subst_hole(node, filler):
    k = node[0]
    if k == "hole":
        return filler
    if k in ("const", "name"):
        return node
    if k in ("paren", "not"):
        return (k, _subst_hole(node[1], filler))
    return (k, _subst_hole(node[1], filler), _subst_hole(node[2], filler))


def _crit(node):
    """Longest chain of dependent ELEMENTARY reductions rooted at node."""
    k = node[0]
    if k in ("const", "name", "hole"):
        return 0
    if k in ("paren", "not"):
        return _crit(node[1]) + 1
    return max(_crit(node[1]), _crit(node[2])) + 1


# --- innermost-first reduction, used to render the "= ... = ..." chains ---------

def _reduce_once(n):
    """Reduce the leftmost-innermost reducible node.  Returns (node, changed)."""
    k = n[0]
    if k in ("const", "name", "hole"):
        return n, False
    if k == "paren":
        x, ch = _reduce_once(n[1])
        if ch:
            return ("paren", x), True
        if n[1][0] == "const":
            return n[1], True
        return n, False
    if k == "not":
        x, ch = _reduce_once(n[1])
        if ch:
            return ("not", x), True
        if n[1][0] == "const":
            return ("const", not n[1][1]), True
        return n, False
    l, ch = _reduce_once(n[1])
    if ch:
        return (k, l, n[2]), True
    r, ch = _reduce_once(n[2])
    if ch:
        return (k, n[1], r), True
    if n[1][0] == "const" and n[2][0] == "const":
        v = (n[1][1] and n[2][1]) if k == "and" else (n[1][1] or n[2][1])
        return ("const", v), True
    return n, False


def _strip_const_parens(n):
    """Drop every `( True )` / `( False )` wrapper, as the published trace does."""
    k = n[0]
    if k in ("const", "name", "hole"):
        return n
    if k == "paren":
        x = _strip_const_parens(n[1])
        return x if x[0] == "const" else ("paren", x)
    if k == "not":
        return ("not", _strip_const_parens(n[1]))
    return (k, _strip_const_parens(n[1]), _strip_const_parens(n[2]))


def _reduction_chain(node, prefix=None):
    """['expr', 'expr after one reduction', ..., 'True'] with duplicates collapsed."""
    chain = list(prefix or [])
    s = _render(node)
    if not chain or chain[-1] != s:
        chain.append(s)
    cur = node
    guard = 0
    while cur[0] != "const":
        guard += 1
        if guard > 10000:
            raise RuntimeError("reduction did not terminate")
        cur, changed = _reduce_once(cur)
        if not changed:
            raise RuntimeError("stuck reducing " + _render(cur))
        cur = _strip_const_parens(cur)
        s = _render(cur)
        if s != chain[-1]:
            chain.append(s)
    return chain


# ---------------------------------------------------------------------------
# Independent reference evaluator (used by solve()).  Deliberately a SECOND,
# separate implementation: a direct-evaluating recursive-descent parser over the
# surface string, sharing no code with the generator's AST.  This makes
# solve() == answer a real differential test of the renderer.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\(|\)|not|and|or|True|False|\S+")


class _ParseError(ValueError):
    pass


def _eval_expression(text):
    toks = _TOKEN_RE.findall(text.strip())
    for t in toks:
        if t not in ("(", ")", "not", "and", "or", "True", "False"):
            raise _ParseError("bad token %r in %r" % (t, text))
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def p_or():
        v = p_and()
        while peek() == "or":
            take()
            v = p_and() or v
        return v

    def p_and():
        v = p_not()
        while peek() == "and":
            take()
            v = p_not() and v
        return v

    def p_not():
        if peek() == "not":
            take()
            return not p_not()
        return p_atom()

    def p_atom():
        t = take()
        if t == "(":
            v = p_or()
            if take() != ")":
                raise _ParseError("expected ) in %r" % text)
            return v
        if t == "True":
            return True
        if t == "False":
            return False
        raise _ParseError("unexpected %r in %r" % (t, text))

    if not toks:
        raise _ParseError("empty expression")
    v = p_or()
    if pos[0] != len(toks):
        raise _ParseError("trailing tokens in %r" % text)
    return v


# A second parser that builds the generator AST -- only used for the FIXED BBH set,
# where we have a surface string but no tree.  Not used by solve().

def _parse_ast(text):
    toks = _TOKEN_RE.findall(text.strip())
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def p_or():
        v = p_and()
        while peek() == "or":
            take()
            v = ("or", v, p_and())
        return v

    def p_and():
        v = p_not()
        while peek() == "and":
            take()
            v = ("and", v, p_not())
        return v

    def p_not():
        if peek() == "not":
            take()
            return ("not", p_not())
        return p_atom()

    def p_atom():
        t = take()
        if t == "(":
            v = p_or()
            if take() != ")":
                raise _ParseError("expected )")
            return ("paren", v)
        if t == "True":
            return ("const", True)
        if t == "False":
            return ("const", False)
        raise _ParseError("unexpected %r" % (t,))

    v = p_or()
    if pos[0] != len(toks):
        raise _ParseError("trailing tokens")
    return v


# ---------------------------------------------------------------------------
# Trace rendering, in the published BBH format
# ---------------------------------------------------------------------------

_NAMES = [chr(ord("A") + i) for i in range(26)]


def _bool_str(b):
    return "True" if b else "False"


def _join_clauses(clauses):
    if len(clauses) == 1:
        return clauses[0]
    return ", ".join(clauses[:-1]) + " and " + clauses[-1]


def _build_trace(root, templates, leaf):
    """Render the gold trace.

    root       -- full AST of Z
    templates  -- list of one-hole templates, templates[i] builds level i+1 from level i;
                  templates[-1] builds Z.  len(templates) == number of named levels.
    leaf       -- AST of the innermost named sub-expression A (None when there are none).

    Returns (preamble_lines, steps, states, answer).
    """
    m = len(templates) - 1 if templates else 0  # number of NAMED sub-expressions
    if m > len(_NAMES):
        raise ValueError("depth too large: need %d names, have %d" % (m, len(_NAMES)))
    names = _NAMES[:m]

    full = _render(root)
    steps, states = [], []

    if m == 0:
        # Degenerate depth-1 instance: nothing to name, one reduction line.
        chain = _reduction_chain(root)
        val = _ast_eval(root)
        steps.append(
            "Z = " + " = ".join(chain) + ". So the answer is " + _bool_str(val) + "."
        )
        states.append("Z=" + _bool_str(val))
        return [_PREAMBLE_1, _PREAMBLE_2], steps, states, _bool_str(val)

    # level nodes: level_nodes[0] = A (the leaf), level_nodes[i] = templates[i](level i-1)
    level_nodes = [leaf]
    for i in range(1, m):
        level_nodes.append(_subst_hole(templates[i], level_nodes[i - 1]))
    level_vals = [_ast_eval(n) for n in level_nodes]

    # skeletons: what each level looks like with the inner level replaced by its NAME
    skeletons = [_render(leaf)]
    for i in range(1, m):
        skeletons.append(_render(_subst_hole(templates[i], ("name", names[i - 1]))))
    z_skeleton = _render(_subst_hole(templates[m], ("name", names[m - 1])))

    clauses = ['"%s = %s"' % (names[i], skeletons[i]) for i in range(m)]
    simplify = (
        'We first simplify this expression "Z" as follows: "Z = %s = %s" where %s.'
        % (full, z_skeleton, _join_clauses(clauses))
    )

    for i in range(m):
        if i == 0:
            chain = _reduction_chain(leaf)
        else:
            sub = _subst_hole(templates[i], ("const", level_vals[i - 1]))
            chain = _reduction_chain(sub, prefix=[skeletons[i]])
        steps.append(
            "Let's evaluate %s: %s = %s." % (names[i], names[i], " = ".join(chain))
        )
        states.append("%s=%s" % (names[i], _bool_str(level_vals[i])))

    z_sub = _subst_hole(templates[m], ("const", level_vals[m - 1]))
    z_chain = _reduction_chain(z_sub, prefix=[z_skeleton])
    answer = _bool_str(_ast_eval(root))
    steps.append(
        "Plugging in %s, we get: Z = %s. So the answer is %s."
        % (names[m - 1], " = ".join(z_chain), answer)
    )
    states.append("Z=" + answer)
    return [_PREAMBLE_1, _PREAMBLE_2, simplify], steps, states, answer


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _resolve_knobs(knobs):
    out = {k: v[0] for k, v in KNOBS.items()}
    for k, v in knobs.items():
        if k not in KNOBS:
            raise TypeError("unknown knob %r" % (k,))
        out[k] = v
    out["ops"] = tuple(out["ops"])
    if not out["ops"]:
        raise ValueError("ops must be non-empty")
    return out


def _knob_key(kn):
    return "|".join("%s=%s" % (k, kn[k]) for k in sorted(kn))


def _random_const_expr(rng, n_ops, kn):
    """A random expression over True/False literals with exactly n_ops operators."""
    if n_ops <= 0:
        node = ("const", rng.random() < 0.5)
    elif rng.random() < 0.34:
        node = _mk_not(_random_const_expr(rng, n_ops - 1, kn))
    else:
        left = rng.randint(0, n_ops - 1)
        node = _mk_bin(
            rng.choice(kn["ops"]),
            _random_const_expr(rng, left, kn),
            _random_const_expr(rng, n_ops - 1 - left, kn),
        )
    if n_ops > 0 and rng.random() < kn["extra_parens"]:
        node = ("paren", node)
    return node


def _random_template(rng, inner_prec, kn):
    """A one-hole template adding kn['wrap_ops'] operators around the hole."""
    cur = ("hole",)
    for _ in range(max(1, int(kn["wrap_ops"]))):
        pick = rng.random()
        if pick < 0.34:
            cur = _mk_not(cur, hole_prec=inner_prec)
        else:
            operand = _random_const_expr(rng, rng.randint(0, int(kn["operand_ops"])), kn)
            op = rng.choice(kn["ops"])
            if pick < 0.67:
                cur = _mk_bin(op, cur, operand, hole_prec=inner_prec)
            else:
                cur = _mk_bin(op, operand, cur, hole_prec=inner_prec)
        inner_prec = _PREC[cur[0]]
    if rng.random() < kn["extra_parens"]:
        cur = ("paren", cur)
    return cur


def _template_depends(tmpl):
    a = _ast_eval(_subst_hole(tmpl, ("const", True)))
    b = _ast_eval(_subst_hole(tmpl, ("const", False)))
    return a != b


def generate(depth, seed, **knobs):
    """Fresh instance with `depth` serial named-sub-expression resolutions."""
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    kn = _resolve_knobs(knobs)
    if depth - 1 > len(_NAMES):
        raise ValueError("depth %d needs %d names; max %d" % (depth, depth - 1, len(_NAMES)))
    rng = random.Random("%s|d=%d|s=%d|%s" % (SLUG, depth, int(seed), _knob_key(kn)))

    m = depth - 1  # number of named sub-expressions A, B, C, ...
    if m == 0:
        root = _random_const_expr(rng, max(1, int(kn["leaf_size"])), kn)
        templates, leaf = [], None
        level_nodes = []
    else:
        leaf = _random_const_expr(rng, max(1, int(kn["leaf_size"])), kn)
        templates = [None]  # templates[0] is unused: level 0 IS the leaf
        cur = leaf
        level_nodes = [leaf]
        for _ in range(m):  # m-1 intermediate levels + 1 for Z
            for _try in range(200):
                t = _random_template(rng, _prec(cur), kn)
                if not kn["require_dependence"] or _template_depends(t):
                    break
            else:
                raise RuntimeError("could not build a dependent level")
            templates.append(t)
            cur = _subst_hole(t, cur)
            level_nodes.append(cur)
        root = cur

    preamble, steps, states, answer = _build_trace(root, templates, leaf)
    expr = _render(root)
    inst = Instance(
        prompt=expr + " is",
        steps=steps,
        states=states,
        answer=answer,
        depth=depth,
        meta={
            "seed": int(seed),
            "source": "generated",
            "expression": expr,
            "preamble": preamble,
            "names": _NAMES[:m],
            "level_values": [_bool_str(_ast_eval(n)) for n in level_nodes],
            "elementary_depth": _crit(root),
            "n_operators": _count_ops(root),
            "knobs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kn.items()},
        },
    )
    return inst


def _count_ops(node):
    k = node[0]
    if k in ("const", "name", "hole"):
        return 0
    if k == "paren":
        return _count_ops(node[1])
    if k == "not":
        return 1 + _count_ops(node[1])
    return 1 + _count_ops(node[1]) + _count_ops(node[2])


# ---------------------------------------------------------------------------
# Reference solver -- independent of generate()
# ---------------------------------------------------------------------------

def solve(inst):
    """Evaluate inst.prompt from scratch with the independent recursive-descent
    evaluator.  Never looks at inst.steps / inst.states / inst.answer / inst.meta."""
    text = inst.prompt.strip()
    if text.endswith(" is"):
        text = text[:-3]
    return "True" if _eval_expression(text) else "False"


# ---------------------------------------------------------------------------
# Gold completion + answer checking
# ---------------------------------------------------------------------------

def format_cot(inst):
    preamble = inst.meta.get("preamble") or [_PREAMBLE_1, _PREAMBLE_2]
    return "\n".join(list(preamble) + list(inst.steps)) + "\nAnswer: " + inst.answer


def step_spans(inst):
    """Char offsets (start, end) of each element of inst.steps inside format_cot()."""
    text = format_cot(inst)
    spans, cursor = [], 0
    for s in inst.steps:
        i = text.index(s, cursor)
        spans.append((i, i + len(s)))
        cursor = i + len(s)
    return spans


_ANSWER_LINE_RE = re.compile(r"^[ \t>*_-]*answer\s*[:\-]\s*(.*)$", re.IGNORECASE)


def _normalize_answer(raw):
    s = raw.strip()
    s = s.replace("**", "").replace("`", "").replace("$", "")
    s = s.strip().strip('"').strip("'")
    s = s.rstrip(".").rstrip("!").strip()
    parts = s.split()
    if parts:
        s = parts[-1]
    s = s.strip().strip('"').strip("'").rstrip(".").strip()
    return s.lower()


def check(inst, completion):
    """Extract the last `Answer:` line (else the last non-empty line, else the text
    after the last 'the answer is'), then exact-match against inst.answer."""
    if completion is None:
        return False
    lines = completion.splitlines()
    candidate = None
    for line in reversed(lines):
        m = _ANSWER_LINE_RE.match(line)
        if m and m.group(1).strip():
            candidate = m.group(1)
            break
    if candidate is None:
        for line in reversed(lines):
            if line.strip():
                candidate = line
                break
    if candidate is None:
        return False
    low = candidate.lower()
    if "the answer is" in low:
        candidate = candidate[low.rindex("the answer is") + len("the answer is"):]
    got = _normalize_answer(candidate)
    if got not in ("true", "false"):
        return False
    return got == inst.answer.lower()


# ---------------------------------------------------------------------------
# Prompt redaction (SPEC AMENDMENT 4: prompt blinding)
# ---------------------------------------------------------------------------

# For this task the expression IS the operator list, so redaction removes the
# sub-expressions that steps 1..k have already resolved -- i.e. the named groups
# A, B, ... in evaluation order -- and leaves the enclosing structure and the
# not-yet-evaluated operands intact.  Because the named levels are nested
# (A inside B inside ... inside Z), the k already-resolved levels form ONE
# contiguous span: the outermost resolved level, which is the (k-1)-th name.
# That span is replaced by a single placeholder, per "once per removed span".
REDACTION_MEANINGFUL = True

REDACTION_PLACEHOLDER = "[\u2026]"  # the literal string "[...]" with a real ellipsis


def _trace_skeletons(inst):
    """Read the named levels back out of the gold trace.

    Returns (skeletons, z_skeleton) where skeletons[i] is level i rendered with
    level i-1 replaced by its NAME, and z_skeleton is the whole expression with
    the outermost named level replaced by its name.  (None, None) for a
    degenerate depth-1 instance, which has no named sub-expressions.

    This reads inst.steps rather than re-deriving from a tree, so it works for
    generated instances, the vendored BBH set and the verbatim published
    exemplar alike.  The first "=" clause of each step is, by construction of
    _build_trace, exactly that level's skeleton.
    """
    steps = list(inst.steps)
    if len(steps) < 2:
        return None, None
    skeletons = []
    for s in steps[:-1]:
        body = s.split(": ", 1)[1]           # "A = <skeleton> = ... = True."
        skeletons.append(body.split(" = ")[1])
    body = steps[-1].split(", we get: ", 1)[1]  # "Z = <z_skeleton> = ... . So ..."
    return skeletons, body.split(" = ")[1]


def _expand_levels(skeletons, z_skeleton, stop):
    """Substitute skeletons back into z_skeleton down to (but not including)
    level `stop`.  stop=0 rebuilds the full expression."""
    text = z_skeleton
    for i in range(len(skeletons) - 1, stop - 1, -1):
        repl = skeletons[i]
        text = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % _NAMES[i], lambda m: repl, text)
    return text


def _prompt_expression(inst):
    expr = inst.meta.get("expression")
    if expr:
        return expr
    text = inst.prompt.strip()
    return text[:-3].rstrip() if text.endswith(" is") else text


def redact_prompt(inst, k):
    """The problem statement with the sub-expressions resolved by steps 1..k
    replaced by the placeholder, in place.

    k == 0 returns inst.prompt unchanged; k == inst.depth removes the whole
    expression (the last step resolves Z itself), leaving only the question.
    The outer structure and every operator/operand not yet consumed survive.
    """
    k = int(k)
    if k < 0 or k > inst.depth:
        raise ValueError("k must be in 0..%d, got %d" % (inst.depth, k))
    if k == 0:
        return inst.prompt

    skeletons, z_skeleton = _trace_skeletons(inst)
    if skeletons is None:  # degenerate depth-1 instance: one step, resolves Z
        full = _prompt_expression(inst)
        redacted = REDACTION_PLACEHOLDER
    else:
        full = _expand_levels(skeletons, z_skeleton, 0)
        if k > len(skeletons):  # k == depth: the final step resolves Z itself
            redacted = REDACTION_PLACEHOLDER
        else:
            redacted = _expand_levels(skeletons, z_skeleton, k)
            redacted = re.sub(
                r"(?<![A-Za-z])%s(?![A-Za-z])" % _NAMES[k - 1],
                lambda m: REDACTION_PLACEHOLDER,
                redacted,
            )

    i = inst.prompt.find(full)
    if i < 0:
        raise ValueError("could not locate the expression inside the prompt")
    return inst.prompt[:i] + redacted + inst.prompt[i + len(full):]


# ---------------------------------------------------------------------------
# The fixed, published BBH material
# ---------------------------------------------------------------------------

_BBH_DIR = os.path.join(_HERE, "vendor", "BIG-Bench-Hard")
_BBH_JSON = os.path.join(_BBH_DIR, "bbh", "boolean_expressions.json")
_BBH_COT = os.path.join(_BBH_DIR, "cot-prompts", "boolean_expressions.txt")
_PUBLISHED_TRACE = os.path.join(_HERE, "published_trace.txt")

# Fallback copy of the first published exemplar, used only if neither the vendored
# cot-prompt file nor published_trace.txt is readable.  Verbatim from
# suzgunmirac/BIG-Bench-Hard @ 9ee07bd, cot-prompts/boolean_expressions.txt.
_FALLBACK_EXEMPLAR = (
    "not ( ( not not True ) ) is",
    [
        "Let's think step by step.",
        'Remember that (i) expressions inside brackets are always evaluated first and that (ii) the order of operations from highest priority to lowest priority is "not", "and", "or", respectively.',
        'We first simplify this expression "Z" as follows: "Z = not ( ( not not True ) ) = not ( ( A ) )" where "A = not not True".',
        "Let's evaluate A: A = not not True = not (not True) = not False = True.",
        "Plugging in A, we get: Z = not ( ( A ) ) = not ( ( True ) ) = not True = False. So the answer is False.",
    ],
)


def _read_published_exemplar():
    """(prompt, all_lines) of BBH's FIRST published CoT exemplar, verbatim."""
    for path, is_cot_prompt in ((_BBH_COT, True), (_PUBLISHED_TRACE, False)):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            continue
        if is_cot_prompt and "-----" in raw:
            raw = raw.split("-----", 1)[1]
        body = [ln for ln in raw.splitlines() if not ln.startswith("#")]
        qi = next((i for i, ln in enumerate(body) if ln.startswith("Q: ")), None)
        if qi is None:
            continue
        prompt = body[qi][3:].strip()
        lines = []
        for ln in body[qi + 1:]:
            if not ln.strip():
                break
            lines.append(ln[3:] if ln.startswith("A: ") else ln)
        if lines:
            return prompt, lines
    return _FALLBACK_EXEMPLAR


def published_exemplar():
    """The verbatim BBH exemplar as an Instance (depth 2)."""
    prompt, lines = _read_published_exemplar()
    preamble, steps = lines[:-2], lines[-2:]
    answer = "True" if _eval_expression(prompt[:-3] if prompt.endswith(" is") else prompt) else "False"
    # states: the value each step establishes, read off the trace's own names.
    states = []
    for s in steps[:-1]:
        name = s.split(":", 1)[0].replace("Let's evaluate", "").strip()
        val = s.rstrip(".").rsplit("=", 1)[-1].strip()
        states.append("%s=%s" % (name, val))
    states.append("Z=" + answer)
    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=answer,
        depth=len(steps),
        meta={
            "source": "bbh_cot_prompt",
            "verbatim": True,
            "citation": "Suzgun et al. 2022 (arXiv:2210.09261), BBH cot-prompts/boolean_expressions.txt @ 9ee07bd",
            "preamble": preamble,
            "expression": prompt[:-3] if prompt.endswith(" is") else prompt,
        },
    )


def bbh_examples():
    """The fixed 250 canonical BBH examples as (input, target) pairs."""
    with open(_BBH_JSON, "r", encoding="utf-8") as fh:
        return [(e["input"], e["target"]) for e in json.load(fh)["examples"]]


def _chain_decompose(node):
    """Name the binary nodes along the critical path of an arbitrary parsed tree.

    Returns (templates, leaf) in the same shape generate() produces, so the fixed BBH
    set renders through exactly the same trace renderer.  Levels are cut at binary
    nodes on the critical path; unary/paren wrappers are absorbed into the level above.
    """
    # Walk the critical path from the root down, recording the cut points.
    path = []
    cur = node
    while cur[0] in ("paren", "not", "and", "or"):
        if cur[0] in ("paren", "not"):
            nxt = cur[1]
        else:
            nxt = cur[1] if _crit(cur[1]) >= _crit(cur[2]) else cur[2]
        path.append((cur, nxt))
        cur = nxt
    cuts = [child for parent, child in path if parent[0] in ("and", "or") and _crit(child) > 0]
    if not cuts:
        return [], None

    def hole_at(root_node, target):
        """Copy root_node with `target` (by identity) replaced by ("hole",)."""
        if root_node is target:
            return ("hole",)
        k = root_node[0]
        if k in ("const", "name", "hole"):
            return root_node
        if k in ("paren", "not"):
            return (k, hole_at(root_node[1], target))
        return (k, hole_at(root_node[1], target), hole_at(root_node[2], target))

    inner_first = list(reversed(cuts))  # innermost named sub-expression first
    leaf = inner_first[0]
    templates = [None]
    outers = inner_first[1:] + [node]
    for i, outer in enumerate(outers):
        templates.append(hole_at(outer, inner_first[i]))
    return templates, leaf


def bbh_instances():
    """All 250 canonical BBH examples as Instances, with reconstructed gold traces.

    BBH ships no per-example CoT (only the 3 prompt exemplars), so `steps` here are
    OURS, rendered in the published format.  `answer` is BBH's own target.
    """
    out = []
    for i, (inp, target) in enumerate(bbh_examples()):
        expr = inp[:-3] if inp.endswith(" is") else inp
        root = _parse_ast(expr)
        templates, leaf = _chain_decompose(root)
        preamble, steps, states, answer = _build_trace(root, templates, leaf)
        assert answer == target, (i, inp, answer, target)
        out.append(
            Instance(
                prompt=inp,
                steps=steps,
                states=states,
                answer=answer,
                depth=len(steps),
                meta={
                    "source": "bbh_fixed",
                    "bbh_index": i,
                    "expression": expr,
                    "preamble": preamble,
                    "names": _NAMES[: len(steps) - 1],
                    "elementary_depth": _crit(root),
                    "trace_is_reconstructed": True,
                },
            )
        )
    return out


def exemplars(k=3, seed=0):
    """Few-shot exemplars.  [0] is the verbatim published BBH exemplar."""
    k = int(k)
    if k <= 0:
        return []
    out = [published_exemplar()]
    modest = [3, 4]
    for i in range(1, k):
        out.append(generate(depth=modest[(i - 1) % len(modest)], seed=1_000_000 + 97 * seed + i))
    return out


# ---------------------------------------------------------------------------
# selftest / demo
# ---------------------------------------------------------------------------

def _verify_trace(inst):
    """Re-derive every intermediate expression in the gold trace with the INDEPENDENT
    evaluator: substitute the already-established name values into each `= ... =`
    element of each step and check it evaluates to that step's recorded state.
    Catches mis-parenthesised skeletons and bad substitutions."""
    problems = []
    known = {}
    for step, state in zip(inst.steps, inst.states):
        name, val = state.split("=")
        body = step
        if body.startswith("Let's evaluate "):
            body = body.split(":", 1)[1]
        elif body.startswith("Plugging in "):
            body = body.split(", we get:", 1)[1]
        body = body.split(". So the answer is")[0].rstrip().rstrip(".")
        body = body.split("=", 1)[1] if body.lstrip().startswith(name + " =") or \
            body.lstrip().startswith("Z =") else body
        for piece in body.split(" = "):
            text = piece.strip()
            for nm, nv in known.items():
                text = re.sub(r"(?<![A-Za-z])%s(?![A-Za-z])" % nm, nv, text)
            try:
                got = "True" if _eval_expression(text) else "False"
            except _ParseError as exc:
                problems.append("unparseable %r in step %r (%s)" % (text, step[:40], exc))
                continue
            if got != val:
                problems.append("%r evaluates to %s, step claims %s" % (text, got, val))
        known[name] = val
    return problems


def _selftest():
    rng = random.Random(20260916)
    depths = DEPTHS + [1, 5, 7, 10, 20]
    fails = []
    n = 200
    for i in range(n):
        d = rng.choice(depths)
        s = rng.randrange(1 << 30)
        inst = generate(d, s)
        # determinism
        again = generate(d, s)
        if (again.prompt, again.steps, again.states, again.answer) != (
            inst.prompt, inst.steps, inst.states, inst.answer):
            fails.append("nondeterministic at depth=%d seed=%d" % (d, s))
        # independent solver
        got = solve(inst)
        if got != inst.answer:
            fails.append("solve mismatch d=%d s=%d: %s != %s (%s)" % (d, s, got, inst.answer, inst.prompt))
        # gold completion checks out
        if not check(inst, format_cot(inst)):
            fails.append("check(gold) False at d=%d s=%d" % (d, s))
        wrong = "False" if inst.answer == "True" else "True"
        if check(inst, "Answer: " + wrong):
            fails.append("check(wrong) True at d=%d s=%d" % (d, s))
        # shape
        if not (len(inst.steps) == len(inst.states) == inst.depth == d):
            fails.append("shape mismatch d=%d s=%d: %d/%d" % (d, s, len(inst.steps), len(inst.states)))
        # every step is a contiguous substring of the gold completion, in order
        try:
            spans = step_spans(inst)
            if spans != sorted(spans):
                fails.append("step spans out of order d=%d s=%d" % (d, s))
        except ValueError:
            fails.append("step not found in format_cot d=%d s=%d" % (d, s))
        # each step ends by asserting the state it records
        for st, state in zip(inst.steps, inst.states):
            name, val = state.split("=")
            if not st.rstrip(".").endswith(val):
                if ("So the answer is %s." % val) not in st:
                    fails.append("state %s not at end of step d=%d s=%d" % (state, d, s))
        # every intermediate expression in the trace re-evaluates correctly
        for p in _verify_trace(inst):
            fails.append("trace d=%d s=%d: %s" % (d, s, p))
    print("[1/7] %d random instances: solve/check/determinism/shape/trace ... %s"
          % (n, "OK" if not fails else "%d FAILURES" % len(fails)))

    # knob sweep
    kn_fails = []
    for kn in (
        {"leaf_size": 1},
        {"leaf_size": 4},
        {"wrap_ops": 2},
        {"wrap_ops": 3, "operand_ops": 2},
        {"extra_parens": 0.0},
        {"extra_parens": 0.9},
        {"require_dependence": False},
        {"ops": ("and",)},
        {"ops": ("or",)},
    ):
        for d in (2, 5, 9):
            for s in range(6):
                inst = generate(d, s, **kn)
                if solve(inst) != inst.answer:
                    kn_fails.append((kn, d, s, inst.prompt))
                if len(inst.steps) != d:
                    kn_fails.append(("shape", kn, d, s))
    print("[2/7] knob sweep (9 knob settings x 3 depths x 6 seeds) ... %s"
          % ("OK" if not kn_fails else "%d FAILURES" % len(kn_fails)))
    fails += ["knob: %r" % (f,) for f in kn_fails]

    # dependence: with require_dependence, flipping the innermost value flips Z
    dep_fails = []
    for s in range(40):
        inst = generate(6, s)
        vals = inst.meta["level_values"]
        if len(vals) != 6:
            dep_fails.append(("level_values", s, len(vals)))
    print("[3/7] level bookkeeping ... %s" % ("OK" if not dep_fails else "%d FAILURES" % len(dep_fails)))
    fails += ["dep: %r" % (f,) for f in dep_fails]

    # verbatim published exemplar
    ex_fails = []
    try:
        ex = exemplars(3, 0)
        e0 = ex[0]
        pub_prompt, pub_lines = _read_published_exemplar()
        if e0.prompt != pub_prompt:
            ex_fails.append("exemplar prompt not verbatim")
        if list(e0.meta["preamble"]) + list(e0.steps) != pub_lines:
            ex_fails.append("exemplar trace not verbatim")
        if e0.prompt != "not ( ( not not True ) ) is":
            ex_fails.append("unexpected published exemplar: %r" % e0.prompt)
        if e0.answer != "False" or solve(e0) != "False":
            ex_fails.append("exemplar answer wrong")
        if e0.states != ["A=True", "Z=False"]:
            ex_fails.append("exemplar states wrong: %r" % (e0.states,))
        if not check(e0, format_cot(e0)):
            ex_fails.append("check(gold) False for exemplar")
        for e in ex[1:]:
            if solve(e) != e.answer:
                ex_fails.append("generated exemplar solve mismatch")
            format_cot(e)
        if len(ex) != 3:
            ex_fails.append("exemplars(3,0) returned %d" % len(ex))
    except Exception as exc:  # pragma: no cover
        ex_fails.append("exemplars raised %r" % (exc,))
    print("[4/7] exemplars(3,0): [0] verbatim BBH, rest render ... %s"
          % ("OK" if not ex_fails else "%d FAILURES" % len(ex_fails)))
    fails += ex_fails

    # the fixed BBH set: independent solver must reproduce all 250 published targets
    bbh_fails = []
    try:
        pairs = bbh_examples()
        for inp, target in pairs:
            expr = inp[:-3] if inp.endswith(" is") else inp
            got = "True" if _eval_expression(expr) else "False"
            if got != target:
                bbh_fails.append((inp, got, target))
        insts = bbh_instances()
        for inst in insts:
            if solve(inst) != inst.answer:
                bbh_fails.append(("instance", inst.prompt))
            if len(inst.steps) != len(inst.states):
                bbh_fails.append(("shape", inst.prompt))
            if not check(inst, format_cot(inst)):
                bbh_fails.append(("check", inst.prompt))
            for p in _verify_trace(inst):
                bbh_fails.append(("trace", inst.prompt, p))
        depth_hist = {}
        for inst in insts:
            depth_hist[inst.depth] = depth_hist.get(inst.depth, 0) + 1
        print("[5/7] vendored BBH set: %d examples, solver matches published targets ... %s"
              "  (reconstructed gold-trace depths: %s)"
              % (len(pairs), "OK" if not bbh_fails else "%d FAILURES" % len(bbh_fails),
                 ", ".join("d%d:%d" % kv for kv in sorted(depth_hist.items()))))
    except OSError as exc:
        print("[5/7] vendored BBH set ... SKIPPED (%s)" % exc)
    fails += ["bbh: %r" % (f,) for f in bbh_fails]

    # tolerant parsing
    p_fails = []
    inst = generate(4, 7)
    a = inst.answer
    other = "False" if a == "True" else "True"
    good = [
        "Answer: " + a,
        "blah blah\nAnswer: %s" % a,
        "Answer: %s." % a,
        "Answer: **%s**" % a,
        "Answer:  %s  " % a.lower(),
        "... So the answer is %s." % a,
        "Some rambling\nSo the answer is %s." % a,
    ]
    bad = [
        "Answer: " + other,
        "So the answer is %s." % other,
        "Answer: maybe",
        "",
        "Answer:",
    ]
    for c in good:
        if not check(inst, c):
            p_fails.append(("should accept", c))
    for c in bad:
        if check(inst, c):
            p_fails.append(("should reject", c))
    print("[6/7] tolerant answer extraction (%d accept / %d reject cases) ... %s"
          % (len(good), len(bad), "OK" if not p_fails else "%d FAILURES" % len(p_fails)))
    fails += ["parse: %r" % (f,) for f in p_fails]

    # AMENDMENT 4: prompt redaction
    r_fails = []
    rng2 = random.Random(20260918)
    red_depths = [d for d in depths if d >= 2]
    cases = [(rng2.choice(red_depths), rng2.randrange(1 << 30)) for _ in range(20)]
    for d, s in cases:
        inst = generate(d, s)
        for k in sorted({0, 1, inst.depth // 2, inst.depth}):
            try:
                red = redact_prompt(inst, k)
            except Exception as exc:
                r_fails.append("redact_prompt raised at d=%d s=%d k=%d: %r" % (d, s, k, exc))
                continue
            if k == 0:
                if red != inst.prompt:
                    r_fails.append("k=0 not identity at d=%d s=%d" % (d, s))
                continue
            if red == inst.prompt:
                r_fails.append("k=%d does not change the prompt at d=%d s=%d" % (k, d, s))
            if REDACTION_PLACEHOLDER not in red:
                r_fails.append("k=%d has no placeholder at d=%d s=%d" % (k, d, s))
            if red.count(REDACTION_PLACEHOLDER) != 1:
                r_fails.append("k=%d has %d placeholders (want 1) at d=%d s=%d"
                               % (k, red.count(REDACTION_PLACEHOLDER), d, s))
            if not red.endswith(" is"):
                r_fails.append("k=%d dropped the question at d=%d s=%d" % (k, d, s))
            if k == inst.depth:
                # task-specific: nothing of the initial state (True/False literals)
                # or of the operator list survives -- only the question.
                if red != REDACTION_PLACEHOLDER + " is":
                    r_fails.append("k=depth left %r at d=%d s=%d" % (red, d, s))
                for tok in ("True", "False", "not", "and", "or", "(", ")"):
                    if tok in red:
                        r_fails.append("k=depth still shows %r at d=%d s=%d" % (tok, d, s))
            else:
                # the unresolved tail must survive verbatim: plugging the value of
                # the redacted level back in reproduces the original answer.
                probe = red.replace(REDACTION_PLACEHOLDER, "( %s )" % inst.meta["level_values"][k - 1])
                if solve(Instance(prompt=probe, steps=[], states=[], answer="", depth=k)) != inst.answer:
                    r_fails.append("k=%d tail does not reconstruct at d=%d s=%d: %r" % (k, d, s, red))
                if len(red) >= len(inst.prompt):
                    r_fails.append("k=%d removed nothing at d=%d s=%d" % (k, d, s))
    # the fixed/published material redacts too
    try:
        e = published_exemplar()
        if redact_prompt(e, 0) != e.prompt:
            r_fails.append("exemplar k=0 not identity")
        if redact_prompt(e, 1) != "not ( ( %s ) ) is" % REDACTION_PLACEHOLDER:
            r_fails.append("exemplar k=1 wrong: %r" % redact_prompt(e, 1))
        if redact_prompt(e, e.depth) != REDACTION_PLACEHOLDER + " is":
            r_fails.append("exemplar k=depth wrong: %r" % redact_prompt(e, e.depth))
    except Exception as exc:
        r_fails.append("exemplar redaction raised %r" % (exc,))
    try:
        for inst in bbh_instances():
            if redact_prompt(inst, 0) != inst.prompt:
                r_fails.append("bbh k=0 not identity: %r" % inst.prompt)
            if redact_prompt(inst, inst.depth) != REDACTION_PLACEHOLDER + " is":
                r_fails.append("bbh k=depth wrong: %r" % inst.prompt)
    except OSError:
        pass
    if not REDACTION_MEANINGFUL:
        r_fails.append("REDACTION_MEANINGFUL should be True for this task")
    print("[7/7] redaction: 20 instances x k in {0, 1, depth//2, depth} "
          "(+ exemplar, + vendored set) ... %s"
          % ("OK" if not r_fails else "%d FAILURES" % len(r_fails)))
    fails += ["redact: %r" % (f,) for f in r_fails]

    if fails:
        print("\nFAILED (%d):" % len(fails))
        for f in fails[:25]:
            print("  " + str(f))
        return 1
    print("\nSELFTEST PASSED")
    return 0


def _demo():
    lines = []
    lines.append("# %s -- rendered examples (python3 task.py --demo)" % SLUG)
    lines.append("# format: BBH CoT (Suzgun et al. 2022). depth = named sub-expression")
    lines.append("# resolutions on the critical path == len(steps).")
    lines.append("# ANSWER_FORMAT: %s" % ANSWER_FORMAT)
    lines.append("# DEPTHS: %s" % DEPTHS)
    lines.append("")
    lines.append("=" * 78)
    lines.append("PUBLISHED EXEMPLAR (verbatim, exemplars(3, 0)[0])")
    lines.append("=" * 78)
    e = published_exemplar()
    lines.append("PROMPT: " + e.prompt)
    lines.append("GOLD:")
    lines.append(format_cot(e))
    lines.append("STATES: " + " | ".join(e.states))
    lines.append("")
    for d in (DEPTHS[0], DEPTHS[-1]):
        lines.append("=" * 78)
        lines.append("DEPTH %d" % d)
        lines.append("=" * 78)
        for s in range(3):
            inst = generate(d, s)
            lines.append("--- depth=%d seed=%d (operators=%d, elementary critical path=%d) ---"
                         % (d, s, inst.meta["n_operators"], inst.meta["elementary_depth"]))
            lines.append("PROMPT: " + inst.prompt)
            lines.append("GOLD:")
            lines.append(format_cot(inst))
            lines.append("STATES: " + " | ".join(inst.states))
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    if args.demo:
        print(_demo())
        return
    inst = generate(args.depth, args.seed)
    print("PROMPT: " + inst.prompt)
    print(format_cot(inst))
    print("STATES: " + " | ".join(inst.states))


if __name__ == "__main__":
    main()
