"""random_lookup_table -- iterated application of random lookup tables over a small alphabet.

Primary source (see desk.json / published_trace.txt):
    Ramesh, Bothwell, Ravikumar et al., "Compositional Capabilities of Autoregressive
    Transformers: A Study on Synthetic, Interpretable Tasks", arXiv:2311.12997, ICML 2024.
    Repo: https://github.com/rahul13ramesh/compositional_capabilities
          commit 54256841ef5f3af3a39d47288d071f7d04b6da5e (MIT), vendored under vendor/.

The published document (step-by-step / direct=False setting) is a flat token stream

    S T0_3 T1_0 T2_0 T3_3 T4_2  X5X9X1X5X7X8  X9X7X5X9X3X2  ...  X8X7X4X8X6X1

i.e. a start token, one task token T{d}_{i} per composition step naming which member of that
step's function pool was applied, the initial length-seq_len vector of alphabet tokens, and then
one vector per step.  This module reproduces that vocabulary and that per-step emission; see
README.md "Format decision" for the three deviations (plain ASCII digits instead of the repo
pretty-printer's Unicode subscripts, one step per line with its task token, and the lookup
tables written into the prompt, which the paper's from-scratch models learned during training).

Pure python + stdlib.  No network.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------
# interface constants
# --------------------------------------------------------------------------------------------

ANSWER_FORMAT = "a single alphabet symbol written like X7"

# AMENDMENT 4: the prompt carries an initial state (the start symbol) and a per-step operator
# list (the document's task tokens) that are separable from the static lookup tables, so prompt
# blinding is meaningful here.
REDACTION_MEANINGFUL = True

# 2 is trivially short; 4 and 8 bracket the paper's own function.depth=5; a 7B should lose the
# no-CoT condition by ~8 and degrade under CoT somewhere in 24-48.  See README "Depth semantics".
DEPTHS = [2, 4, 8, 16, 24, 32, 48]

KNOBS = {
    "n_alphabets": (10, "alphabet size N (repo conf.yaml n_alphabets=10); state space per step"),
    "n_functions": (3, "size of the lookup-table pool (repo conf.yaml function.n_functions=3)"),
    "bijective": (
        True,
        "True = tables are permutations of the alphabet, exactly as published "
        "(np.random.permutation).  False = arbitrary [N]->[N] maps; see README caveats, the "
        "non-bijective task collapses at depth and must not be the default.",
    ),
    "fresh_tables_per_step": (
        False,
        "False = one shared pool reused at every step (repo function.repeat=True).  True = a "
        "fresh pool per step (repo default function.repeat=False); prompt then grows with depth.",
    ),
    "include_identity": (
        False,
        "True = choice 0 of each pool is the identity, as published (T{d}_0); such steps are "
        "no-ops and dilute the depth knob, so off by default.",
    ),
    "seq_len": (
        1,
        "number of alphabet positions carried in parallel (repo conf.yaml seq_len=6).  1 is this "
        "task's single-tracked-symbol definition and the setting desk.json's answer_space=10 and "
        "state_bits=log2(10) describe.",
    ),
    "format": (
        "published",
        '"published" = the paper\'s flat token stream, one line per step (`T2_1 X6`); the '
        'default, so the as-published rows stay reproducible.  "ergonomic" = AMENDMENT 5: a '
        "plain-language prompt and a self-contained per-step line (`step 2: apply F3 to X7. "
        "F3: X7 -> X0. Now at X0.`).  The instance distribution, `answer`, `states`, `depth`, "
        "`check` and `solve` are identical across the two; only the wording of `prompt` and "
        "`steps` differs.",
    ),
}

_DEFAULTS = {k: v[0] for k, v in KNOBS.items()}


# --------------------------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------------------------


@dataclass
class Instance:
    prompt: str
    steps: list[str]
    states: list[str]
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _resolve(knobs: dict) -> dict:
    unknown = set(knobs) - set(_DEFAULTS)
    if unknown:
        raise TypeError("unknown knob(s): %s" % sorted(unknown))
    cfg = dict(_DEFAULTS)
    cfg.update(knobs)
    if cfg["n_alphabets"] < 2:
        raise ValueError("n_alphabets must be >= 2")
    if cfg["n_functions"] < 1:
        raise ValueError("n_functions must be >= 1")
    if cfg["seq_len"] < 1:
        raise ValueError("seq_len must be >= 1")
    if cfg["format"] not in ("published", "ergonomic"):
        raise ValueError('format must be "published" or "ergonomic", not %r' % (cfg["format"],))
    if cfg["seq_len"] > 1 and cfg["n_alphabets"] > 10:
        # published vectors concatenate symbols with no separator (X9X7X5...), which is only
        # unambiguously parseable while every symbol index is a single digit.
        raise ValueError("seq_len > 1 requires n_alphabets <= 10 (published vectors are unseparated)")
    return cfg


# AMENDMENT 5: `format` is a rendering choice, so it is excluded from the rng key -- the same
# (depth, seed, other knobs) must draw the same tables, choices and start symbol in either format.
_RNG_IRRELEVANT = frozenset({"format"})


def _rng(depth: int, seed: int, cfg: dict) -> random.Random:
    key = "|".join([str(depth), str(seed)]
                   + ["%s=%r" % (k, cfg[k]) for k in sorted(cfg) if k not in _RNG_IRRELEVANT])
    return random.Random(key)


def _sym(i: int) -> str:
    return "X%d" % i


def _vec(v: list[int]) -> str:
    return "".join(_sym(i) for i in v)


def _task_token(d: int, i: int) -> str:
    return "T%d_%d" % (d, i)


def _make_table(rng: random.Random, n: int, bijective: bool) -> list[int]:
    if bijective:
        t = list(range(n))
        rng.shuffle(t)
        return t
    return [rng.randrange(n) for _ in range(n)]


def _make_pool(rng: random.Random, cfg: dict) -> list[list[int]]:
    """One pool: index 0 is the identity iff include_identity, then n_functions real tables."""
    n = cfg["n_alphabets"]
    pool: list[list[int]] = []
    if cfg["include_identity"]:
        pool.append(list(range(n)))
    for _ in range(cfg["n_functions"]):
        pool.append(_make_table(rng, n, cfg["bijective"]))
    return pool


def _has_common_fixed_point(pool: list[list[int]], n: int, skip_identity: bool) -> bool:
    real = pool[1:] if skip_identity else pool
    return any(all(t[x] == x for t in real) for x in range(n))


def _table_line(name: str, table: list[int]) -> str:
    return "%s: %s" % (name, " ".join("%s->%s" % (_sym(i), _sym(o)) for i, o in enumerate(table)))


def _render_prompt(cfg: dict, pools: list[list[list[int]]], choices: list[int], start: list[int]) -> str:
    """pools is one pool per step (shared mode passes the same object `depth` times)."""
    n = cfg["n_alphabets"]
    depth = len(choices)
    # `first` is the choice index of pools[*][0]; the identity is never written out as a table.
    first = 0 if cfg["include_identity"] else 1
    real = lambda pool: [(i + first, t) for i, t in enumerate(pool) if i + first != 0]
    lines = ["Alphabet: " + " ".join(_sym(i) for i in range(n)), ""]

    if cfg["fresh_tables_per_step"]:
        lines.append("Lookup tables (a fresh set of tables is used at every step):")
        for d in range(depth):
            for idx, table in real(pools[d]):
                lines.append(_table_line(_task_token(d, idx), table))
    else:
        lines.append("Lookup tables (the same tables are used at every step):")
        for idx, table in real(pools[0]):
            lines.append(_table_line("F%d" % idx, table))

    lines.append("")
    header = " ".join(_task_token(d, c) for d, c in enumerate(choices))
    lines.append("Document: S %s %s" % (header, _vec(start)))
    lines.append("")

    if cfg["fresh_tables_per_step"]:
        naming = 'The task token T{d}_{i} names the lookup table applied at step d.'
    else:
        naming = 'The task token T{d}_{i} means "at step d, apply table F{i}".'
    lines.append(naming)
    if cfg["include_identity"]:
        lines.append("A task token T{d}_0 is the identity: it leaves the symbol unchanged.")

    if cfg["seq_len"] == 1:
        lines.append(
            "Starting from the symbol %s, apply the tables named by T0_*, T1_*, ... in that "
            "order, each to the result of the previous step." % _sym(start[0])
        )
        lines.append("What is the symbol after the last step?")
    else:
        lines.append(
            "Starting from the vector %s, apply the tables named by T0_*, T1_*, ... in that "
            "order, each applied to every position of the vector in parallel." % _vec(start)
        )
        lines.append("What is the vector after the last step?")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# AMENDMENT 5: the "ergonomic" rendering.  Same instance, same states, same answer; a prompt in
# plain language and a step line that restates the operator, its relevant table entry and the new
# state, so a prompted (never fine-tuned) model can compute in it from a handful of exemplars.
# --------------------------------------------------------------------------------------------


def _table_name(cfg: dict, d: int, idx: int) -> str:
    """The name a table is written under: shared pool -> F{i}, fresh pool per step -> T{d}_{i}."""
    return _task_token(d, idx) if cfg["fresh_tables_per_step"] else "F%d" % idx


def _table_line_ergonomic(name: str, table: list[int]) -> str:
    return "%s: %s" % (name, ", ".join("%s -> %s" % (_sym(i), _sym(o)) for i, o in enumerate(table)))


def _order_line(step_no: int, name: str) -> str:
    return "step %d: %s" % (step_no, name)


def _render_prompt_ergonomic(cfg: dict, pools: list[list[list[int]]], choices: list[int],
                             start: list[int]) -> str:
    n = cfg["n_alphabets"]
    depth = len(choices)
    first = 0 if cfg["include_identity"] else 1
    real = lambda pool: [(i + first, t) for i, t in enumerate(pool) if i + first != 0]

    lines = ["There are %d symbols: %s." % (n, ", ".join(_sym(i) for i in range(n))), ""]

    if cfg["fresh_tables_per_step"]:
        lines.append("Every step has its own lookup tables.  A lookup table says what each symbol "
                     "turns into.  Here are all of them:")
        lines.append("")
        for d in range(depth):
            for idx, table in real(pools[d]):
                lines.append(_table_line_ergonomic(_table_name(cfg, d, idx), table))
    else:
        names = [_table_name(cfg, 0, idx) for idx, _ in real(pools[0])]
        lines.append("There are %d lookup tables, and the same ones are used at every step.  A "
                     "lookup table says what each symbol turns into:" % len(names))
        lines.append("")
        for idx, table in real(pools[0]):
            lines.append(_table_line_ergonomic(_table_name(cfg, 0, idx), table))
    if cfg["include_identity"]:
        lines.append("")
        lines.append("A table whose number is 0 (%s) is the identity table: it leaves the symbol "
                     "unchanged." % _table_name(cfg, 0, 0))

    lines.append("")
    if cfg["seq_len"] == 1:
        lines.append("Start at %s." % _sym(start[0]))
    else:
        lines.append("Start at %s (that is %d symbols side by side; every table is applied to each "
                     "of them)." % (_vec(start), cfg["seq_len"]))
    lines.append("")
    lines.append("Apply the tables in this order, one table per step, each one to what you are "
                 "holding after the previous step:")
    for i, c in enumerate(choices):
        lines.append(_order_line(i + 1, _table_name(cfg, i, c)))
    lines.append("")
    lines.append("What do you end at after step %d?" % depth)
    return "\n".join(lines)


def _step_line_ergonomic(cfg: dict, step_no: int, name: str, table: list[int] | None,
                         before: list[int], after: list[int]) -> str:
    """`step 2: apply F3 to X7. F3: X7 -> X0. Now at X0.`  (table None = the identity.)"""
    if table is None:
        return "step %d: apply %s to %s. %s is the identity. Now at %s." % (
            step_no, name, _vec(before), name, _vec(after))
    seen: list[int] = []
    for x in before:
        if x not in seen:
            seen.append(x)
    detail = ", ".join("%s -> %s" % (_sym(x), _sym(table[x])) for x in seen)
    return "step %d: apply %s to %s. %s: %s. Now at %s." % (
        step_no, name, _vec(before), name, detail, _vec(after))


def _build(cfg: dict, pools: list[list[list[int]]], choices: list[int], start: list[int],
           meta_extra: dict | None = None) -> Instance:
    base = 0 if cfg["include_identity"] else 1
    ergonomic = cfg["format"] == "ergonomic"
    cur = list(start)
    steps: list[str] = []
    states: list[str] = []
    for d, c in enumerate(choices):
        table = pools[d][c - base]
        prev, cur = cur, [table[x] for x in cur]
        states.append(_vec(cur))
        if ergonomic:
            is_id = cfg["include_identity"] and c == 0
            steps.append(_step_line_ergonomic(
                cfg, d + 1, _table_name(cfg, d, c), None if is_id else table, prev, cur))
        else:
            steps.append("%s %s" % (_task_token(d, c), _vec(cur)))
    meta = {
        "n_alphabets": cfg["n_alphabets"],
        "n_functions": cfg["n_functions"],
        "bijective": cfg["bijective"],
        "fresh_tables_per_step": cfg["fresh_tables_per_step"],
        "include_identity": cfg["include_identity"],
        "seq_len": cfg["seq_len"],
        "format": cfg["format"],
        "start": _vec(start),
        "choices": list(choices),
    }
    if meta_extra:
        meta.update(meta_extra)
    render = _render_prompt_ergonomic if ergonomic else _render_prompt
    return Instance(
        prompt=render(cfg, pools, choices, start),
        steps=steps,
        states=states,
        answer=states[-1] if states else _vec(start),
        depth=len(choices),
        meta=meta,
    )


# --------------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------------


def generate(depth: int, seed: int, **knobs) -> Instance:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    cfg = _resolve(knobs)
    rng = _rng(depth, seed, cfg)
    n = cfg["n_alphabets"]
    base = 0 if cfg["include_identity"] else 1

    if cfg["fresh_tables_per_step"]:
        pools = [_make_pool(rng, cfg) for _ in range(depth)]
    else:
        # Reject a pool with a symbol that every real table fixes: the walk would be absorbed
        # there and the remaining steps would stop mattering.  Vanishingly rare for permutations,
        # ~1% for arbitrary maps at N=10, n_functions=3.
        for _ in range(64):
            pool = _make_pool(rng, cfg)
            if not _has_common_fixed_point(pool, n, cfg["include_identity"]):
                break
        pools = [pool] * depth

    n_choices = cfg["n_functions"] + (1 if cfg["include_identity"] else 0)
    choices = [base + rng.randrange(n_choices) for _ in range(depth)]
    start = [rng.randrange(n) for _ in range(cfg["seq_len"])]
    return _build(cfg, pools, choices, start, {"seed": seed})


# --------------------------------------------------------------------------------------------
# solve -- independent reference solver.  Parses the rendered prompt and simulates; it shares no
# code path with generate() beyond the string helpers, so a rendering bug fails the selftest.
# --------------------------------------------------------------------------------------------

_TABLE_RE = re.compile(r"^(F\d+|T\d+_\d+):\s+(X\d.*)$")
_PAIR_RE = re.compile(r"X(\d+)\s*->\s*X(\d+)")
_DOC_RE = re.compile(r"^Document:\s+S\s+(.*?)\s+((?:X\d)+|X\d+)\s*$")
# ergonomic prompt (AMENDMENT 5)
_START_AT_RE = re.compile(r"^Start at ((?:X\d+)+)")
_ORDER_RE = re.compile(r"^step (\d+): (F\d+|T\d+_\d+)\s*$")


def _parse_state(txt: str, n: int) -> list[int]:
    """Published vectors concatenate symbols with no separator, which is unambiguous exactly while
    every symbol index is one digit; an alphabet larger than 10 therefore implies seq_len == 1."""
    if n <= 10:
        return [int(x) for x in re.findall(r"X(\d)", txt)]
    return [int(txt[1:])]


def _read_tables(prompt: str) -> tuple[dict[str, dict[int, int]], int]:
    tables: dict[str, dict[int, int]] = {}
    n = 0
    for line in prompt.splitlines():
        m = _TABLE_RE.match(line)
        if m:
            tbl = {int(a): int(b) for a, b in _PAIR_RE.findall(m.group(2))}
            if not tbl:
                continue
            n = max(n, max(tbl) + 1)
            tables[m.group(1)] = tbl
    return tables, n


def solve(inst: Instance) -> str:
    """Reference solver.  Reads the rendered prompt back and simulates; it shares no code path
    with generate() beyond the string helpers, and it returns the same answer for either
    `format` because the two renderings describe the same instance."""
    if any(line.startswith("Document: S ") for line in inst.prompt.splitlines()):
        return _solve_published(inst)
    return _solve_ergonomic(inst)


def _solve_ergonomic(inst: Instance) -> str:
    tables, n = _read_tables(inst.prompt)
    start = None
    order: list[tuple[int, str]] = []
    for line in inst.prompt.splitlines():
        m = _START_AT_RE.match(line)
        if m:
            start = _parse_state(m.group(1), n)
            continue
        m = _ORDER_RE.match(line)
        if m:
            order.append((int(m.group(1)), m.group(2)))
    if start is None:
        raise ValueError("no 'Start at ...' line in prompt")
    if [i for i, _ in order] != list(range(1, len(order) + 1)):
        raise ValueError("step order list is not 1..depth in sequence")

    cur = list(start)
    for _, name in order:
        if name in tables:
            tbl = tables[name]
        elif name.rsplit("_", 1)[-1].lstrip("FT") == "0":
            tbl = {x: x for x in range(n)}     # choice 0 = identity, never written out as a table
        else:
            raise ValueError("no table named %s" % name)
        cur = [tbl[x] for x in cur]
    return "".join("X%d" % x for x in cur)


def _solve_published(inst: Instance) -> str:
    tables, n = _read_tables(inst.prompt)
    doc_line = None
    for line in inst.prompt.splitlines():
        m = _DOC_RE.match(line)
        if m:
            doc_line = m
    if doc_line is None:
        raise ValueError("no Document: line in prompt")

    header = doc_line.group(1).split()
    start = _parse_state(doc_line.group(2), n)

    cur = list(start)
    for d, tok in enumerate(header):
        dd, ii = tok[1:].split("_")
        if int(dd) != d:
            raise ValueError("task token %s out of position %d" % (tok, d))
        if tok in tables:                      # fresh-tables-per-step naming
            tbl = tables[tok]
        elif ("F" + ii) in tables:             # shared-pool naming
            tbl = tables["F" + ii]
        elif ii == "0":                        # identity, never written out
            tbl = {x: x for x in range(n)}
        else:
            raise ValueError("no table for task token %s" % tok)
        cur = [tbl[x] for x in cur]
    return "".join("X%d" % x for x in cur)


# --------------------------------------------------------------------------------------------
# check / format_cot / step_spans
# --------------------------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Canonicalize an answer fragment to 'X9' / 'X9X7X5'.  Accepts bare digits and spacing."""
    t = text.strip().rstrip(".").strip()
    syms = re.findall(r"[Xx](\d+)", t)
    if syms:
        return "".join("X%d" % int(s) for s in syms)
    digits = re.findall(r"\d+", t)
    if digits:
        return "".join("X%d" % int(d) for d in digits)
    return ""


def check(inst: Instance, completion: str) -> bool:
    lines = completion.splitlines()
    payload = None
    for line in reversed(lines):
        if re.match(r"^\s*Answer\s*:", line, flags=re.I):
            payload = re.split(r"[Aa]nswer\s*:", line, maxsplit=1)[1]
            break
    if payload is None:
        for line in reversed(lines):
            if line.strip():
                payload = line
                break
    if payload is None:
        return False
    return _normalize(payload) == _normalize(inst.answer)


def format_cot(inst: Instance) -> str:
    return "\n".join(inst.steps) + "\nAnswer: " + inst.answer


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    text = format_cot(inst)
    spans = []
    pos = 0
    for s in inst.steps:
        i = text.index(s, pos)
        spans.append((i, i + len(s)))
        pos = i + len(s)
    return spans


# --------------------------------------------------------------------------------------------
# redact_prompt -- prompt blinding (AMENDMENT 4)
# --------------------------------------------------------------------------------------------

_REDACTED = "[…]"
_START_LINE_RE = re.compile(r"^(Starting from the (?:symbol|vector) )((?:X\d+)+)")


def redact_prompt(inst: Instance, k: int) -> str:
    """Return inst.prompt with everything steps 1..k consumed replaced by ``[…]``.

    Removed: the start symbol/vector -- the initial state, consumed by step 1 -- wherever it is
    stated (the ``Document:`` line and the ``Starting from the ...`` sentence), and the first k
    task tokens of the document header, which say which table each of steps 1..k applied.  One
    ``[…]`` per removed item.

    Kept: the lookup tables, which are static material needed by every step including k+1..depth
    (with ``fresh_tables_per_step`` the table block is keyed T{d}_{i}, but it lists every pool
    member of every step and so discloses no choice; the choices live only in the header), the
    task tokens for steps k+1..depth, the alphabet, the task-token gloss and the question.

    k == 0 returns the prompt unchanged; k == inst.depth leaves only the question answerable from
    a carried state.

    Both formats are supported (AMENDMENT 5).  In the ergonomic rendering the same two things are
    removed: the ``Start at ...`` symbol, and the table name on each of the first k lines of the
    ``step i: <table>`` order list.
    """
    if k < 0 or k > inst.depth:
        raise ValueError("k must be in [0, %d]" % inst.depth)
    if k == 0:
        return inst.prompt
    if any(line.startswith("Document: S ") for line in inst.prompt.splitlines()):
        return _redact_published(inst, k)
    return _redact_ergonomic(inst, k)


def _redact_ergonomic(inst: Instance, k: int) -> str:
    out: list[str] = []
    seen_start = False
    seen_order = 0
    for line in inst.prompt.splitlines():
        m = _START_AT_RE.match(line)
        if m:
            out.append("Start at " + _REDACTED + line[m.end(1):])
            seen_start = True
            continue
        m = _ORDER_RE.match(line)
        if m:
            seen_order += 1
            i = int(m.group(1))
            out.append(_order_line(i, _REDACTED) if i <= k else line)
            continue
        out.append(line)
    if not seen_start:
        raise ValueError("no 'Start at ...' line in prompt")
    if seen_order != inst.depth:
        raise ValueError("order list has %d entries, expected %d" % (seen_order, inst.depth))
    return "\n".join(out)


def _redact_published(inst: Instance, k: int) -> str:
    out: list[str] = []
    seen_doc = False
    for line in inst.prompt.splitlines():
        m = _DOC_RE.match(line)
        if m:
            header = m.group(1).split()
            header = [_REDACTED] * min(k, len(header)) + header[k:]
            out.append("Document: S %s %s" % (" ".join(header), _REDACTED))
            seen_doc = True
            continue
        out.append(_START_LINE_RE.sub(lambda mm: mm.group(1) + _REDACTED, line))
    if not seen_doc:
        raise ValueError("no Document: line in prompt")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# corrupt_step -- mistake propagation (AMENDMENT 6)
# --------------------------------------------------------------------------------------------

# The two renderings of a step line.  Both are anchored, and both keep the step's ACTION (the
# operator token / name and the state going in) in their own groups, so the corruption can only
# touch the reported result -- and so `--selftest` can assert exactly that.
_PUB_STEP_RE = re.compile(r"^(T\d+_\d+) ((?:X\d+)+)$")
_ERGO_STEP_RE = re.compile(
    r"^step (\d+): apply (F\d+|T\d+_\d+) to ((?:X\d+)+)\. (.+?)\. Now at ((?:X\d+)+)\.$")


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """Rewrite step k (1-based) so it reports a plausible WRONG state, and return that state.

    Returns ``(step_text, corrupted_state)``.  The corruption is one misread table entry: a
    distinct input symbol ``Xx`` of the state going into step k is picked, and the output the
    step reports for it is moved to a different symbol of the same alphabet.  With the default
    ``seq_len=1`` that is simply "the step names the wrong result symbol"; with ``seq_len > 1``
    every position holding ``Xx`` moves together, which is what a single wrong lookup actually
    does to a vector and what keeps the ergonomic line internally consistent.

    The step's action text is untouched in both formats -- the published task token ``T{d}_{i}``,
    and the ergonomic ``step k: apply F3 to X7.`` prefix (and the operator's name in the table
    entry) all survive verbatim; only the reported result moves.  In the ergonomic rendering the
    state is written twice, so both places are rewritten together: the table entry ``X7 -> X0``
    and the closing ``Now at X0.``  (An ``include_identity`` no-op step states no table entry --
    ``F0 is the identity.`` is action text, not a result -- so only ``Now at`` moves there.)

    Deterministic in ``(inst, k, seed)``.  The rng key deliberately excludes `format`, exactly as
    `generate`'s does (`_RNG_IRRELEVANT`): the same instance corrupts to the same wrong state in
    either rendering, so a published and an ergonomic mistake run are comparable.
    """
    if not 1 <= k <= inst.depth:
        raise ValueError("k must be in [1, %d], not %r" % (inst.depth, k))

    n = inst.meta["n_alphabets"]
    before = _parse_state(inst.meta["start"] if k == 1 else inst.states[k - 2], n)
    after = _parse_state(inst.states[k - 1], n)

    rng = random.Random("corrupt|%d|%d|%d|%s|%s" % (
        seed, k, n, inst.meta["start"], inst.meta["choices"]))
    seen: list[int] = []
    for x in before:
        if x not in seen:
            seen.append(x)
    x = seen[rng.randrange(len(seen))]
    old = after[before.index(x)]
    new = (old + 1 + rng.randrange(n - 1)) % n          # n >= 2, so new != old
    corrupted = _vec([new if b == x else a for b, a in zip(before, after)])

    step = inst.steps[k - 1]
    if inst.meta["format"] == "ergonomic":
        m = _ERGO_STEP_RE.match(step)
        if not m:
            raise ValueError("step %d is not an ergonomic step line: %r" % (k, step))
        detail = re.sub(r"(?<![0-9])X%d -> X%d(?![0-9])" % (x, old),
                        "X%d -> X%d" % (x, new), m.group(4), count=1)
        return ("step %s: apply %s to %s. %s. Now at %s." % (
            m.group(1), m.group(2), m.group(3), detail, corrupted), corrupted)

    m = _PUB_STEP_RE.match(step)
    if not m:
        raise ValueError("step %d is not a published step line: %r" % (k, step))
    return ("%s %s" % (m.group(1), corrupted), corrupted)


# --------------------------------------------------------------------------------------------
# exemplars
# --------------------------------------------------------------------------------------------

# published_trace.txt, subscripts transcribed to digits.  The document is
#   S T0_3 T1_0 T2_0 T3_3 T4_2  X5X9X1X5X7X8  X9X7X5X9X3X2 (x3)  X7X9X5X7X0X3  X8X7X4X8X6X1
_PUB_CHOICES = [3, 0, 0, 3, 2]
_PUB_VECTORS = [
    [5, 9, 1, 5, 7, 8],   # initial
    [9, 7, 5, 9, 3, 2],   # after T0_3
    [9, 7, 5, 9, 3, 2],   # after T1_0 (identity)
    [9, 7, 5, 9, 3, 2],   # after T2_0 (identity)
    [7, 9, 5, 7, 0, 3],   # after T3_3
    [8, 7, 4, 8, 6, 1],   # after T4_2
]


def _complete_bijection(partial: dict[int, int], n: int, rng: random.Random) -> list[int]:
    used = set(partial.values())
    free_in = [i for i in range(n) if i not in partial]
    free_out = [o for o in range(n) if o not in used]
    rng.shuffle(free_out)
    table = [0] * n
    for i, o in partial.items():
        table[i] = o
    for i, o in zip(free_in, free_out):
        table[i] = o
    return table


def _published_exemplar() -> Instance:
    """Reconstruct the published depth-5 step-by-step document from published_trace.txt.

    The published document applies the same function elementwise to seq_len=6 INDEPENDENT
    positions, so any single column of it is itself a verbatim published single-symbol trace of
    the same instance.  We take column 0 (X5 -> X9 -> X9 -> X9 -> X7 -> X8): depth, the task
    tokens T0_3 T1_0 T2_0 T3_3 T4_2, the start symbol and every intermediate state are verbatim.
    All six columns are used as constraints on the applied tables; the entries the trace does not
    determine, and the pool members the trace never exercises, are completed deterministically
    (seeded) so that a full lookup table can be written into the prompt.  Disclosed in meta.
    """
    n = 10
    cfg = _resolve(dict(n_alphabets=n, n_functions=3, bijective=True,
                        fresh_tables_per_step=True, include_identity=True, seq_len=1))
    rng = random.Random("random_lookup_table/published-exemplar/v1")

    pools: list[list[list[int]]] = []
    for d, c in enumerate(_PUB_CHOICES):
        pool = [list(range(n))]  # index 0 = identity
        constrained: dict[int, int] = {}
        if c != 0:
            for a, b in zip(_PUB_VECTORS[d], _PUB_VECTORS[d + 1]):
                if a in constrained and constrained[a] != b:
                    raise AssertionError("published trace is not a function at depth %d" % d)
                constrained[a] = b
        for i in range(1, 4):
            pool.append(_complete_bijection(constrained if i == c else {}, n, rng))
        pools.append(pool)

    inst = _build(
        cfg, pools, list(_PUB_CHOICES), [_PUB_VECTORS[0][0]],
        {
            "published_exemplar": True,
            "trace_source": "published_trace.txt (Ramesh et al. 2024, repo commit 5425684), "
                            "column 0 of the seq_len=6 step-by-step document",
            "reconstruction": "task tokens, start symbol and all 5 states verbatim; table entries "
                              "not determined by the trace's 6 columns, and unexercised pool "
                              "members, filled deterministically",
            "published_vectors": [_vec(v) for v in _PUB_VECTORS],
        },
    )
    assert inst.states == ["X9", "X9", "X9", "X7", "X8"], inst.states
    return inst


def exemplars(k: int, seed: int, **knobs) -> list[Instance]:
    """Few-shot exemplars, rendered in the same `format` as the instance being solved.

    AMENDMENT 3 wants the verbatim published exemplar first; that reconstruction only exists for
    the published token stream, so under `format="ergonomic"` (AMENDMENT 5) it is skipped and all
    k exemplars are generated instead.  All other knobs are forwarded to `generate`.
    """
    if k <= 0:
        return []
    cfg = _resolve(knobs)                       # validates knob names / values up front
    out = [_published_exemplar()] if cfg["format"] == "published" else []
    d = 4  # modest depth, brackets the paper's own function.depth=5
    i = 0
    while len(out) < k:
        out.append(generate(d, 10_000_000 + seed * 1000 + i, **knobs))
        i += 1
    return out[:k]


# --------------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------------


def _selftest() -> int:
    rng = random.Random(20260916)
    failures = []
    for trial in range(200):
        depth = rng.choice([1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48])
        seed = rng.randrange(10**6)
        knobs = {}
        if trial % 7 == 1:
            knobs["n_alphabets"] = rng.choice([5, 8, 12, 16])
        if trial % 7 == 2:
            knobs["n_functions"] = rng.choice([1, 2, 5])
        if trial % 7 == 3:
            knobs["bijective"] = False
        if trial % 7 == 4:
            knobs["fresh_tables_per_step"] = True
        if trial % 7 == 5:
            knobs["include_identity"] = True
        if trial % 7 == 6:
            knobs["seq_len"] = rng.choice([2, 6])
            knobs["n_alphabets"] = 10
        # AMENDMENT 5: every trial is run in one format and cross-checked against the other.
        knobs["format"] = "ergonomic" if trial % 2 else "published"

        inst = generate(depth, seed, **knobs)

        got = solve(inst)
        if got != inst.answer:
            failures.append("solve mismatch d=%d s=%d %r: %r != %r" % (depth, seed, knobs, got, inst.answer))
        if not check(inst, format_cot(inst)):
            failures.append("check(gold) false d=%d s=%d %r" % (depth, seed, knobs))
        n = inst.meta["n_alphabets"]
        wrong = "X%d" % ((int(re.findall(r"X(\d+)", inst.answer)[-1]) + 1) % n)
        wrong_full = inst.answer[: len(inst.answer) - len(wrong)] + wrong if inst.meta["seq_len"] > 1 else wrong
        if check(inst, "Answer: " + wrong_full):
            failures.append("check(wrong) true d=%d s=%d %r (%s vs %s)" % (depth, seed, knobs, wrong_full, inst.answer))
        if check(generate(depth, seed, **dict(knobs, format="ergonomic")), "Answer: " + wrong_full) \
                != check(generate(depth, seed, **dict(knobs, format="published")), "Answer: " + wrong_full):
            failures.append("check() disagrees across formats d=%d s=%d %r" % (depth, seed, knobs))
        if inst != generate(depth, seed, **knobs):
            failures.append("nondeterministic d=%d s=%d %r" % (depth, seed, knobs))
        if not (len(inst.steps) == len(inst.states) == depth):
            failures.append("length mismatch d=%d s=%d %r" % (depth, seed, knobs))
        if step_spans(inst)[-1][1] >= len(format_cot(inst)):
            failures.append("bad step_spans d=%d s=%d" % (depth, seed))

        # ---- AMENDMENT 5: the two formats are the same instance, differently written -------
        other = dict(knobs)
        other["format"] = "published" if knobs["format"] == "ergonomic" else "ergonomic"
        alt = generate(depth, seed, **other)
        tag = "d=%d s=%d %r" % (depth, seed, knobs)
        if (alt.answer, alt.states, alt.depth) != (inst.answer, inst.states, inst.depth):
            failures.append("format changes answer/states/depth %s" % tag)
        if alt.meta["start"] != inst.meta["start"] or alt.meta["choices"] != inst.meta["choices"]:
            failures.append("format changes the instance distribution %s" % tag)
        if solve(alt) != inst.answer:
            failures.append("solve() differs across formats %s" % tag)
        if not check(alt, format_cot(alt)):
            failures.append("check(gold) false in the other format %s" % tag)
        if alt.prompt == inst.prompt or alt.steps == inst.steps:
            failures.append("the two formats rendered identically %s" % tag)
        if len(alt.steps) != depth or len(alt.states) != depth:
            failures.append("length mismatch in the other format %s" % tag)
        if alt.meta["format"] != other["format"] or inst.meta["format"] != knobs["format"]:
            failures.append("meta['format'] not recorded %s" % tag)

    ex = exemplars(3, 0)
    if len(ex) != 3:
        failures.append("exemplars(3,0) returned %d" % len(ex))
    if not ex[0].meta.get("published_exemplar"):
        failures.append("exemplars(3,0)[0] is not the published exemplar")
    for e in ex:
        if solve(e) != e.answer:
            failures.append("exemplar solve mismatch: %r != %r" % (solve(e), e.answer))
        if not check(e, format_cot(e)):
            failures.append("exemplar check(gold) false")
        format_cot(e)

    # AMENDMENT 5: exemplars() forwards `format` (and every other knob) to generate()
    ex_e = exemplars(3, 0, format="ergonomic")
    if len(ex_e) != 3:
        failures.append("exemplars(3,0,ergonomic) returned %d" % len(ex_e))
    for e in ex_e:
        if e.meta.get("published_exemplar"):
            failures.append("ergonomic exemplars must not include the published reconstruction")
        if e.meta["format"] != "ergonomic":
            failures.append("exemplars() did not forward format=%r" % e.meta["format"])
        if not e.prompt.startswith("There are ") or "Start at " not in e.prompt:
            failures.append("ergonomic exemplar is not in the ergonomic format")
        if solve(e) != e.answer or not check(e, format_cot(e)):
            failures.append("ergonomic exemplar solve/check failed")
    if [e.meta["n_alphabets"] for e in exemplars(2, 0, format="ergonomic", n_alphabets=5)] != [5, 5]:
        failures.append("exemplars() did not forward a non-format knob")
    for bad, exc in ((dict(format="bogus"), ValueError), (dict(nope=1), TypeError)):
        try:
            exemplars(2, 0, **bad)
        except exc:
            pass
        else:
            failures.append("exemplars(**%r) should have raised %s" % (bad, exc.__name__))

    # tolerant parsing
    probe = generate(4, 1)
    tol = [
        ("Answer: %s" % probe.answer, True),
        ("Answer: %s." % probe.answer.lower(), True),
        ("Answer:  %s " % probe.answer, True),
        ("blah\nAnswer: %s\ntrailing noise" % probe.answer, True),
        ("So the final symbol is %s" % probe.answer, True),
        (probe.answer[1:], True),
        ("Answer: X%d" % ((int(probe.answer[1:]) + 3) % 10), False),
    ]
    for text, want in tol:
        if check(probe, text) != want:
            failures.append("tolerant-parse: %r expected %s" % (text, want))

    # ---- redaction (AMENDMENT 4), in both formats (AMENDMENT 5) ------------------------------
    rrng = random.Random(20260918)
    for trial in range(20):
        depth = rrng.choice([1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48])
        seed = rrng.randrange(10**6)
        base_knobs = {}
        if trial % 4 == 1:
            base_knobs["fresh_tables_per_step"] = True
        if trial % 4 == 2:
            base_knobs["include_identity"] = True
        if trial % 4 == 3:
            base_knobs["seq_len"] = rrng.choice([2, 6])
            base_knobs["n_alphabets"] = 10

        for fmt in ("published", "ergonomic"):
            knobs = dict(base_knobs, format=fmt)
            inst = generate(depth, seed, **knobs)
            tag = "d=%d s=%d %r" % (depth, seed, knobs)

            if redact_prompt(inst, 0) != inst.prompt:
                failures.append("redact k=0 not identity %s" % tag)

            for k in sorted({0, 1, depth // 2, depth}):
                red = redact_prompt(inst, k)
                if k == 0:
                    continue
                if red == inst.prompt:
                    failures.append("redact k=%d did not change the prompt %s" % (k, tag))
                if _REDACTED not in red:
                    failures.append("redact k=%d has no placeholder %s" % (k, tag))

                if fmt == "published":
                    doc = [l for l in red.splitlines() if l.startswith("Document:")]
                    start_l = [l for l in red.splitlines() if l.startswith("Starting from the ")]
                    if len(doc) != 1 or len(start_l) != 1:
                        failures.append("redact k=%d lost Document:/Starting-from line %s" % (k, tag))
                        continue
                    # the initial state is gone from both places that stated it, at every k >= 1
                    if re.search(r"X\d", doc[0]) or re.search(r"X\d", start_l[0]):
                        failures.append("redact k=%d leaks the start symbol %s" % (k, tag))
                    # the first k task tokens are gone; the rest survive verbatim
                    head = doc[0][len("Document: S "):].split()
                    if head[:k] != [_REDACTED] * k:
                        failures.append("redact k=%d did not blank the first %d task tokens %s" % (k, k, tag))
                    want_tail = [_task_token(d, c) for d, c in enumerate(inst.meta["choices"])][k:]
                    if head[k:-1] != want_tail:
                        failures.append("redact k=%d damaged the surviving task tokens %s" % (k, tag))
                    if k == depth:
                        # task-specific k=depth assertion: no operator token and no state anywhere
                        # in the document or the question; only tables and the question remain
                        if re.search(r"T\d+_\d+", doc[0]) or re.search(r"T\d+_\d+", start_l[0]):
                            failures.append("redact k=depth leaks an operator token %s" % tag)
                        if "What is the " not in red:
                            failures.append("redact k=depth dropped the question %s" % tag)
                else:
                    start_l = [l for l in red.splitlines() if l.startswith("Start at ")]
                    order = [l for l in red.splitlines() if _ORDER_RE.match(l)
                             or l.startswith("step ") and _REDACTED in l]
                    if len(start_l) != 1 or len(order) != depth:
                        failures.append("redact k=%d lost the start line / order list %s" % (k, tag))
                        continue
                    if re.search(r"X\d", start_l[0]):
                        failures.append("redact k=%d leaks the start symbol %s" % (k, tag))
                    want = [_order_line(i + 1, _REDACTED) if i < k
                            else _order_line(i + 1, _table_name(
                                _resolve(knobs), i, inst.meta["choices"][i]))
                            for i in range(depth)]
                    if order != want:
                        failures.append("redact k=%d mangled the order list %s" % (k, tag))
                    if k == depth:
                        if any(_REDACTED not in l for l in order):
                            failures.append("redact k=depth leaks an operator name %s" % tag)
                        if "What do you end at" not in red:
                            failures.append("redact k=depth dropped the question %s" % tag)

                # static material is untouched
                for line in inst.prompt.splitlines():
                    if (line.startswith(("Alphabet:", "There are ")) or ": X" in line[:24]) \
                            and line not in red:
                        failures.append("redact k=%d altered static material %r %s" % (k, line[:40], tag))
                        break
            for bad in (-1, inst.depth + 1):
                try:
                    redact_prompt(inst, bad)
                except ValueError:
                    pass
                else:
                    failures.append("redact k=%d should have raised %s" % (bad, tag))

    # ---- step corruption (AMENDMENT 6), in both formats ---------------------------------------
    crng = random.Random(20260919)
    for trial in range(20):
        depth = crng.choice([1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48])
        seed = crng.randrange(10**6)
        cseed = crng.randrange(10**6)
        base_knobs = {}
        if trial % 5 == 1:
            base_knobs["n_alphabets"] = crng.choice([2, 5, 16])
        if trial % 5 == 2:
            base_knobs["include_identity"] = True
        if trial % 5 == 3:
            base_knobs["seq_len"] = crng.choice([2, 6])
            base_knobs["n_alphabets"] = 10
        if trial % 5 == 4:
            base_knobs["fresh_tables_per_step"] = True

        insts = {f: generate(depth, seed, **dict(base_knobs, format=f))
                 for f in ("published", "ergonomic")}
        for k in sorted({1, max(1, depth // 2), depth}):
            bad_states = {}
            for fmt, inst in insts.items():
                tag = "d=%d s=%d k=%d %r fmt=%s" % (depth, seed, k, base_knobs, fmt)
                n = inst.meta["n_alphabets"]
                gold = inst.steps[k - 1]
                text, bad = corrupt_step(inst, k, cseed)
                bad_states[fmt] = bad

                if bad == inst.states[k - 1]:
                    failures.append("corrupt_step k=%d reports the TRUE state %s" % (k, tag))
                if text == gold:
                    failures.append("corrupt_step k=%d did not change the step text %s" % (k, tag))
                if (text, bad) != corrupt_step(inst, k, cseed):
                    failures.append("corrupt_step nondeterministic %s" % tag)

                # the wrong state is a legal state of this instance: right length, right alphabet
                syms = _parse_state(bad, n)
                if len(syms) != inst.meta["seq_len"] or not all(0 <= x < n for x in syms) \
                        or _vec(syms) != bad:
                    failures.append("corrupt_step k=%d state %r is not well formed %s" % (k, bad, tag))

                # task-specific structure: the step line still parses, and its ACTION is verbatim
                before = _parse_state(inst.meta["start"] if k == 1 else inst.states[k - 2], n)
                if fmt == "published":
                    m, m0 = _PUB_STEP_RE.match(text), _PUB_STEP_RE.match(gold)
                    if not m:
                        failures.append("corrupt_step k=%d broke the step line %r %s" % (k, text, tag))
                        continue
                    if m.group(1) != m0.group(1):
                        failures.append("corrupt_step k=%d changed the task token %s" % (k, tag))
                    if m.group(2) != bad:
                        failures.append("corrupt_step k=%d text/state disagree %s" % (k, tag))
                else:
                    m, m0 = _ERGO_STEP_RE.match(text), _ERGO_STEP_RE.match(gold)
                    if not m:
                        failures.append("corrupt_step k=%d broke the step line %r %s" % (k, text, tag))
                        continue
                    if m.group(1, 2, 3) != m0.group(1, 2, 3):
                        failures.append("corrupt_step k=%d changed the action text %s" % (k, tag))
                    if m.group(5) != bad:
                        failures.append("corrupt_step k=%d 'Now at' disagrees with the state %s" % (k, tag))
                    entries = {int(a): int(b) for a, b in _PAIR_RE.findall(m.group(4))}
                    if entries:
                        # the rewritten table entry and the rewritten 'Now at' tell one story
                        if sorted(entries) != sorted(set(before)):
                            failures.append("corrupt_step k=%d damaged the entry list %s" % (k, tag))
                        elif [entries[b] for b in before] != syms:
                            failures.append("corrupt_step k=%d entry disagrees with 'Now at' %s" % (k, tag))
                    elif "is the identity" not in m.group(4):
                        failures.append("corrupt_step k=%d lost the table entry %s" % (k, tag))
            if bad_states["published"] != bad_states["ergonomic"]:
                failures.append("corrupt_step differs across formats d=%d s=%d k=%d %r"
                                % (depth, seed, k, base_knobs))
        for inst in insts.values():
            for bad_k in (0, -1, depth + 1):
                try:
                    corrupt_step(inst, bad_k, cseed)
                except ValueError:
                    pass
                else:
                    failures.append("corrupt_step k=%d should have raised d=%d %r"
                                    % (bad_k, depth, base_knobs))

    if failures:
        for f in failures[:20]:
            print("FAIL:", f)
        print("selftest FAILED: %d failure(s)" % len(failures))
        return 1
    print("200 random (depth, seed) pairs across 7 knob settings x both formats")
    print("  solve() == answer                         ok")
    print("  check(gold CoT)                           ok")
    print("  check(wrong answer) == False              ok")
    print("  generate() deterministic                  ok")
    print("  len(steps) == len(states) == depth        ok")
    print("  step_spans() well formed                  ok")
    print("  format='ergonomic' == format='published' on instance, answer, states, depth, solve, check   ok")
    print("  exemplars(3, 0) renders; [0] is the published depth-5 trace   ok")
    print("  exemplars(3, 0, format='ergonomic') forwards the knob; no published reconstruction  ok")
    print("  tolerant answer parsing (7 probes)        ok")
    print("20 instances x both formats x k in {0, 1, depth//2, depth}")
    print("  redact_prompt(inst, 0) == inst.prompt     ok")
    print("  k>0: prompt changes and contains the placeholder             ok")
    print("  k>0: start symbol gone; first k operators blanked; rest verbatim   ok")
    print("  k=depth: no operator left in the document / order list; question kept   ok")
    print("  lookup tables / alphabet untouched at every k                ok")
    print("20 instances x both formats x k in {1, depth//2, depth}")
    print("  corrupt_step() state != the true state at k              ok")
    print("  corrupt_step() step text != the gold step text           ok")
    print("  corrupted state well formed (length / alphabet)          ok")
    print("  step line still parses; action text verbatim; reported state == returned state   ok")
    print("  ergonomic: rewritten table entry agrees with 'Now at'    ok")
    print("  same corruption in both formats; deterministic in (inst, k, seed)   ok")
    print("  k outside [1, depth] raises                              ok")
    print("selftest PASSED")
    return 0


def _render(inst: Instance, title: str) -> None:
    print("=" * 92)
    print(title)
    print("-" * 92)
    print("--- prompt ---")
    print(inst.prompt)
    print("--- gold CoT ---")
    print(format_cot(inst))
    print()


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, 1000 + i)
            _render(inst, "depth=%d seed=%d  knobs=%s" % (
                depth, 1000 + i,
                ", ".join("%s=%r" % (k, inst.meta[k]) for k in
                          ("n_alphabets", "n_functions", "bijective",
                           "fresh_tables_per_step", "include_identity", "seq_len", "format"))))

    # AMENDMENT 5: one instance per format.  Same depth, same seed, same knobs -> the same
    # instance (same tables, start symbol and table order), written two ways.
    d, sd = 6, 2026
    insts = {f: generate(d, sd, format=f) for f in ("published", "ergonomic")}
    print("#" * 92)
    print("# AMENDMENT 5 -- one instance per format, at depth=%d seed=%d" % (d, sd))
    print("#" * 92)
    for fmt, inst in insts.items():
        _render(inst, "format=%r  depth=%d seed=%d" % (fmt, d, sd))
    a, b = insts["published"], insts["ergonomic"]
    print("same instance across formats: start=%s choices=%s answer=%s states=%s  (all equal: %s)" % (
        a.meta["start"] == b.meta["start"], a.meta["choices"] == b.meta["choices"],
        a.answer == b.answer, a.states == b.states,
        (a.meta["start"], a.meta["choices"], a.answer, a.states)
        == (b.meta["start"], b.meta["choices"], b.answer, b.states)))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.demo:
        _demo()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
