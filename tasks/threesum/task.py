"""
threesum -- Pfau, Merrill & Bowman (2024), "Let's Think Dot by Dot: Hidden
Computation in Transformer Language Models" (arXiv:2404.15758), the 3SUM task
with the paper's *serial* ("instance-adaptive") chain of thought.

Problem. n tuples of `dimension` digits mod `mod` are listed. Decide whether
some three of them sum to the all-zero vector coordinate-wise mod `mod`.
Answer is the literal token True or False.

Published format (reproduced exactly; see published_trace.txt and README.md):

    901 457 892 282 752 384 P 0- 1- 4- 1 7 2 0 0 A True

Everything up to and including `P` is the problem statement. After `P` the
serial CoT enumerates candidate triples. A candidate is a triple of positions
d < e < f whose *first* coordinate already sums to 0 mod `mod`; each one is
emitted as

    <d>- <e>- <f>- <x> <y> <z> <s1> [<s2> ...]

i.e. the three positions (digit + literal '-'), then one raw input digit copied
out of a single randomly chosen coordinate of each of the three tuples, then the
coordinate-wise sums mod `mod` for coordinates 1, 2, ... -- emission stopping at
the first nonzero sum, so a non-match is proved lazily. The enumeration stops at
the first candidate whose every coordinate sums to zero. `A` then closes the
trace with True/False.

Depth = number of enumerated candidate-triple checks = len(steps). See README.md
("Depth semantics") for the depth -> n mapping and the sampling conditions.

AMENDMENT 5 adds the knob format="ergonomic": the same instances, the same
candidate order, the same depth/answer/states, rendered as a plain-language
prompt and one self-contained line per candidate check, e.g.

    (0,1,4): 901 + 457 + 752 -> (9+4+7, 0+5+5, 1+7+2) = (20, 10, 10)
             -> (0, 0, 0) mod 10 -> yes

(on one line). format="published" stays the default.

AMENDMENT 6 adds corrupt_step(inst, k, seed): candidate check k's verdict flipped,
in whichever format the instance is written. See README.md ("Corruption").

Vendored for reference only (never imported; it needs numpy/torch/transformers):
vendor/fillerTokens (JacobPfau/fillerTokens @ cb39af6458b7476ba07f25e89a9c8fd339c1e229),
src/match3.py -- Match3.get_true_instance / get_corrupted_instance / serial_solve
and the serial_cot string builder. The algorithm below is a faithful pure-python
transcription of those; no code was copied (the repo ships no license).
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass, field


@dataclass
class Instance:
    prompt: str
    steps: list[str]
    states: list[str]
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "either True or False"

# depth = number of candidate-triple checks in the serial trace. The generator
# picks the list length n whose expected candidate count C(n,3)/mod is closest to
# the requested depth, so this ladder is also the paper's own length sweep:
#   depth  2 ->  n=6    depth 12 -> n=10
#   depth  4 ->  n=7    depth 22 -> n=12
#   depth  8 ->  n=9    depth 36 -> n=14
# Pfau et al. sweep length 6..14 at dimension 3 (Fig. 2); their from-scratch
# 34M-parameter model is already at ~66% without intermediate tokens at n=12.
DEPTHS = [2, 4, 8, 12, 22, 36]

KNOBS = {
    "dimension": (3, "tuple width; coordinate 0 selects candidates, coordinates 1.. verify them. Must be >= 2"),
    "mod": (10, "modulus; each coordinate is one digit, so 2 <= mod <= 10"),
    "n": (None, "list length. None = derived from depth (argmin over n of |C(n,3)/mod - depth|)"),
    "label": (None, "force the answer: True, False, or None to draw 50/50 from the seed"),
    "corruption_rate": (4 / 3, "Pfau's knob for negative instances: a planted zero-sum triple is "
                               "corrupted in min(Geom(1/corruption_rate), 3) rows, giving hard near-misses"),
    "max_attempts": (400000, "rejection-sampling budget before generate() gives up"),
    "format": ("published", "trace/prompt style: 'published' = Pfau's symbol string (default); "
                            "'ergonomic' = plain-language prompt + one full-state line per candidate "
                            "check (AMENDMENT 5). Instances, candidate order, states, depth, answer "
                            "and solve() are identical under both"),
}

FORMATS = ("published", "ergonomic")

# Redaction (AMENDMENT 4). The rows are the inputs; the state a step updates is the
# "has a full match been seen yet" flag plus the enumeration cursor. A row consumed by
# steps 1..k that is ALSO consumed by a later candidate must stay (it is needed to
# continue), so what can be removed is the verifying digits (coordinates 1..) of rows
# used only by steps 1..k. Coordinate 0 always stays: it is what makes a triple a
# candidate at all, so without it the remaining enumeration is unreadable.
REDACTION_MEANINGFUL = True

_PLACEHOLDER = "[\u2026]"

# Verbatim from published_trace.txt (regenerated from the paper's own repo; see
# that file's header). Used as exemplars(k, seed)[0]; also the fallback if the
# file is unreadable.
_PUBLISHED_LINE = " 901 457 892 282 752 384 P 0- 1- 4- 1 7 2 0 0 A True"


# --------------------------------------------------------------------------
# core task logic (transcribed from Match3)
# --------------------------------------------------------------------------

def _cat(row: tuple[int, ...]) -> str:
    return "".join(str(x) for x in row)


def _candidates(rows: list[tuple[int, ...]], dimension: int, mod: int) -> list[tuple[int, int, int, bool]]:
    """Every (d, e, f), d < e < f, whose coordinate-0 digits sum to 0 mod `mod`,
    in Match3.serial_solve's enumeration order, tagged with whether the *whole*
    tuple sum is zero. No early stop -- serial_solve's early stop is applied by
    the caller."""
    first = [r[0] for r in rows]
    n = len(rows)
    out = []
    for d in range(n - 2):
        for e in range(d + 1, n - 1):
            inv = (mod - (first[d] + first[e]) % mod) % mod
            for f in range(e + 1, n):
                if first[f] == inv:
                    full = all((rows[d][j] + rows[e][j] + rows[f][j]) % mod == 0
                               for j in range(1, dimension))
                    out.append((d, e, f, full))
    return out


def _lazy_emit(sums: list[int]) -> list[int]:
    """Match3.serial_cot's laziness: emit coordinate sums up to and including the
    first nonzero one, which is all it takes to disprove a candidate."""
    out = []
    for v in sums:
        out.append(v)
        if v != 0:
            break
    return out


def _render(rows, cands, dimension, mod, rng):
    """Match3.serial_solve + serial_cot: walk the candidates, emit one step each,
    stop after the first full match."""
    steps, states = [], []
    found = False
    for (d, e, f, full) in cands:
        cot_dim = rng.randrange(dimension)          # serial_solve: self.random.choice(self.dimension)
        digits = [rows[d][cot_dim], rows[e][cot_dim], rows[f][cot_dim]]
        sums = [(rows[d][j] + rows[e][j] + rows[f][j]) % mod for j in range(1, dimension)]
        emitted = _lazy_emit(sums)              # lazy: stop at the first nonzero
        toks = [f"{d}-", f"{e}-", f"{f}-"] + [str(x) for x in digits] + [str(x) for x in emitted]
        steps.append(" ".join(toks))
        if full:
            found = True
        states.append(f"{d}-{e}-{f}/{'T' if found else 'F'}")
        if found:
            break
    return steps, states, found


def _render_ergonomic(rows, cands, dimension, mod):
    """AMENDMENT 5: the same walk over the same candidates, one self-contained line
    each, with every digit-wise sum written out in full and an explicit verdict.
    Same lazy stop across steps (the enumeration ends at the first hit)."""
    steps, states = [], []
    found = False
    for (d, e, f, full) in cands:
        terms = ", ".join("+".join(str(rows[p_][j]) for p_ in (d, e, f)) for j in range(dimension))
        totals = [sum(rows[p_][j] for p_ in (d, e, f)) for j in range(dimension)]
        mods = [t % mod for t in totals]
        hit = all(m == 0 for m in mods)             # == full: coordinate 0 is 0 by candidacy
        steps.append(
            f"({d},{e},{f}): {_cat(rows[d])} + {_cat(rows[e])} + {_cat(rows[f])}"
            f" -> ({terms})"
            f" = ({', '.join(str(t) for t in totals)})"
            f" -> ({', '.join(str(m) for m in mods)}) mod {mod}"
            f" -> {'yes' if hit else 'no'}")
        if full:
            found = True
        states.append(f"{d}-{e}-{f}/{'T' if found else 'F'}")
        if found:
            break
    return steps, states, found


def _published_prompt(rowstrs: list[str]) -> str:
    """Pfau's problem statement: the tuples, then the literal `P`."""
    return " ".join(rowstrs) + " P"


def _ergonomic_prompt(rowstrs: list[str], dimension: int, mod: int) -> str:
    """AMENDMENT 5: the same problem, stated in plain language, one number per line
    so the positions the trace names are readable off the page."""
    n = len(rowstrs)
    body = "\n".join(f"{i}: {r}" for i, r in enumerate(rowstrs))
    return (f"Here are {n} {dimension}-digit numbers (read each one as {dimension} separate digits), "
            f"numbered 0 to {n - 1}:\n\n{body}\n\n"
            f"Is there a triple of positions i < j < k whose digit-wise sums are all 0 mod {mod} "
            f"-- that is, the first digits of the three numbers sum to 0 mod {mod}, the second digits "
            f"sum to 0 mod {mod}, and so on for all {dimension} digit positions? Answer True or False.")


def _prompt_for(fmt: str, rowstrs: list[str], dimension: int, mod: int) -> str:
    return _published_prompt(rowstrs) if fmt == "published" else _ergonomic_prompt(rowstrs, dimension, mod)


def _geometric(rng: random.Random, p: float) -> int:
    """Number of Bernoulli(p) trials up to and including the first success (>= 1),
    matching numpy Generator.geometric."""
    k = 1
    while rng.random() >= p:
        k += 1
        if k > 64:
            break
    return k


def _planted(rng: random.Random, dimension: int, mod: int) -> list[list[int]]:
    a = [rng.randrange(mod) for _ in range(dimension)]
    b = [rng.randrange(mod) for _ in range(dimension)]
    c = [(mod - (a[j] + b[j]) % mod) % mod for j in range(dimension)]
    return [a, b, c]


def _draw(rng: random.Random, n: int, dimension: int, mod: int,
          want_true: bool, corruption_rate: float) -> list[tuple[int, ...]]:
    """One candidate instance, Match3-style: plant a zero-sum triple, corrupt it
    if a negative is wanted, pad with noise, shuffle."""
    rows = _planted(rng, dimension, mod)
    if not want_true:
        # Match3.get_corrupted_instance: inputs[:k, columns] = vals, i.e. every one
        # of the first k rows gets vals[j] written at column columns[j].
        k = min(_geometric(rng, 1.0 / corruption_rate), 3)
        cols = [rng.randrange(dimension) for _ in range(k)]
        vals = [rng.randrange(mod) for _ in range(k)]
        for r in range(min(k, 3)):
            for j, col in enumerate(cols):
                rows[r][col] = vals[j]
    rows += [[rng.randrange(mod) for _ in range(dimension)] for _ in range(n - 3)]
    rng.shuffle(rows)
    return [tuple(r) for r in rows]


def _n_for_depth(depth: int, mod: int) -> int:
    """argmin over n >= 6 of |C(n,3)/mod - depth|; ties go to the smaller n."""
    best, best_err = 6, None
    n = 6
    while n <= 400:
        err = abs((n * (n - 1) * (n - 2) / 6.0) / mod - depth)
        if best_err is None or err < best_err - 1e-12:
            best, best_err = n, err
        elif err > best_err:
            break
        n += 1
    return best


# --------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------

def generate(depth: int, seed: int, dimension: int = 3, mod: int = 10, n: int | None = None,
             label: bool | str | None = None, corruption_rate: float = 4 / 3,
             max_attempts: int = 400000, format: str = "published", **_ignored) -> Instance:
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    if depth < 1:
        raise ValueError("depth must be >= 1 (a trace with no candidate check has no steps)")
    if dimension < 2:
        raise ValueError("dimension must be >= 2: at dimension 1 the selector coordinate is also the "
                         "only verifier coordinate, so every candidate matches and the serial trace "
                         "collapses to a single step")
    if not (2 <= mod <= 10):
        raise ValueError("mod must be in 2..10 (the published format writes one digit per coordinate)")

    if isinstance(label, str):
        label = {"true": True, "false": False}[label.strip().lower()]
    if n is None:
        n = _n_for_depth(depth, mod)
    if n < 6:
        raise ValueError("n must be >= 6 (3 slots for the planted triple plus at least 3 noise tuples)")

    rng = random.Random(f"threesum|{seed}|{depth}|{dimension}|{mod}|{n}|{label}|{corruption_rate}")
    want_true = rng.random() < 0.5 if label is None else bool(label)

    rows = None
    cands: list = []
    for _ in range(max_attempts):
        cand_rows = _draw(rng, n, dimension, mod, want_true, corruption_rate)
        cs = _candidates(cand_rows, dimension, mod)
        if len(cs) != depth:
            continue
        matches = [i for i, c in enumerate(cs) if c[3]]
        # Both labels are conditioned on having exactly `depth` candidates, and a
        # positive must match on the LAST one -- so the answer is never inferable
        # from the candidate count, and every emitted step is load-bearing.
        if want_true:
            if matches != [depth - 1]:
                continue
        elif matches:
            continue
        rows, cands = cand_rows, cs
        break
    if rows is None:
        raise RuntimeError(
            f"threesum: could not sample depth={depth} label={want_true} at n={n}, dimension={dimension}, "
            f"mod={mod} within {max_attempts} attempts; try a different n or a depth nearer C(n,3)/mod")

    # The rng is only consumed past this point by the published renderer's cot_dim
    # draws, so both formats see the identical instance for a given (depth, seed).
    if format == "published":
        steps, states, found = _render(rows, cands, dimension, mod, rng)
    else:
        steps, states, found = _render_ergonomic(rows, cands, dimension, mod)
    assert found == want_true and len(steps) == depth

    rowstrs = [_cat(r) for r in rows]
    return Instance(
        prompt=_prompt_for(format, rowstrs, dimension, mod),
        steps=steps,
        states=states,
        answer="True" if found else "False",
        depth=depth,
        meta={"seed": seed, "n": n, "dimension": dimension, "mod": mod,
              "corruption_rate": corruption_rate, "rows": rowstrs,
              "n_candidates": len(cands), "published_exemplar": False,
              "format": format},
    )


_ERGONOMIC_ROW = re.compile(r"^\s*(\d+):\s*([0-9]+)\s*$")


def _rows_from_prompt(inst: Instance) -> list[tuple[int, ...]]:
    """The tuples, read back out of the problem statement in whichever format it is
    written (published: whitespace-separated digit strings before `P`; ergonomic:
    the numbered `i: digits` lines)."""
    if inst.meta.get("format", "published") == "ergonomic":
        out = []
        for line in inst.prompt.splitlines():
            m = _ERGONOMIC_ROW.match(line)
            if m:
                out.append(tuple(int(ch) for ch in m.group(2)))
        return out
    return [tuple(int(ch) for ch in t) for t in inst.prompt.split() if t != "P"]


def solve(inst: Instance) -> str:
    """Independent reference solver.

    generate() decides the answer by enumerating O(n^3) position triples whose
    coordinate-0 digits cancel and then verifying the remaining coordinates.
    This instead parses the tuples straight back out of the problem statement and
    runs the textbook O(n^2) hash reduction: index every tuple by value, then for
    each ordered pair look up the single complement vector that would complete
    it. Different algorithm, different data structure, same decision.
    """
    mod = inst.meta["mod"]
    rows = _rows_from_prompt(inst)

    last: dict[tuple[int, ...], int] = {}
    for k, row in enumerate(rows):
        last[row] = k                # latest index holding this value

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            need = tuple((-rows[i][c] - rows[j][c]) % mod for c in range(len(rows[i])))
            if last.get(need, -1) > j:
                return "True"
    return "False"


_ANSWER_LINE = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)
_TF = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)


def _extract(completion: str) -> str:
    """Last `Answer:` line, else last non-empty line; then the last True/False in it."""
    lines = completion.splitlines()
    payload = None
    for line in reversed(lines):
        m = _ANSWER_LINE.match(line)
        if m:
            payload = m.group(1)
            break
    if payload is None:
        for line in reversed(lines):
            if line.strip():
                payload = line
                break
    if payload is None:
        return ""
    hits = _TF.findall(payload)
    if not hits:
        return ""
    word = hits[-1].lower()
    return "True" if word in ("true", "yes") else "False"


def check(inst: Instance, completion: str) -> bool:
    return _extract(completion) == inst.answer


def format_cot(inst: Instance) -> str:
    """published: the one-line serial trace closed by Pfau's `A <label>`.
    ergonomic: one line per candidate check, closed by a plain-language sentence.
    Both then get the harness-mandated final `Answer:` line."""
    if inst.meta.get("format", "published") == "published":
        return " ".join(inst.steps) + " A " + inst.answer + "\nAnswer: " + inst.answer
    mod = inst.meta["mod"]
    if inst.answer == "True":
        d, e, f = _triples(inst)[-1]
        closing = (f"The triple ({d},{e},{f}) has every digit-wise sum 0 mod {mod}. "
                   f"So the answer is True.")
    else:
        closing = (f"No triple of positions has every digit-wise sum 0 mod {mod}. "
                   f"So the answer is False.")
    return "\n".join(inst.steps) + "\n" + closing + "\nAnswer: " + inst.answer


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    """Char offsets of each step inside format_cot(inst)."""
    spans, pos = [], 0
    for s in inst.steps:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1
    return spans


def _triples(inst: Instance) -> list[tuple[int, ...]]:
    """The candidate triple checked by each step, read off `states` (which is the
    same in both formats: `<d>-<e>-<f>/<T|F>`)."""
    return [tuple(int(x) for x in s.split("/")[0].split("-")) for s in inst.states]


def redact_prompt(inst: Instance, k: int) -> str:
    """AMENDMENT 4. Remove what steps 1..k consumed and nothing that steps k+1..depth
    need, in whichever format the prompt is written.

    The consumable input here is a row's *verifying* digits (coordinates 1..). Its
    coordinate-0 digit is what makes a triple a candidate, so it is the enumeration
    itself, not state, and always stays -- as does every row still named by a later
    candidate. So: rows used by steps 1..k and by no step after k lose their
    verifying digits to a single `[...]` span each. k=0 returns the prompt unchanged;
    k=depth blanks the verifying digits of every row the trace ever touches, which is
    exactly what is needed to decide the answer.

    The enumeration is lexicographic, so low-index rows drop out of it early and the
    set is non-empty in practice (no no-op was observed at any depth in DEPTHS); a row
    that recurs in every later candidate would legitimately produce one.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return inst.prompt
    k = min(k, inst.depth)

    triples = _triples(inst)
    early = {p for t in triples[:k] for p in t}
    late = {p for t in triples[k:] for p in t}
    hide = early - late
    if not hide:
        return inst.prompt

    rowstrs = [r[0] + _PLACEHOLDER if i in hide else r
               for i, r in enumerate(inst.meta["rows"])]
    return _prompt_for(inst.meta.get("format", "published"), rowstrs,
                       inst.meta["dimension"], inst.meta["mod"])


# --------------------------------------------------------------------------
# corrupt_step -- mistake propagation (SPEC AMENDMENT 6)
# --------------------------------------------------------------------------

# The state a step reports is `<d>-<e>-<f>/<T|F>`: the candidate cursor plus the
# one bit "has a full match been seen yet". The cursor is the enumeration, not
# state the model computes, so the only thing there is to get wrong is the bit --
# i.e. the verdict of candidate check k. That is also the task's own error mode
# (Pfau's negatives are near-misses: one coordinate off), and it is the minimal
# edit AMENDMENT 6 asks for ("report the wrong branch of a boolean/threesum
# check", "alter one digit of an arithmetic result").
#
# Because the enumeration stops at the first hit, a true `T` can only appear on
# the last step, so no step before k has already flipped the bit: flipping check
# k's verdict flips the reported flag, in both directions.
#
#   no  -> yes   every verifying coordinate is fabricated to 0 mod `mod`
#   yes -> no    exactly one verifying coordinate is fabricated nonzero
#
# The fabricated residues are drawn from an rng keyed on format-independent data
# (the rows, the step's true state, k, seed), so both formats tell the same lie
# about the same coordinate -- only the rendering differs.


def _fabricated_mods(mods: list[int], mod: int, rng: random.Random) -> list[int]:
    """The coordinate-wise sums mod `mod` that the flipped verdict reports.

    `mods[0]` is coordinate 0, which is 0 for every candidate (that is what made the
    triple a candidate) and is left alone in both directions -- corrupting it would
    contradict the enumeration itself rather than the check.
    """
    if all(m == 0 for m in mods):                    # a true match: break exactly one coordinate
        out = list(mods)
        j = rng.randrange(1, len(mods))              # a verifying coordinate, never coordinate 0
        out[j] = rng.randrange(1, mod)
        return out
    return [0] * len(mods)                           # a true non-match: fabricate the full match


def _plausible_total(true_total: int, residue: int, mod: int, rng: random.Random) -> int:
    """A raw (un-modded) three-digit-sum in 0..3*(mod-1) congruent to `residue`, as
    close to the truth as possible and never equal to it -- the arithmetic slip a
    model would actually make. At least two such values always exist."""
    hi = 3 * (mod - 1)
    cands = [t for t in range(hi + 1) if t % mod == residue and t != true_total]
    best = min(abs(t - true_total) for t in cands)
    return rng.choice([t for t in cands if abs(t - true_total) == best])


def _corrupt_published_step(step: str, dimension: int, mod: int, rng: random.Random) -> str:
    """`<d>- <e>- <f>- <x> <y> <z> <s1> [<s2> ...]` -> the same positions and the same
    copied digits, with the reported sums rewritten for the flipped verdict (and
    re-truncated by the published trace's own lazy-emission rule)."""
    toks = step.split()
    head, emitted = toks[:6], [int(t) for t in toks[6:]]
    # The emission stops at the first nonzero, so what is on the page is a prefix of
    # the verifying sums; pad it back out with zeros. Only "are they all zero" matters
    # to _fabricated_mods, and a padded tail is all-zero exactly when the step matched.
    mods = [0] + emitted + [0] * (dimension - 1 - len(emitted))
    return " ".join(head + [str(v) for v in _lazy_emit(_fabricated_mods(mods, mod, rng)[1:])])


def _corrupt_ergonomic_step(step: str, mod: int, rng: random.Random) -> str:
    """`(d,e,f): A + B + C -> (<terms>) = (<totals>) -> (<mods>) mod m -> yes|no` ->
    the same triple, the same copied-out numbers and the same `a+b+c` term list, with
    one (or, for a non-match with several offending coordinates, each offending) total
    rewritten so that the mod tuple and the verdict flip with it."""
    p0, p1, p2, _verdict = step.split(" -> ")
    terms, totals_str = p1.split(" = ")
    totals = [int(x) for x in totals_str[1:-1].split(", ")]
    new_mods = _fabricated_mods([t % mod for t in totals], mod, rng)
    new_totals = [t if t % mod == m else _plausible_total(t, m, mod, rng)
                  for t, m in zip(totals, new_mods)]
    return " -> ".join([
        p0,
        f"{terms} = ({', '.join(str(t) for t in new_totals)})",
        f"({', '.join(str(m) for m in new_mods)}) mod {mod}",
        "yes" if all(m == 0 for m in new_mods) else "no",
    ])


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """AMENDMENT 6. Return `(step_text, corrupted_state)` for step `k` (1-based): check
    k's verdict flipped, rendered in the instance's own `format`.

    The positions `(d,e,f)` and every digit copied out of the input are byte-identical
    to `inst.steps[k-1]`; only the reported sums and the verdict change. A `no` becomes
    a fabricated `-> (0, 0, 0) mod 10 -> yes` (published: the lazily-emitted nonzero sum
    becomes the full run of zeros), a `yes` becomes a `no` with exactly one nonzero
    coordinate. `corrupted_state` is `<d>-<e>-<f>/<T|F>` as in `inst.states`, with the
    match flag flipped.

    Note the lazy stop: a fabricated `yes` at k < depth is a state the gold trace would
    have ended on. That is intended -- the model continues however it likes and only the
    final answer is scored.

    Deterministic in `(inst, k, seed)`, and keyed on format-independent data, so the two
    formats fabricate the same residue on the same coordinate.
    """
    if not 1 <= k <= inst.depth:
        raise ValueError(f"k must be in 1..{inst.depth}, got {k}")

    fmt = inst.meta.get("format", "published")
    mod, dimension = inst.meta["mod"], inst.meta["dimension"]
    rng = random.Random("threesum-corrupt|%s|%s|%d|%d"
                        % (" ".join(inst.meta["rows"]), inst.states[k - 1], k, seed))

    step = inst.steps[k - 1]
    if fmt == "published":
        text = _corrupt_published_step(step, dimension, mod, rng)
        hit = text.split()[6:] == ["0"] * (dimension - 1)
    else:
        text = _corrupt_ergonomic_step(step, mod, rng)
        hit = text.endswith(" -> yes")

    d, e, f = _triples(inst)[k - 1]
    return text, f"{d}-{e}-{f}/{'T' if hit else 'F'}"


def _published_instance() -> Instance:
    """exemplars()[0]: the verbatim published trace, parsed back into an Instance.

    NB this instance does not obey the generator's "match on the last candidate"
    condition -- it is published data, reproduced as published (its triple 0,1,4
    matches on the first of its two candidates). See README.md.
    """
    line = _PUBLISHED_LINE
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_trace.txt")
    try:
        with open(path, encoding="utf-8") as fh:
            body = [ln for ln in fh.read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if body and " P " in body[-1] and " A " in body[-1]:
            line = body[-1]
    except OSError:
        pass

    head, rest = line.split(" P ", 1)
    cot, ans = rest.rsplit(" A ", 1)
    rows = head.split()
    toks = cot.split()

    steps, states, found = [], [], False
    i = 0
    while i < len(toks):
        j = i + 3                                   # three '<idx>-' tokens
        d, e, f = (int(t.rstrip("-")) for t in toks[i:j])
        j += 3                                      # three copied digits
        sums = []
        while j < len(toks) and not toks[j].endswith("-"):
            sums.append(toks[j])
            j += 1
            if sums[-1] != "0":
                break
        steps.append(" ".join(toks[i:j]))
        if len(sums) == 2 and all(s == "0" for s in sums):
            found = True
        states.append(f"{d}-{e}-{f}/{'T' if found else 'F'}")
        i = j

    return Instance(
        prompt=" ".join(rows) + " P",
        steps=steps,
        states=states,
        answer=ans.strip(),
        depth=len(steps),
        meta={"seed": None, "n": len(rows), "dimension": 3, "mod": 10,
              "corruption_rate": None, "rows": rows, "n_candidates": None,
              "published_exemplar": True, "format": "published"},
    )


def exemplars(k: int, seed: int, format: str = "published", dimension: int = 3, mod: int = 10,
              **_ignored) -> list[Instance]:
    """published: [0] is the published trace verbatim, the rest generated at modest
    depth with alternating answers so a few-shot block does not bias the label.
    ergonomic: all k are generated in that format -- the published symbol string is
    not a useful exemplar for a format it was never written in (AMENDMENT 5)."""
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}")
    plan = [(2, False), (3, True), (2, True), (4, False), (3, False), (2, False)]
    out: list[Instance] = []
    start = 1
    if format == "published":
        if k > 0:
            out.append(_published_instance())
    else:
        start = 0
    for i in range(start, k):
        depth, lab = plan[i % len(plan)] if start == 0 else plan[(i - 1) % len(plan)]
        out.append(generate(depth, (seed * 1000003 + 7919 * (i + 1 - start)) % (2 ** 31),
                            label=lab, dimension=dimension, mod=mod, format=format))
    return out


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _row_field(prompt: str, i: int, fmt: str) -> str:
    """The i-th row as it appears in a (possibly redacted) prompt. Selftest helper."""
    if fmt == "published":
        return prompt.split()[i]
    for line in prompt.splitlines():
        if line.startswith(f"{i}: "):
            return line[len(f"{i}: "):]
    raise AssertionError(f"row {i} not found in prompt")


def _check_redaction(inst: Instance, fmt: str, tag: str) -> None:
    """AMENDMENT 4 checks, run for k in {0, 1, depth//2, depth}."""
    assert redact_prompt(inst, 0) == inst.prompt, f"k=0 changed the prompt {tag}"
    triples = _triples(inst)
    for k in sorted({1, inst.depth // 2, inst.depth} - {0}):
        red = redact_prompt(inst, k)
        early = {q for t in triples[:k] for q in t}
        late = {q for t in triples[k:] for q in t}
        hide = early - late
        for i in range(inst.meta["n"]):
            field = _row_field(red, i, fmt)
            want = inst.meta["rows"][i][0] + _PLACEHOLDER if i in hide else inst.meta["rows"][i]
            assert field == want, f"row {i} at k={k} is {field!r}, expected {want!r} {tag}"
        if hide:
            assert _PLACEHOLDER in red and red != inst.prompt, f"k={k} redacted nothing visible {tag}"
        else:
            assert red == inst.prompt, f"k={k} changed the prompt with nothing to hide {tag}"
    # k=depth: every row the trace touched has lost its verifying digits, so the
    # answer cannot be recomputed from the problem statement at all.
    red = redact_prompt(inst, inst.depth)
    assert red != inst.prompt and _PLACEHOLDER in red, f"k=depth redacted nothing {tag}"
    touched = {q for t in triples for q in t}
    for i in touched:
        assert _row_field(red, i, fmt) == inst.meta["rows"][i][0] + _PLACEHOLDER, \
            f"k=depth left operator row {i} intact {tag}"
    assert redact_prompt(inst, inst.depth + 5) == red, f"k>depth differs from k=depth {tag}"


def _parse_corrupt_published(text: str, dimension: int) -> list[int]:
    """The full mod-tuple a published step reports, un-truncating its lazy emission.
    Selftest helper; also asserts the emission rule still holds."""
    emitted = [int(t) for t in text.split()[6:]]
    assert 1 <= len(emitted) <= dimension - 1, f"published step emits {len(emitted)} sums"
    assert all(v == 0 for v in emitted[:-1]), "published step ran past a nonzero sum"
    assert emitted[-1] != 0 or len(emitted) == dimension - 1, "published step stopped early on a zero"
    return [0] + emitted + [0] * (dimension - 1 - len(emitted))


def _parse_corrupt_ergonomic(text: str) -> tuple[str, list[int], list[int], str]:
    """(head, totals, mods, verdict) of an ergonomic step line. Selftest helper."""
    p0, p1, p2, verdict = text.split(" -> ")
    terms, totals_str = p1.split(" = ")
    return (f"{p0} -> {terms}",
            [int(x) for x in totals_str[1:-1].split(", ")],
            [int(x) for x in p2.split(")")[0][1:].split(", ")],
            verdict)


def _check_corruption(insts: dict, tag: str, cseed: int) -> None:
    """AMENDMENT 6 checks, run in both formats for k in {1, depth//2, depth}."""
    depth = insts["published"].depth
    dimension = insts["published"].meta["dimension"]
    mod = insts["published"].meta["mod"]
    for k in sorted({1, depth // 2, depth} - {0}):
        fabricated = {}
        for fmt, inst in insts.items():
            ftag = f"[{fmt}] k={k} {tag}"
            text, state = corrupt_step(inst, k, cseed)
            true_text, true_state = inst.steps[k - 1], inst.states[k - 1]
            d, e, f_ = _triples(inst)[k - 1]

            assert state != true_state, f"corrupted state == true state {ftag}"
            assert text != true_text, f"corrupted step text == true step text {ftag}"
            assert "\n" not in text, f"corrupted step spans several lines {ftag}"
            # (a)+(b): same shape, same cursor, flipped flag
            assert state == f"{d}-{e}-{f_}/{'F' if true_state.endswith('T') else 'T'}", \
                f"corrupted state {state!r} is not the true state with the flag flipped {ftag}"

            if fmt == "published":
                assert text.split()[:6] == true_text.split()[:6], f"positions/digits changed {ftag}"
                assert text.count(" ") >= 6, f"malformed published step {ftag}"
                mods = _parse_corrupt_published(text, dimension)
                true_mods = _parse_corrupt_published(true_text, dimension)
            else:
                head, totals, mods, verdict = _parse_corrupt_ergonomic(text)
                true_head, true_totals, true_mods, _ = _parse_corrupt_ergonomic(true_text)
                assert head == true_head, f"triple/operands/terms changed {ftag}"
                assert text.startswith(f"({d},{e},{f_}): "), f"step mislabels its triple {ftag}"
                assert text.count(" -> ") == 3 and f" mod {mod} " in text, f"malformed step {ftag}"
                assert verdict in ("yes", "no"), f"bad verdict {verdict!r} {ftag}"
                assert (verdict == "yes") == state.endswith("T"), f"verdict != state flag {ftag}"
                assert [t % mod for t in totals] == mods, f"mods disagree with totals {ftag}"
                assert all(0 <= t <= 3 * (mod - 1) for t in totals), f"implausible total {ftag}"
                # minimal edit: a total moves only where its reported mod had to move
                moved = [j for j in range(dimension) if totals[j] != true_totals[j]]
                assert moved == [j for j in range(dimension) if mods[j] != true_mods[j]], \
                    f"totals and mods disagree about what changed {ftag}"

            assert mods[0] == 0, f"coordinate 0 corrupted {ftag}"
            assert all(m == 0 for m in mods) == state.endswith("T"), f"mods != flag {ftag}"
            if true_state.endswith("T"):                 # yes -> no: exactly one coordinate breaks
                assert sum(1 for m in mods if m != 0) == 1, f"flipped yes is not a one-coordinate no {ftag}"
            else:                                        # no -> yes: the fabricated full match
                assert mods == [0] * dimension, f"flipped no is not an all-zero match {ftag}"
            fabricated[fmt] = mods

            # (c) determinism, and a pure function of (inst, k, seed)
            assert corrupt_step(inst, k, cseed) == (text, state), f"not deterministic {ftag}"
            again = generate(depth, inst.meta["seed"], dimension=dimension, mod=mod, format=fmt)
            assert corrupt_step(again, k, cseed) == (text, state), \
                f"not a pure function of (inst, k, seed) {ftag}"

        # (e) both formats tell the same lie about the same coordinate
        assert fabricated["published"] == fabricated["ergonomic"], \
            f"formats fabricated different sums at k={k} {tag}"

    for bad_k in (0, -1, depth + 1):
        try:
            corrupt_step(insts["published"], bad_k, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"corrupt_step accepted k={bad_k} {tag}")


def _selftest() -> None:
    rng = random.Random(0)
    seen_answers = set()
    for i in range(200):
        depth = rng.choice(DEPTHS + [1, 3, 5, 7, 9, 11, 14, 18, 25, 30])
        seed = rng.randint(0, 10 ** 9)
        dimension = rng.choice([2, 3, 3, 3, 4])
        mod = rng.choice([10, 10, 10, 8, 6])
        insts = {f: generate(depth, seed, dimension=dimension, mod=mod, format=f) for f in FORMATS}
        base = insts["published"]
        ergo = insts["ergonomic"]
        tag = f"(depth={depth} seed={seed} dim={dimension} mod={mod} n={base.meta['n']})"

        # AMENDMENT 5: same instance, same candidate order, same everything but the
        # rendering -- answer, depth, states, solve() and the row list are identical.
        assert base.meta["rows"] == ergo.meta["rows"], f"formats drew different instances {tag}"
        assert base.answer == ergo.answer, f"answer differs across formats {tag}"
        assert base.depth == ergo.depth == depth, f"depth differs across formats {tag}"
        assert base.states == ergo.states, f"states differ across formats {tag}"
        assert solve(base) == solve(ergo) == base.answer, f"solve differs across formats {tag}"
        assert base.prompt != ergo.prompt, f"formats rendered the same prompt {tag}"

        for fmt, inst in insts.items():
            ftag = f"[{fmt}] {tag}"
            got = solve(inst)
            assert got == inst.answer, f"solve mismatch: {got} != {inst.answer} {ftag}"
            assert check(inst, format_cot(inst)), f"check(gold) failed {ftag}"
            wrong = "False" if inst.answer == "True" else "True"
            assert not check(inst, f"Answer: {wrong}"), f"check accepted a wrong answer {ftag}"

            again = generate(depth, seed, dimension=dimension, mod=mod, format=fmt)
            assert again.prompt == inst.prompt and again.steps == inst.steps, f"not deterministic {ftag}"

            assert len(inst.steps) == len(inst.states) == depth, f"len(steps)/len(states) != depth {ftag}"
            assert inst.states[-1].endswith("T") == (inst.answer == "True"), f"state flag != answer {ftag}"
            assert all("\n" not in s for s in inst.steps), f"step spans several lines {ftag}"
            cot = format_cot(inst)
            assert all(cot[a:b] == s for (a, b), s in zip(step_spans(inst), inst.steps)), f"bad spans {ftag}"
            assert cot.splitlines()[-1] == f"Answer: {inst.answer}", f"missing Answer line {ftag}"

        assert all(s.count(" ") >= 6 for s in base.steps), f"malformed published step {tag}"
        for j, (s, st) in enumerate(zip(ergo.steps, ergo.states)):
            d, e, f_ = _triples(ergo)[j]
            assert s.startswith(f"({d},{e},{f_}): "), f"ergonomic step {j} mislabels its triple {tag}"
            assert s.endswith(" -> yes") == st.endswith("T"), f"ergonomic verdict != state {tag}"
            assert s.count(" -> ") == 3 and f" mod {mod} " in s, f"malformed ergonomic step {tag}"
            assert (j == depth - 1) or s.endswith(" -> no"), f"enumeration ran past a hit {tag}"
        assert ergo.steps[-1].endswith(" -> yes") == (ergo.answer == "True"), f"last verdict != answer {tag}"

        if i < 20:                                     # AMENDMENT 4, both formats
            for fmt, inst in insts.items():
                _check_redaction(inst, fmt, f"[{fmt}] {tag}")
            _check_corruption(insts, tag, (seed * 7919 + 13) % (10 ** 6))   # AMENDMENT 6

        seen_answers.add(base.answer)
    assert seen_answers == {"True", "False"}, "generator produced only one label"
    assert REDACTION_MEANINGFUL is True

    # AMENDMENT 6 spot checks. The published exemplar corrupts too, and the seed matters
    # wherever there is a choice to make: flipping a `yes` picks which coordinate breaks
    # and on what residue, while flipping a `no` is forced (the only consistent lie is the
    # all-zero match).
    ex0 = _published_instance()
    text, state = corrupt_step(ex0, 1, 0)
    assert state == "0-1-4/F" and ex0.states[0] == "0-1-4/T", "published exemplar did not flip"
    assert text != ex0.steps[0] and text.split()[:6] == ex0.steps[0].split()[:6], \
        "published exemplar corruption touched the action text"
    for fmt in FORMATS:
        pos = generate(4, 7, label=True, format=fmt)
        assert pos.states[-1].endswith("T")
        assert len({corrupt_step(pos, pos.depth, s) for s in range(30)}) > 1, \
            f"corrupt_step ignores its seed [{fmt}]"
        neg = generate(4, 7, label=False, format=fmt)
        assert corrupt_step(neg, 2, 0)[1].endswith("T"), f"flipped `no` did not set the flag [{fmt}]"

    # tolerant-parser spot checks
    inst = generate(12, 42)
    assert check(inst, f"Answer: {inst.answer}")
    assert check(inst, f"Answer: {inst.answer}.")
    assert check(inst, f"Answer:  {inst.answer.lower()} ")
    assert check(inst, format_cot(inst).splitlines()[0])          # no Answer: line -> trailing 'A <label>'
    assert check(inst, f"Answer: {'False' if inst.answer == 'True' else 'True'}\nAnswer: {inst.answer}")
    assert check(inst, "Answer: " + ("yes" if inst.answer == "True" else "no"))
    assert not check(inst, "Answer: maybe")
    assert not check(inst, "")

    # knob sanity
    assert generate(5, 1, label=True).answer == "True"
    assert generate(5, 1, label=False).answer == "False"
    assert generate(5, 1, n=9).meta["n"] == 9
    assert len(generate(6, 3, dimension=5).steps) == 6
    assert generate(5, 1, label=True, format="ergonomic").answer == "True"
    for bad in (lambda: generate(0, 0), lambda: generate(3, 0, dimension=1),
                lambda: generate(3, 0, mod=11), lambda: generate(3, 0, n=5),
                lambda: generate(3, 0, format="prose"), lambda: exemplars(2, 0, format="prose"),
                lambda: redact_prompt(generate(3, 0), -1)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    ex = exemplars(3, 0)
    assert len(ex) == 3 and ex[0].meta["published_exemplar"]
    assert ex[0].prompt == "901 457 892 282 752 384 P"
    assert " ".join(ex[0].steps) == "0- 1- 4- 1 7 2 0 0"
    assert ex[0].answer == "True" and solve(ex[0]) == "True"
    assert {e.answer for e in ex} == {"True", "False"}
    for e in ex:
        assert e.prompt and format_cot(e) and solve(e) == e.answer
        assert e.meta["format"] == "published"
        assert redact_prompt(e, 0) == e.prompt
        assert _PLACEHOLDER in redact_prompt(e, e.depth)

    exe = exemplars(3, 0, format="ergonomic")
    assert len(exe) == 3 and not any(e.meta["published_exemplar"] for e in exe)
    assert {e.answer for e in exe} == {"True", "False"}
    for e in exe:
        assert e.meta["format"] == "ergonomic"
        assert e.prompt.startswith("Here are ") and solve(e) == e.answer
        assert format_cot(e).splitlines()[-1] == f"Answer: {e.answer}"
        assert format_cot(e).splitlines()[-2].startswith(("The triple ", "No triple "))
    assert exemplars(3, 0, format="ergonomic") == exe, "exemplars not deterministic"

    print("selftest OK: 200 random instances over depth/dimension/mod, EACH RENDERED IN BOTH "
          "FORMATS\n             (solve==answer, check(gold)=True, check(wrong)=False, deterministic, "
          "len(steps)==len(states)==depth,\n             step_spans exact, both labels seen)")
    print("             + published vs ergonomic identical in rows/answer/depth/states/solve")
    print("             + redact_prompt, both formats, k in {0, 1, depth//2, depth} on 20 instances")
    print("             + corrupt_step, both formats, k in {1, depth//2, depth} on 20 instances "
          "(flag flipped,\n               action text intact, lazy emission/mod tuple consistent, "
          "same lie in both formats, deterministic)")
    print("             + tolerant-parser spot checks + knob/validation checks")
    print("             + exemplars(3, 0): [0] is published_trace.txt verbatim, labels mixed; "
          "exemplars(3, 0, format='ergonomic') all generated")


def _render_demo(inst: Instance, header: str) -> None:
    print(f"===== {header} depth={inst.depth} seed={inst.meta['seed']} n={inst.meta['n']} "
          f"dimension={inst.meta['dimension']} mod={inst.meta['mod']} "
          f"format={inst.meta['format']} =====")
    print("--- prompt ---")
    print(inst.prompt)
    print("--- gold CoT ---")
    print(format_cot(inst))
    print()


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, 1000 + i)
            _render_demo(inst, "")

    # AMENDMENT 5: the same instance under each format, side by side.
    print("##### the same instance (depth=4, seed=7) in each format #####\n")
    for fmt in FORMATS:
        _render_demo(generate(4, 7, format=fmt), f"[{fmt}]")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--demo" in sys.argv:
        _demo()
    else:
        print("usage: python3 task.py [--selftest|--demo]")
