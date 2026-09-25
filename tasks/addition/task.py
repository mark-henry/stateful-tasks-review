"""
addition -- multi-digit column addition with the Nye et al. (2021) carry scratchpad.

Published format (Nye et al., "Show Your Work: Scratchpads for Intermediate Computation with
Language Models", arXiv:2112.00114, Figure 2), reproduced byte-for-byte by this module:

    Input:
    2 9 + 5 7

    Target:
    <scratch>
    2 9 + 5 7 ,  C: 0
    2 + 5 , 6 C: 1  # added 9 + 7 = 6 carry 1
    , 8 6 C: 0  # added 2 + 5 + 1 = 8 carry 0
    0 8 6
    </scratch>
    8 6

The '#' comments are shown in the figure but the caption states they are not part of the
target, so they are OFF by default and available through the `comments` knob.

AMENDMENT 5 adds an SFT-free rendering, `format="ergonomic"` (the published format stays the
default). Same instance distribution, same answers, plain-language prompt and a schoolbook
right-to-left trace:

    What is 29 + 57?

    ones: 9 + 7 + 0 = 16 -> write 6, carry 1
    tens: 2 + 5 + 1 = 8 -> write 8, carry 0
    carry out: 0
    sum: 86
    Answer: 86

depth == number of digit columns == number of digits in each operand, and len(steps) == depth
(one step per digit column) in BOTH formats. See README.md for the format decision, the
ergonomic variant, depth semantics, sourcing and caveats. Pure python + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import random
import re
import sys


# --------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------

FORMATS = ("published", "ergonomic")


@dataclass
class Instance:
    prompt: str          # problem statement only, published wording ("Input:\n2 9 + 5 7")
    steps: list          # gold trace, one element per digit column, published format
    states: list         # state after each step: "<result digits so far>|<carry out>"
    answer: str          # exact-match target: plain integer, no spaces ("86")
    depth: int           # number of digit columns == len(steps)
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a single integer written with no spaces, commas, or other separators (e.g. 86)"

# Justified in README.md: 2 is a floor anchor both conditions should ace; no-CoT is expected to
# fall apart between 6 and 8 digits; 24 is where the serial carry chain should break CoT too
# while keeping the (quadratic) published trace under ~2.5k tokens.
DEPTHS = [2, 4, 6, 8, 12, 16, 24]

KNOBS = {
    "format": (
        "published",
        "'published' = the Nye et al. Figure 2 <scratch> notation (default, so the 'as "
        "published' rows stay reproducible); 'ergonomic' = the SFT-free variant of AMENDMENT 5: "
        "plain-language prompt ('What is 828306 + 181525?') and a schoolbook right-to-left "
        "trace, one line per column ('ones: 6 + 5 + 0 = 11 -> write 1, carry 1'), then "
        "'carry out: <c>' and 'sum: <answer>'. Instance distribution, answer, check, depth "
        "semantics and solve are identical across formats",
    ),
    "comments": (
        False,
        "append the figure's clarifying '  # added a + b + c = d carry e' comment to each column "
        "line; Figure 2 shows them but its caption says they are not part of the target. "
        "format='published' only -- ignored by the ergonomic format, whose column lines already "
        "spell out the addition",
    ),
    "carry_rate": (
        None,
        "None = digits sampled uniformly (a column carries 45% of the time with carry-in 0, "
        "55% with carry-in 1); a float in [0,1] "
        "forces each column to carry with that probability, so 1.0 gives a maximal carry chain "
        "and 0.0 gives a carry-free instance",
    ),
    "force_carry_out": (
        False,
        "force the most significant column to carry, so the consolidation line prepends a 1 and "
        "the answer has depth+1 digits",
    ),
}


# --------------------------------------------------------------------------------------
# published-format rendering
# --------------------------------------------------------------------------------------

def _spaced(seq) -> str:
    """Digits, space separated, as the published format writes them."""
    return " ".join(str(d) for d in seq)


def _line(a_rem: str, b_rem: str, result: str, carry: int) -> str:
    """One scratchpad line: unprocessed columns, ',', result so far, 'C: <carry>'.

    Reproduces both published spellings: while columns remain the line keeps the 'a + b' part
    (and an empty result yields the figure's double space before 'C:'); once both operands are
    exhausted the 'a + b' part and its trailing space disappear entirely (", 8 6 C: 0").
    """
    if a_rem or b_rem:
        return f"{a_rem} + {b_rem} , {result} C: {carry}"
    return f", {result} C: {carry}"


def _comment(da: int, db: int, carry_in: int, digit: int, carry_out: int) -> str:
    """The figure's clarifying comment; the carry_in term is written only when nonzero."""
    body = f"{da} + {db}"
    if carry_in:
        body += f" + {carry_in}"
    return f"  # added {body} = {digit} carry {carry_out}"


# --------------------------------------------------------------------------------------
# ergonomic-format rendering (AMENDMENT 5)
# --------------------------------------------------------------------------------------

_COLUMN_NAMES = ("ones", "tens", "hundreds", "thousands")


def _column_name(i: int) -> str:
    """Name of column i, counted from the right, 0-based. Place-value words run out after
    'thousands'; past that the columns are numbered 1-based from the right ('column 5')."""
    if i < len(_COLUMN_NAMES):
        return _COLUMN_NAMES[i]
    return f"column {i + 1}"


def _erg_line(i: int, da: int, db: int, carry_in: int, total: int, digit: int, carry_out: int) -> str:
    """One schoolbook column line: 'ones: 6 + 5 + 0 = 11 -> write 1, carry 1'. The carry-in is
    always written, even when it is 0, so every line has the same shape."""
    return (f"{_column_name(i)}: {da} + {db} + {carry_in} = {total} "
            f"-> write {digit}, carry {carry_out}")


# --------------------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------------------

def _build(a_str: str, b_str: str, comments: bool = False, fmt: str = "published",
           meta: dict | None = None) -> Instance:
    """Render one instance. a_str and b_str must be equal length. `fmt` selects the trace and
    prompt notation; everything the harness scores (answer, depth, states) is format-agnostic."""
    if len(a_str) != len(b_str):
        raise ValueError("operands must have the same number of digits")
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS!r}, got {fmt!r}")
    depth = len(a_str)

    a_rev = [int(c) for c in reversed(a_str)]
    b_rev = [int(c) for c in reversed(b_str)]

    steps: list = []
    states: list = []
    result: list = []  # digits emitted so far, most significant first
    carry = 0
    for i in range(depth):
        da, db = a_rev[i], b_rev[i]
        total = da + db + carry
        digit, carry_out = total % 10, total // 10
        result.insert(0, digit)
        if fmt == "published":
            # columns not yet consumed are the leading digits of each operand
            a_rem = _spaced(a_str[: depth - 1 - i])
            b_rem = _spaced(b_str[: depth - 1 - i])
            line = _line(a_rem, b_rem, _spaced(result), carry_out)
            if comments:
                line += _comment(da, db, carry, digit, carry_out)
        else:
            line = _erg_line(i, da, db, carry, total, digit, carry_out)
        steps.append(line)
        states.append("".join(str(d) for d in result) + "|" + str(carry_out))
        carry = carry_out

    answer_digits = ([carry] if carry else []) + result
    answer = "".join(str(d) for d in answer_digits)

    m = {
        "a": a_str,
        "b": b_str,
        "digits": depth,
        "format": fmt,
        "carry_out": carry,
    }
    if fmt == "published":
        prompt = "Input:\n" + _spaced(a_str) + " + " + _spaced(b_str)
        m.update({
            "comments": comments,
            "setup_line": _line(_spaced(a_str), _spaced(b_str), "", 0),
            # consolidation line: the final carry is written out even when it is 0 ("0 8 6")
            "final_line": _spaced([carry] + result),
            "answer_line": _spaced(answer_digits),
        })
    else:
        prompt = f"What is {a_str} + {b_str}?"
        m.update({
            "carry_line": f"carry out: {carry}",
            "sum_line": f"sum: {answer}",
        })
    if meta:
        m.update(meta)
    return Instance(prompt=prompt, steps=steps, states=states, answer=answer, depth=depth, meta=m)


# --------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------

def _column_pair(rng: random.Random, carry_in: int, top: bool, want_carry):
    """Pick (digit_a, digit_b) for one column. `top` forbids a leading zero in either operand;
    `want_carry` in {None, True, False} constrains whether the column carries out."""
    lo = 1 if top else 0
    pairs = [(x, y) for x in range(lo, 10) for y in range(lo, 10)]
    if want_carry is not None:
        filtered = [(x, y) for (x, y) in pairs if ((x + y + carry_in) >= 10) is bool(want_carry)]
        if filtered:
            pairs = filtered
    return pairs[rng.randrange(len(pairs))]


def generate(depth: int, seed: int, **knobs) -> Instance:
    """Pure function of (depth, seed, knobs). depth = number of digit columns; both operands
    are exactly `depth` digits with no leading zero, so len(steps) == depth exactly."""
    if depth < 1:
        raise ValueError("depth must be >= 1")
    unknown = set(knobs) - set(KNOBS)
    if unknown:
        raise TypeError(f"unknown knobs: {sorted(unknown)}")
    fmt = knobs.get("format", KNOBS["format"][0])
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS!r}, got {fmt!r}")
    comments = bool(knobs.get("comments", KNOBS["comments"][0]))
    carry_rate = knobs.get("carry_rate", KNOBS["carry_rate"][0])
    force_carry_out = bool(knobs.get("force_carry_out", KNOBS["force_carry_out"][0]))
    if carry_rate is not None and not (0.0 <= float(carry_rate) <= 1.0):
        raise ValueError("carry_rate must be in [0, 1] or None")

    rng = random.Random(f"addition|{depth}|{seed}")

    a_rev: list = []
    b_rev: list = []
    carry = 0
    for i in range(depth):
        top = i == depth - 1
        if top and force_carry_out:
            want = True
        elif carry_rate is not None:
            want = rng.random() < float(carry_rate)
        else:
            want = None
        da, db = _column_pair(rng, carry, top, want)
        a_rev.append(da)
        b_rev.append(db)
        carry = 1 if (da + db + carry) >= 10 else 0

    a_str = "".join(str(d) for d in reversed(a_rev))
    b_str = "".join(str(d) for d in reversed(b_rev))
    return _build(
        a_str,
        b_str,
        comments=comments,
        fmt=fmt,
        meta={"seed": seed, "carry_rate": carry_rate, "force_carry_out": force_carry_out},
    )


# --------------------------------------------------------------------------------------
# reference solver (independent of generate()/_build()'s column loop)
# --------------------------------------------------------------------------------------

_DIGIT_GROUP = r"[0-9](?:[ ]?[0-9])*"          # "57" or the published "5 7"
_OPERANDS = re.compile(rf"({_DIGIT_GROUP})\s*\+\s*({_DIGIT_GROUP})")


def solve(inst: Instance) -> str:
    """Read the two operands back out of the rendered prompt and add them with python ints.
    Touches neither the per-column loop nor the digit lists that produced the gold trace.
    Format-agnostic: one regex reads both the spaced published prompt ('2 9 + 5 7') and the
    ergonomic one ('What is 29 + 57?')."""
    m = _OPERANDS.search(inst.prompt)
    if m is None:
        raise ValueError(f"no 'a + b' found in prompt: {inst.prompt!r}")
    a = int(m.group(1).replace(" ", ""))
    b = int(m.group(2).replace(" ", ""))
    return str(a + b)


# --------------------------------------------------------------------------------------
# answer extraction
# --------------------------------------------------------------------------------------

_ANSWER_MARKER = re.compile(r"answer\s*:", re.IGNORECASE)
_DIGIT_RUN = re.compile(r"\d+")


def _normalize(text) -> str | None:
    """Strip the published digit spacing and the usual decoration; return a bare integer."""
    if text is None:
        return None
    cand = text.strip().rstrip(".").strip()
    cand = re.sub(r"[\s,_]", "", cand)
    cand = cand.lstrip("+")
    if not cand:
        return None
    if not cand.isdigit():
        # last resort: the trailing digit run (e.g. "the sum is 86")
        runs = _DIGIT_RUN.findall(cand)
        if not runs:
            return None
        cand = runs[-1]
    return cand.lstrip("0") or "0"


def check(inst: Instance, completion: str) -> bool:
    """Extract the last `Answer:` line; if there is none, fall back to the last non-empty line.
    Exact match after normalization (digit spacing, commas and leading zeros are forgiven)."""
    if not completion:
        return False
    lines = completion.splitlines()
    cand = None
    for line in reversed(lines):
        hits = list(_ANSWER_MARKER.finditer(line))
        if hits:
            cand = line[hits[-1].end():]
            break
    if cand is None:
        for line in reversed(lines):
            if line.strip():
                cand = line
                break
    got = _normalize(cand)
    want = _normalize(inst.answer)
    return got is not None and got == want


# --------------------------------------------------------------------------------------
# gold completion
# --------------------------------------------------------------------------------------

def format_cot(inst: Instance) -> str:
    """Gold trace in the instance's format, then the harness-imposed `Answer:` line.

    published: the Figure 2 <scratch> block verbatim.
    ergonomic: the column lines, then `carry out: <c>` and `sum: <answer>`.
    """
    if inst.meta.get("format", "published") == "ergonomic":
        body = list(inst.steps) + [inst.meta["carry_line"], inst.meta["sum_line"]]
    else:
        body = ["<scratch>", inst.meta["setup_line"]]
        body += list(inst.steps)
        body += [inst.meta["final_line"], "</scratch>", inst.meta["answer_line"]]
    return "\n".join(body) + "\nAnswer: " + inst.answer


def step_spans(inst: Instance):
    """Char offsets of each step within format_cot() output (kept for later ablations)."""
    text = format_cot(inst)
    spans = []
    pos = 0
    for step in inst.steps:
        start = text.index(step, pos)
        end = start + len(step)
        spans.append((start, end))
        pos = end
    return spans


# --------------------------------------------------------------------------------------
# exemplars
# --------------------------------------------------------------------------------------

PUBLISHED_A = "29"
PUBLISHED_B = "57"


def published_exemplar(comments: bool = False, fmt: str = "published") -> Instance:
    """Nye et al. 2021 Figure 2's instance, 29 + 57. With fmt="published" (and comments=True)
    this renders the figure byte-for-byte; the default renders the true target the caption
    describes (comments stripped). With fmt="ergonomic" the same 29 + 57 instance is rendered
    in the AMENDMENT 5 notation, so the exemplar block still opens on the published example."""
    return _build(PUBLISHED_A, PUBLISHED_B, comments=comments, fmt=fmt,
                  meta={"source": "Nye et al. 2021, arXiv:2112.00114, Figure 2",
                        "published": fmt == "published"})


def exemplars(k: int, seed: int, **knobs) -> list:
    """exemplars(k, seed)[0] is the Figure 2 instance (29 + 57), rendered in the requested
    format; the rest are generated at a modest depth (3-4 columns), one of them forced to carry
    out of the top column so the leading-carry case is demonstrated.

    `**knobs` are forwarded to `generate`, so the harness can ask for exemplars in the same
    format (and with the same knobs) as the instance being solved."""
    if k <= 0:
        return []
    unknown = set(knobs) - set(KNOBS)
    if unknown:
        raise TypeError(f"unknown knobs: {sorted(unknown)}")
    fmt = knobs.get("format", KNOBS["format"][0])
    comments = bool(knobs.get("comments", KNOBS["comments"][0]))
    out = [published_exemplar(comments=comments, fmt=fmt)]
    for i in range(1, k):
        depth = 3 + (i % 2)
        rest = dict(knobs)
        rest["force_carry_out"] = i == 1 or bool(rest.get("force_carry_out", False))
        out.append(generate(depth, seed * 1000 + i, **rest))
    return out


# --------------------------------------------------------------------------------------
# prompt redaction (AMENDMENT 4)
# --------------------------------------------------------------------------------------

# Both operands are consumed by every column step, so prompt redaction is not meaningful --
# in either format. (Defined here, above __main__, so --selftest can exercise it.)
REDACTION_MEANINGFUL = False


def redact_prompt(inst, k: int) -> str:
    return inst.prompt


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------

_FIGURE_TARGET = "\n".join([
    "<scratch>",
    "2 9 + 5 7 ,  C: 0",
    "2 + 5 , 6 C: 1  # added 9 + 7 = 6 carry 1",
    ", 8 6 C: 0  # added 2 + 5 + 1 = 8 carry 0",
    "0 8 6",
    "</scratch>",
    "8 6",
])


def _published_trace_target():
    """The verbatim target block parsed out of published_trace.txt, if the file is present."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_trace.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    body = [ln for ln in lines if not ln.startswith("#")]
    if "<scratch>" not in body or "</scratch>" not in body:
        return None
    start = body.index("<scratch>")
    end = body.index("</scratch>")
    return "\n".join(body[start:end + 1] + [body[end + 1]])


def _selftest() -> int:
    failures = []

    def fail(tag, *info):
        failures.append((tag, info))

    # 1. the published figure, byte for byte
    fig = format_cot(published_exemplar(comments=True))
    fig_target = fig.rsplit("\nAnswer: ", 1)[0]
    if fig_target != _FIGURE_TARGET:
        fail("figure mismatch", repr(fig_target))
    on_disk = _published_trace_target()
    if on_disk is None:
        print("  note: published_trace.txt not parseable/found; skipped on-disk verbatim check")
    elif on_disk != _FIGURE_TARGET:
        fail("published_trace.txt mismatch", repr(on_disk))
    if published_exemplar().answer != "86":
        fail("published answer", published_exemplar().answer)

    # 2. 200 random (depth, seed) pairs
    n = 200
    rng = random.Random(20260916)
    knob_choices = [
        {},
        {"carry_rate": 1.0},
        {"carry_rate": 0.0},
        {"force_carry_out": True},
        {"comments": True},
        {"format": "ergonomic"},
        {"format": "ergonomic", "carry_rate": 1.0},
        {"format": "ergonomic", "carry_rate": 0.0},
        {"format": "ergonomic", "force_carry_out": True},
        {"format": "ergonomic", "comments": True},
    ]
    for i in range(n):
        depth = rng.randint(1, 30)
        seed = rng.randrange(10 ** 9)
        knobs = knob_choices[i % len(knob_choices)]
        fmt = knobs.get("format", "published")
        inst = generate(depth, seed, **knobs)

        if inst.meta["format"] != fmt:
            fail("format knob ignored", fmt, inst.meta["format"])
            continue
        if solve(inst) != inst.answer:
            fail("solve != answer", inst.meta["a"], inst.meta["b"], solve(inst), inst.answer)
            continue
        if len(inst.steps) != depth or len(inst.states) != depth or inst.depth != depth:
            fail("depth mismatch", depth, len(inst.steps), len(inst.states), inst.depth)
            continue
        cot = format_cot(inst)
        if not check(inst, cot):
            fail("check(format_cot) false", inst.meta["a"], inst.meta["b"])
            continue
        wrong = str(int(inst.answer) + 1 + rng.randrange(9))
        if check(inst, "Answer: " + wrong):
            fail("check accepted wrong answer", inst.answer, wrong)
            continue
        if fmt == "published" and not check(inst, "Answer: " + inst.meta["answer_line"]):
            fail("check rejected published digit spacing", inst.meta["answer_line"])
            continue
        if not check(inst, inst.answer):            # bare no-CoT completion
            fail("check rejected bare answer", inst.answer)
            continue
        again = generate(depth, seed, **knobs)
        if (again.prompt, again.steps, again.states, again.answer, again.depth) != (
            inst.prompt, inst.steps, inst.states, inst.answer, inst.depth
        ):
            fail("nondeterministic", depth, seed, knobs)
            continue
        spans = step_spans(inst)
        if len(spans) != depth or any(cot[s:e] != st for st, (s, e) in zip(inst.steps, spans)):
            fail("step_spans", depth, seed)
            continue
        # knob semantics
        if knobs.get("force_carry_out") and inst.meta["carry_out"] != 1:
            fail("force_carry_out ignored", inst.meta["a"], inst.meta["b"])
        if knobs.get("carry_rate") == 0.0 and inst.meta["carry_out"] != 0:
            fail("carry_rate=0 carried out", inst.meta["a"], inst.meta["b"])
        if fmt == "published":
            if knobs.get("comments") and "#" not in inst.steps[0]:
                fail("comments knob ignored", inst.steps[0])
            if not knobs.get("comments") and "#" in inst.steps[0]:
                fail("unexpected comment", inst.steps[0])
        else:
            # ergonomic: plain prompt, one schoolbook line per column, then carry out + sum
            if inst.prompt != f"What is {inst.meta['a']} + {inst.meta['b']}?":
                fail("ergonomic prompt", inst.prompt)
            if "#" in inst.steps[0] or not inst.steps[0].startswith("ones: "):
                fail("ergonomic first step", inst.steps[0])
            tail = cot.splitlines()[depth:]
            if tail != [f"carry out: {inst.meta['carry_out']}",
                        f"sum: {inst.answer}", f"Answer: {inst.answer}"]:
                fail("ergonomic tail", tail)
            if not check(inst, "\n".join(cot.splitlines()[:-1])):
                fail("ergonomic trace without Answer line not scored", inst.answer)

    # 3. AMENDMENT 5: the two formats differ ONLY in prompt wording and trace notation.
    #    Instance distribution (the sampled operands), answer, check, depth and solve identical.
    rng2 = random.Random(20260918)
    probes = ["Answer: 0", "Answer: 1", "the sum is 12", "", "nonsense", "<scratch>\n7\n</scratch>"]
    for _ in range(60):
        depth = rng2.randint(1, 30)
        seed = rng2.randrange(10 ** 9)
        extra = [{}, {"carry_rate": 1.0}, {"carry_rate": 0.0}, {"force_carry_out": True}]
        kn = extra[rng2.randrange(len(extra))]
        pub = generate(depth, seed, **kn)
        erg = generate(depth, seed, format="ergonomic", **kn)
        if (pub.meta["a"], pub.meta["b"]) != (erg.meta["a"], erg.meta["b"]):
            fail("distribution differs across formats", depth, seed, pub.meta["a"], erg.meta["a"])
            continue
        if pub.answer != erg.answer or pub.depth != erg.depth != depth:
            fail("answer/depth differ across formats", pub.answer, erg.answer)
            continue
        if pub.states != erg.states:
            fail("states differ across formats", depth, seed)
            continue
        if len(erg.steps) != depth:
            fail("ergonomic step count", depth, len(erg.steps))
            continue
        if solve(pub) != solve(erg) or solve(erg) != erg.answer:
            fail("solve differs across formats", solve(pub), solve(erg))
            continue
        for probe in probes + ["Answer: " + pub.answer]:
            if check(pub, probe) != check(erg, probe):
                fail("check differs across formats", probe)
                break
        if pub.prompt == erg.prompt or format_cot(pub) == format_cot(erg):
            fail("formats not actually different", depth, seed)
        # redaction is unchanged and format-agnostic (AMENDMENT 4: not meaningful here)
        for k in (0, 1, depth // 2, depth):
            if redact_prompt(pub, k) != pub.prompt or redact_prompt(erg, k) != erg.prompt:
                fail("redact_prompt changed the prompt", k)
                break
    if REDACTION_MEANINGFUL is not False:
        fail("REDACTION_MEANINGFUL should stay False")

    # 4. exemplars render, in both formats, and forward the format knob
    for fmt in FORMATS:
        ex = exemplars(3, 0, format=fmt) if fmt != "published" else exemplars(3, 0)
        if len(ex) != 3:
            fail("exemplars length", fmt, len(ex))
            continue
        if (ex[0].meta["a"], ex[0].meta["b"]) != (PUBLISHED_A, PUBLISHED_B):
            fail("exemplars[0] is not the 29+57 instance", fmt, ex[0].meta["a"])
        if (ex[0].meta.get("published") is True) != (fmt == "published"):
            fail("exemplars[0] published flag", fmt, ex[0].meta.get("published"))
        for e in ex:
            if e.meta["format"] != fmt:
                fail("exemplar format not forwarded", fmt, e.meta["format"])
            if not check(e, format_cot(e)) or solve(e) != e.answer:
                fail("exemplar inconsistent", fmt, e.meta.get("a"), e.meta.get("b"))
    if exemplars(0, 0) != []:
        fail("exemplars(0) not empty")
    try:
        generate(3, 0, format="schoolbook")
    except ValueError:
        pass
    else:
        fail("bad format value accepted")

    # 5. hand-computed maximal carry chain, both formats
    hand = _build("999", "111")
    if hand.answer != "1110" or hand.states != ["0|1", "10|1", "110|1"]:
        fail("hand check", hand.answer, hand.states)
    hand_e = _build("999", "111", fmt="ergonomic")
    want_e = "\n".join([
        "What is 999 + 111?",
        "ones: 9 + 1 + 0 = 10 -> write 0, carry 1",
        "tens: 9 + 1 + 1 = 11 -> write 1, carry 1",
        "hundreds: 9 + 1 + 1 = 11 -> write 1, carry 1",
        "carry out: 1",
        "sum: 1110",
        "Answer: 1110",
    ])
    if hand_e.prompt + "\n" + format_cot(hand_e) != want_e:
        fail("ergonomic hand check", repr(hand_e.prompt + "\n" + format_cot(hand_e)))
    if hand_e.states != hand.states or hand_e.answer != hand.answer:
        fail("ergonomic hand state/answer", hand_e.states, hand_e.answer)
    # the AMENDMENT 5 worked example, verbatim
    worked = _build("828306", "181525", fmt="ergonomic")
    if worked.prompt != "What is 828306 + 181525?":
        fail("worked prompt", worked.prompt)
    if worked.steps[0] != "ones: 6 + 5 + 0 = 11 -> write 1, carry 1":
        fail("worked first line", worked.steps[0])
    if worked.answer != "1009831" or worked.steps[-1] != "column 6: 8 + 1 + 1 = 10 -> write 0, carry 1":
        fail("worked last line/answer", worked.steps[-1], worked.answer)
    if [_column_name(i) for i in range(6)] != [
            "ones", "tens", "hundreds", "thousands", "column 5", "column 6"]:
        fail("column names", [_column_name(i) for i in range(6)])

    if failures:
        print(f"FAIL: {len(failures)} problem(s)")
        for tag, info in failures[:10]:
            print("  -", tag, info)
        return 1
    print(f"PASS: published Figure 2 reproduced byte-for-byte; {n}/{n} random instances OK")
    print("      (solve==answer, len(steps)==len(states)==depth, check(gold)==True,")
    print("       check(wrong)==False, deterministic, step_spans aligned, exemplars(3,0) render)")
    print("      both formats; 60 paired instances agree on operands, states, answer, depth,")
    print("      solve and check; redact_prompt is the identity in both (REDACTION_MEANINGFUL=False)")
    return 0


def _demo() -> None:
    out = []
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for k in range(3):
            seed = 1000 * depth + k
            inst = generate(depth, seed)
            out.append(f"=== depth={depth} seed={seed} (a={inst.meta['a']}, b={inst.meta['b']}) ===")
            out.append("--- prompt ---")
            out.append(inst.prompt)
            out.append("--- gold completion (format_cot) ---")
            out.append(format_cot(inst))
            out.append("")
    # AMENDMENT 5: one instance per format, same (depth, seed), so the two renderings of the
    # SAME instance sit side by side.
    depth, seed = 6, 6000
    for fmt in FORMATS:
        inst = generate(depth, seed, format=fmt)
        out.append(f"=== format={fmt} depth={depth} seed={seed} "
                   f"(a={inst.meta['a']}, b={inst.meta['b']}) ===")
        out.append("--- prompt ---")
        out.append(inst.prompt)
        out.append("--- gold completion (format_cot) ---")
        out.append(format_cot(inst))
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        _demo()
    else:
        print("usage: python3 task.py [--selftest|--demo]")
