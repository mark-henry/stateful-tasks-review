"""
cellular_automaton -- elementary 1-D cellular automaton on a finite ring.

State: one row of `width` cells, each 0 or 1, on a periodic (circular) lattice.
Each step is one generation: every cell is simultaneously replaced by the value
the rule table assigns to its (left, self, right) neighbourhood. Depth = number
of generations simulated. Answer = the final row, as a bit string.

Why this task: predicting the state of elementary cellular automaton Rule 110 is
P-complete (Neary, T. & Woods, D., "P-completeness of Cellular Automaton Rule
110", ICALP 2006, LNCS 4051, doi:10.1007/11786986_13, building on Cook, M.,
"Universality in Elementary Cellular Automata", Complex Systems 15(1), 2004).
So for the default rule there is no known way to leap to generation t without
passing through the intervening generations: the scratchpad has to carry the
row. The rule number is a knob precisely so the contrast is available: Rule 90
is additive over GF(2) (x_{t+1,i} = x_{t,i-1} XOR x_{t,i+1}), hence one GF(2)
matrix power, shortcuttable by repeated squaring (Wolfram, S., "Universality and
complexity in cellular automata", Physica D 10, 1984).

Sourcing: no published LLM-facing CoT/scratchpad trace of a 1-D elementary CA
exists (see SOURCING.md); nothing vendored was usable, so generator, solver and
format are written from scratch. See README.md, "Format decision".

Trace format: the default is a per-cell scratchpad (`format="cells"`): for each
generation, one line per cell naming that cell's three neighbours by index AND
value, the neighbourhood they spell, the rule-table lookup it resolves to, and a
running row that grows by exactly one bit per line; a final `row:` line restates
the completed running row. One step is still one generation (a multi-line step,
allowed by AMENDMENT 3). The original one-row-per-generation format is kept as
`format="rows"`.
"""

from __future__ import annotations

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


ANSWER_FORMAT = "a bit string with one character per cell, each 0 or 1, no spaces (e.g. 01101001)"

# AMENDMENT 4 (prompt redaction). The only per-instance state in the prompt is the generation-0
# row; the rule table, the ring size and the boundary condition are static material every
# remaining generation consumes. So redaction is all-or-nothing at k >= 1. See README :: Redaction.
REDACTION_MEANINGFUL = True
REDACTION_PLACEHOLDER = "[\u2026]"

# Justified in README.md :: DEPTHS. Retuned against the second-chance measurements (DeepSeek
# 0.78/0.54/0.36 and Qwen-9B 0.59/0.44/0.12 at depths 1/2/4), which put the knee at depth 2-6,
# and against the ~358 tokens/generation cost of the verbose per-cell format. 1, 2 and 4 are
# retained so the grid stays comparable with that run. Not 16: 216 of the 256 width-8 rule-110
# orbits have period 16, so at depth 16 generation 16 == generation 0 for ~20% of instances.
DEPTHS = [1, 2, 3, 4, 6, 10]

KNOBS = {
    "rule": (110, "Wolfram rule number, 0-255; 110 is P-complete, 90 is linear over GF(2)"),
    "width": (8, "number of cells in the ring; answer space is 2**width"),
    "format": (
        "cells",
        "trace format: 'cells' names each cell's neighbours by index and value, does the "
        "rule lookup, and carries a running row that grows one bit per line; 'rows' shows "
        "only the row (one line per generation, the batch-1 format)",
    ),
    "avoid_absorbing": (
        True,
        "resample the initial row if the orbit hits a fixed point at or before the "
        "final generation (which would make the deep end of the sweep trivial)",
    ),
}

FORMATS = ("cells", "rows")

# Wolfram's standard presentation order for the 8 neighbourhoods: 111 down to 000.
_NEIGHBOURHOODS = [(p >> 2 & 1, p >> 1 & 1, p & 1) for p in range(7, -1, -1)]


def _rule_table_block(rule: int) -> str:
    """The 8 neighbourhood -> bit mappings, published style: 111 first, 000 last."""
    cells = [
        "".join(str(b) for b in nb) + " -> " + str((rule >> ((nb[0] << 2) | (nb[1] << 1) | nb[2])) & 1)
        for nb in _NEIGHBOURHOODS
    ]
    return "   ".join(cells[:4]) + "\n" + "   ".join(cells[4:])


def _next_row(row: tuple[int, ...], rule: int) -> tuple[int, ...]:
    w = len(row)
    return tuple(
        (rule >> ((row[(i - 1) % w] << 2) | (row[i] << 1) | row[(i + 1) % w])) & 1
        for i in range(w)
    )


def _render_step(gen: int, prev: tuple[int, ...], nxt: tuple[int, ...], fmt: str) -> str:
    """One generation of the gold trace. May be several lines (AMENDMENT 3)."""
    row = "".join(str(b) for b in nxt)
    if fmt == "rows":
        return f"generation {gen}: {row}"
    if fmt != "cells":
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    w = len(prev)
    lines = [f"generation {gen}:"]
    for i in range(w):
        li, ri = (i - 1) % w, (i + 1) % w            # ring indices, 0-based
        nb = f"{prev[li]}{prev[i]}{prev[ri]}"
        so_far = row[: i + 1] + "_" * (w - i - 1)    # running row: one more bit per line
        lines.append(
            f"  cell {i + 1}: left c{li + 1}={prev[li]}, self c{i + 1}={prev[i]}, "
            f"right c{ri + 1}={prev[ri]} -> {nb} -> {nxt[i]}  row so far: {so_far}"
        )
    lines.append(f"  row: {row}")
    return "\n".join(lines)


def _build_prompt(rule: int, width: int, initial: str, depth: int) -> str:
    gen_word = "generation" if depth == 1 else "generations"
    return (
        f"An elementary cellular automaton runs rule {rule} on a ring of {width} cells. "
        f"The cells are numbered 1 to {width} from left to right, and the ring is circular: "
        f"the left neighbour of cell 1 is cell {width}, and the right neighbour of cell "
        f"{width} is cell 1.\n"
        "In each generation, every cell is updated at the same time. A cell's new value "
        "depends on the three cells (left neighbour, itself, right neighbour) in the current "
        "generation, according to the rule table:\n"
        "\n"
        f"{_rule_table_block(rule)}\n"
        "\n"
        f"Generation 0 is:\n"
        "\n"
        f"{initial}\n"
        "\n"
        f"Run the automaton for {depth} {gen_word}. What is generation {depth}?"
    )


def generate(
    depth: int,
    seed: int,
    rule: int = 110,
    width: int = 8,
    format: str = "cells",
    avoid_absorbing: bool = True,
    **_ignored,
) -> Instance:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if not (0 <= rule <= 255):
        raise ValueError("rule must be in 0..255")
    if width < 3:
        raise ValueError("width must be >= 3 (a ring needs distinct left/right neighbours)")
    if format not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {format!r}")

    rng = random.Random(
        f"cellular_automaton|{seed}|{depth}|{rule}|{width}|{int(bool(avoid_absorbing))}"
    )

    attempts = 64  # bounded: some rule/width pairs (e.g. rule 90, width 8) always die out
    start: tuple[int, ...] = ()
    rows: list[tuple[int, ...]] = []
    for attempt in range(attempts):
        start = tuple(rng.randint(0, 1) for _ in range(width))
        rows = [start]
        for _ in range(depth):
            rows.append(_next_row(rows[-1], rule))
        if not avoid_absorbing:
            break
        stuck = any(rows[i] == rows[i + 1] for i in range(len(rows) - 1))
        if not stuck or attempt == attempts - 1:
            break

    def s(row: tuple[int, ...]) -> str:
        return "".join(str(b) for b in row)

    states = [s(row) for row in rows[1:]]
    steps = [
        _render_step(k, rows[k - 1], rows[k], format) for k in range(1, depth + 1)
    ]
    answer = states[-1]
    initial = s(start)

    return Instance(
        prompt=_build_prompt(rule, width, initial, depth),
        steps=steps,
        states=states,
        answer=answer,
        depth=depth,
        meta={
            "seed": seed,
            "rule": rule,
            "width": width,
            "format": format,
            "avoid_absorbing": bool(avoid_absorbing),
            "initial": initial,
        },
    )


def solve(inst: Instance) -> str:
    """Independent reference solver.

    generate() evolves the ring cell by cell with a table lookup per cell. This
    instead packs the whole row into one integer and evolves it with bit-parallel
    masks: for each neighbourhood pattern whose rule bit is 1, AND together the
    three rotated/complemented planes and OR the results. Different algorithm,
    different indexing, same semantics.
    """
    rule = inst.meta["rule"]
    width = inst.meta["width"]
    mask = (1 << width) - 1
    x = int(inst.meta["initial"], 2) & mask  # cell 1 is the most significant bit

    live = [p for p in range(8) if (rule >> p) & 1]
    for _ in range(inst.depth):
        left = ((x >> 1) | ((x & 1) << (width - 1))) & mask   # plane of left neighbours
        right = ((x << 1) | (x >> (width - 1))) & mask        # plane of right neighbours
        nxt = 0
        for p in range(8):
            if p not in live:
                continue
            l_plane = left if (p >> 2) & 1 else ~left
            c_plane = x if (p >> 1) & 1 else ~x
            r_plane = right if p & 1 else ~right
            nxt |= l_plane & c_plane & r_plane
        x = nxt & mask

    return format(x, "0{}b".format(width))


_ANSWER_LINE = re.compile(r"^\s*answer\s*:\s*(.*)$", re.IGNORECASE)


def _extract(completion: str) -> str:
    """Last 'Answer:' line, else last non-empty line; then keep the bits."""
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
    if ":" in payload:  # strip a leading label like "generation 12:" or "Answer:"
        payload = payload.rsplit(":", 1)[1]
    payload = payload.strip().strip(".")
    return "".join(ch for ch in payload if ch in "01")


def check(inst: Instance, completion: str) -> bool:
    return _extract(completion) == inst.answer


def format_cot(inst: Instance) -> str:
    return "\n".join(inst.steps) + "\nAnswer: " + inst.answer


# The generation-0 row, as it sits alone on its own line in the prompt.
_INITIAL_LINE = re.compile(r"(?m)^(?P<bits>[01]{3,})$")


def redact_prompt(inst: Instance, k: int) -> str:
    """Hide what generations 1..k consumed, keep what generations k+1..depth need.

    The only per-instance state in the prompt is the generation-0 row. The rule table, the
    ring size, the periodic boundary and the generation count are static material that every
    remaining generation consumes -- there is no per-step operator list to trim -- so the
    redaction is all-or-nothing: k == 0 returns the prompt unchanged, any k >= 1 replaces the
    generation-0 row with a single placeholder and changes nothing else. k matters only
    through the trace prefix the harness supplies alongside this prompt (AMENDMENT 4; same
    shape as the turing_machine tag system).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return inst.prompt
    redacted, n = _INITIAL_LINE.subn(REDACTION_PLACEHOLDER, inst.prompt, count=1)
    if n != 1:
        raise ValueError("prompt has no generation-0 row to redact")
    return redacted


# AMENDMENT 6 (step corruption). The state after a generation is the row, so the plausible
# minimal wrong state is the true row with exactly one cell flipped. In the per-cell format
# the row is reported in three places -- that cell's `-> b` lookup result, every `row so far:`
# prefix from that cell onward, and the closing `row:` line -- and all three have to agree, or
# the corruption reads as a typo rather than as a mistake the model should carry forward. The
# neighbour values and the neighbourhood string they spell stay truthful: the *action* is
# unchanged, only the reported output bit. See README :: Corruption.


def _corrupt_cell(inst: Instance, k: int, seed: int) -> int:
    """Which cell of generation k gets its bit flipped. Deterministic in (inst, k, seed)."""
    rng = random.Random(
        "cellular_automaton|corrupt|{}|{}|{}|{}|{}|{}|{}".format(
            seed, k, inst.meta["seed"], inst.depth, inst.meta["rule"],
            inst.meta["width"], inst.meta["initial"],
        )
    )
    return rng.randrange(inst.meta["width"])


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """Step k (1-based) rewritten to report a plausible wrong row, and that row.

    Exactly one bit of generation k's row is flipped, which keeps the wrong state the same
    length and alphabet as the true one and is the smallest mistake a model could make here
    (one cell's rule lookup misread). The rewrite respects `inst.meta["format"]`:

    * `"rows"` -- the single `generation k: <row>` line restates the corrupted row.
    * `"cells"` -- the flipped cell's line reports the wrong bit as its lookup result, and
      every `row so far:` prefix from that cell onward plus the closing `row:` line carry the
      corrupted row, so the whole step is internally consistent. The left/self/right values
      and the neighbourhood string stay truthful; only the reported output bit and the rows
      change, per AMENDMENT 6 (d).

    Returns (step_text, corrupted_state); corrupted_state is in the same canonical bit-string
    form as `inst.states`.
    """
    if not 1 <= k <= inst.depth:
        raise ValueError(f"k must be in 1..{inst.depth}, got {k}")
    fmt = inst.meta["format"]
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")

    true_row = inst.states[k - 1]
    i = _corrupt_cell(inst, k, seed)
    bits = list(true_row)
    bits[i] = "1" if bits[i] == "0" else "0"
    corrupted = "".join(bits)

    # Re-render the step from the TRUE previous row (so the neighbour lookups stay honest)
    # and the corrupted next row (so every place the row is reported moves together).
    prev = inst.meta["initial"] if k == 1 else inst.states[k - 2]
    step_text = _render_step(
        k,
        tuple(int(c) for c in prev),
        tuple(int(c) for c in corrupted),
        fmt,
    )
    return step_text, corrupted


def exemplars(k: int, seed: int) -> list[Instance]:
    """No published exemplar exists for this task, so all are generated.

    Short depths (2-4) so a few-shot block stays bounded: under the default per-cell
    format one generation is width + 2 lines, so a 3-shot block is already ~90 lines.
    """
    out = []
    for i in range(k):
        out.append(generate(depth=2 + (i % 3), seed=(seed * 1000003 + 7919 * (i + 1)) % (2**31)))
    return out


def _selftest() -> None:
    rng = random.Random(0)
    rules = [110, 90, 30, 54, 150, 184, 22, 45]
    for i in range(200):
        depth = rng.randint(1, 24)
        seed = rng.randint(0, 10**9)
        rule = rng.choice(rules)
        width = rng.randint(6, 12)
        avoid = rng.random() < 0.8
        fmt = FORMATS[i % len(FORMATS)]   # both formats get ~100 instances each
        inst = generate(depth, seed, rule=rule, width=width, format=fmt, avoid_absorbing=avoid)

        got = solve(inst)
        assert got == inst.answer, f"solve mismatch: {got} != {inst.answer} ({inst.meta})"
        assert check(inst, format_cot(inst)), f"check(gold) failed ({inst.meta})"

        flipped = ("1" if inst.answer[0] == "0" else "0") + inst.answer[1:]
        assert not check(inst, f"Answer: {flipped}"), f"check accepted a wrong answer ({inst.meta})"

        again = generate(depth, seed, rule=rule, width=width, format=fmt, avoid_absorbing=avoid)
        assert again.prompt == inst.prompt and again.steps == inst.steps, "generate not deterministic"

        assert len(inst.steps) == len(inst.states) == depth, "len(steps)/len(states) != depth"
        assert all(len(s) == width for s in inst.states), "state width drift"
        assert inst.states[-1] == inst.answer, "answer is not the final state"

        # the choice of trace format must not change the instance, only its rendering
        other = generate(depth, seed, rule=rule, width=width,
                         format=("rows" if fmt == "cells" else "cells"), avoid_absorbing=avoid)
        assert other.prompt == inst.prompt and other.states == inst.states, "format changed the instance"

        # step structure: one step per generation, labelled, state at a fixed position
        for k, step in enumerate(inst.steps, start=1):
            lines = step.split("\n")
            assert lines[0].startswith(f"generation {k}"), f"step {k} mislabelled ({inst.meta})"
            if fmt == "rows":
                assert len(lines) == 1, "rows format is not one line per generation"
                assert lines[0] == f"generation {k}: {inst.states[k - 1]}"
            else:
                assert len(lines) == width + 2, "cells format is not width + 2 lines"
                assert lines[0] == f"generation {k}:"
                assert lines[-1] == f"  row: {inst.states[k - 1]}"
                prev = inst.meta["initial"] if k == 1 else inst.states[k - 2]
                for i_cell in range(width):
                    li, ri = (i_cell - 1) % width, (i_cell + 1) % width
                    nb = prev[li] + prev[i_cell] + prev[ri]
                    bit = (rule >> int(nb, 2)) & 1   # independent of _render_step's arithmetic
                    want = (f"  cell {i_cell + 1}: left c{li + 1}={prev[li]}, "
                            f"self c{i_cell + 1}={prev[i_cell]}, right c{ri + 1}={prev[ri]} "
                            f"-> {nb} -> {bit}  row so far: "
                            + inst.states[k - 1][: i_cell + 1] + "_" * (width - i_cell - 1))
                    assert lines[1 + i_cell] == want, (
                        f"bad cell line {i_cell + 1} in generation {k}:\n"
                        f"  got  {lines[1 + i_cell]!r}\n  want {want!r}\n  ({inst.meta})")
                    assert str(bit) == inst.states[k - 1][i_cell], "cell line disagrees with the row"
                    # the running row must never require gathering bits from earlier lines:
                    # each line's prefix extends the previous line's by exactly one known bit
                    sofar = lines[1 + i_cell].split("row so far: ")[1]
                    assert len(sofar) == width and sofar.count("_") == width - i_cell - 1
                    if i_cell:
                        prev_sofar = lines[i_cell].split("row so far: ")[1]
                        assert sofar[:i_cell] == prev_sofar[:i_cell], "running row rewrote itself"
                # the restated row line must equal the last line's completed running row
                assert lines[-2].split("row so far: ")[1] == inst.states[k - 1], (
                    "final running row != row line")

    # ---- redaction (AMENDMENT 4) ----
    assert REDACTION_MEANINGFUL is True
    for j in range(20):
        depth = rng.randint(2, 16)
        inst = generate(depth, seed=5000 + j)
        initial = inst.meta["initial"]
        assert redact_prompt(inst, 0) == inst.prompt, "redact_prompt(inst, 0) != inst.prompt"
        for k in (1, depth // 2, depth):
            red = redact_prompt(inst, k)
            assert red != inst.prompt, f"redact_prompt k={k} did not change the prompt"
            assert REDACTION_PLACEHOLDER in red, f"redact_prompt k={k} has no placeholder"
            assert red.count(REDACTION_PLACEHOLDER) == 1, "more than one placeholder"
            # the generation-0 row is gone; the static rule table and the question stay
            assert f"\n{initial}\n" not in red, f"redact_prompt k={k} still shows generation 0"
            assert _rule_table_block(inst.meta["rule"]) in red, "redaction dropped the rule table"
            assert f"What is generation {depth}?" in red, "redaction dropped the question"
            assert "the ring is circular" in red, "redaction dropped the boundary condition"
        assert redact_prompt(inst, 1) == redact_prompt(inst, depth), "redaction is not all-or-nothing"
    try:
        redact_prompt(generate(4, 11), -1)
    except ValueError:
        pass
    else:
        raise AssertionError("redact_prompt accepted k < 0")

    # ---- step corruption (AMENDMENT 6) ----
    for j in range(20):
        depth = rng.randint(1, 12)
        width = rng.randint(6, 10)
        rule = rng.choice(rules)
        cseed = rng.randint(0, 10**9)
        for fmt in FORMATS:
            inst = generate(depth, seed=7000 + j, rule=rule, width=width, format=fmt)
            gold_all = inst.steps
            for k in sorted({1, max(1, depth // 2), depth}):
                text, bad = corrupt_step(inst, k, cseed)
                true_row = inst.states[k - 1]
                prev = inst.meta["initial"] if k == 1 else inst.states[k - 2]

                assert bad != true_row, f"corrupted state == true state (k={k}, {inst.meta})"
                assert len(bad) == width and set(bad) <= {"0", "1"}, "corrupted state is not a legal row"
                assert sum(a != b for a, b in zip(bad, true_row)) == 1, "not a single-bit flip"
                assert text != gold_all[k - 1], "corrupt_step returned the gold step text"
                assert (text, bad) == corrupt_step(inst, k, cseed), "corrupt_step is not deterministic"

                lines = text.split("\n")
                gold = gold_all[k - 1].split("\n")
                if fmt == "rows":
                    assert lines == [f"generation {k}: {bad}"], f"rows corruption broke the step line: {lines}"
                else:
                    flipped = [c for c in range(width) if bad[c] != true_row[c]][0]
                    assert len(lines) == width + 2, "corrupted step is not width + 2 lines"
                    assert lines[0] == gold[0] == f"generation {k}:", "corruption touched the label line"
                    assert lines[-1] == f"  row: {bad}", "row: line does not restate the corrupted row"
                    for c in range(width):
                        li, ri = (c - 1) % width, (c + 1) % width
                        nb = prev[li] + prev[c] + prev[ri]
                        # action text: neighbour indices, neighbour values and the neighbourhood
                        # string they spell all stay truthful -- only the reported bit may move
                        head = (f"  cell {c + 1}: left c{li + 1}={prev[li]}, self c{c + 1}={prev[c]}, "
                                f"right c{ri + 1}={prev[ri]} -> {nb} -> ")
                        want = head + bad[c] + "  row so far: " + bad[: c + 1] + "_" * (width - c - 1)
                        assert lines[1 + c] == want, (
                            f"bad corrupted cell line {c + 1} (k={k}, {inst.meta}):\n"
                            f"  got  {lines[1 + c]!r}\n  want {want!r}")
                        if c < flipped:
                            assert lines[1 + c] == gold[1 + c], "corruption rewrote a line before the flip"
                        else:
                            # the flipped cell's lookup result, then every later running row
                            assert lines[1 + c] != gold[1 + c], "corruption missed a line after the flip"
                    assert lines[-1] != gold[-1], "the row: line was not rewritten"
                    # exactly one reported lookup result differs from the gold trace
                    diffs = sum(
                        1 for c in range(width)
                        if lines[1 + c].split(" -> ")[2][0] != gold[1 + c].split(" -> ")[2][0]
                    )
                    assert diffs == 1, f"{diffs} lookup results changed, want 1"
    for bad_k in (0, -1, generate(4, 11).depth + 1):
        try:
            corrupt_step(generate(4, 11), bad_k, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f"corrupt_step accepted k = {bad_k}")

    # tolerant-parser spot checks on the default instance
    inst = generate(depth=6, seed=42)
    assert check(inst, f"Answer: {inst.answer}")
    assert check(inst, f"blah blah\nAnswer: {' '.join(inst.answer)}\n")
    assert check(inst, f"Answer: {inst.answer}.")
    assert check(inst, f"generation 6: {inst.answer}")  # no Answer: line -> last non-empty line
    assert check(inst, f"Answer: wrong\nAnswer: {inst.answer}")  # last Answer: line wins
    assert not check(inst, "Answer: " + "0" * (inst.meta["width"] + 1))

    ex = exemplars(3, 0)
    assert len(ex) == 3
    for e in ex:
        assert e.prompt and format_cot(e) and solve(e) == e.answer

    print("selftest OK: 200 random instances over both trace formats (solve==answer, "
          "check(gold)=True, check(wrong)=False, deterministic,")
    print("             len(steps)==len(states)==depth, per-cell lookups verified against an "
          "independent rule-bit computation)")
    print("             + redaction on 20 instances (k in {0, 1, depth//2, depth}; rule table, "
          "boundary and question kept)")
    print("             + step corruption on 20 instances x both formats x k in {1, depth//2, depth} "
          "(one-bit flip, honest neighbours, running row and row: line rewritten together)")
    print("             + tolerant-parser spot checks + exemplars(3, 0) render")


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            seed = 1000 + i
            inst = generate(depth=depth, seed=seed)
            print(f"===== depth={depth} seed={seed} rule={inst.meta['rule']} "
                  f"width={inst.meta['width']} format={inst.meta['format']} =====")
            print("--- prompt ---")
            print(inst.prompt)
            print("--- gold CoT ---")
            print(format_cot(inst))
            print()
    # one instance in the legacy format, for comparison
    inst = generate(depth=DEPTHS[-1], seed=1000, format="rows")
    print(f"===== depth={inst.depth} seed=1000 rule={inst.meta['rule']} "
          f"width={inst.meta['width']} format=rows (legacy) =====")
    print("--- gold CoT ---")
    print(format_cot(inst))
    print()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--demo" in sys.argv:
        _demo()
    else:
        print("usage: python3 task.py [--selftest|--demo]")
