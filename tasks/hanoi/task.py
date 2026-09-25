"""
hanoi — Tower of Hanoi, Apple "Illusion of Thinking" (arXiv:2506.06941).

Default formulation (batch 2): **execution / state tracking**. The prompt gives a random legal
start configuration in the paper's bracket notation (`[[3, 2, 1], [], []]`) plus a list of `depth`
legal moves in the paper's `[disk_id, from_peg, to_peg]` notation; the question is the configuration
the pegs are in after those moves. Same shape as blocksworld's plan-execution task.

The batch-1 formulation — "plan the first `depth` moves of the optimal solution from a random start,
then report the configuration you reach" — is still available as `generate(..., formulation="plan")`.
It was withdrawn as the default because models do not do it: at depth 2 DeepSeek opened with "We need
to move the entire stack of 3 disks from peg 2 to peg 1" and recited the textbook 7-move solution,
ignoring the random start entirely. That is planning (badly), not tracking. See README.md.

Pure python + stdlib. No network. The verbatim Apple prompt wording is read from
published_trace.txt (recorded in phase 1) and exposed as APPLE_SYSTEM_PROMPT /
APPLE_USER_PROMPT_TEMPLATE / PUBLISHED_EXEMPLAR_MOVES.
"""

from dataclasses import dataclass, field
from pathlib import Path
import random
import re
import sys


# --------------------------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------------------------

@dataclass
class Instance:
    prompt: str          # problem statement only
    steps: list          # gold trace, one element per move
    states: list          # peg configuration after each move (len == len(steps))
    answer: str          # exact-match target: the configuration after the last move
    depth: int           # number of moves
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = (
    "a peg configuration written like [[3, 2], [1], []] — three lists (peg 0, peg 1, peg 2), "
    "each listing that peg's disks from bottom to top"
)

# depth = number of moves applied. Execution is cheap per step (one disk changes peg), so the
# ladder runs much deeper than the batch-1 planning ladder did. See README.md.
DEPTHS = [2, 4, 8, 16, 32, 64]

DEFAULT_DISKS = 4

KNOBS = {
    "formulation": (
        "execute",
        "'execute' (default): the prompt gives a start configuration and `depth` legal moves; the "
        "model reports the resulting configuration. 'plan': the batch-1 formulation — the prompt "
        "gives a start and a single-peg goal, and the model must itself work out the first `depth` "
        "moves of the (unique) optimal solution and report the configuration they reach.",
    ),
    "disks": (
        None,
        f"number of disks n. None = {DEFAULT_DISKS} for the execute formulation with a random move "
        "list; for formulation='plan' or optimal_prefix=True, None = max(3, smallest n with "
        "2**n - 1 > depth), so that the start is always strictly farther from the goal than depth.",
    ),
    "optimal_prefix": (
        False,
        "execute formulation only: instead of a random legal walk, use the first `depth` moves of "
        "the optimal solution to a single-peg goal (bridge to the old 'plan' formulation — same "
        "move list, but the moves are given to the model instead of being derived by it).",
    ),
    "allow_undo": (
        False,
        "allow the random walk to immediately reverse the previous move (a step that cancels the "
        "one before it is trivially trackable, so it is excluded by default).",
    ),
    "goal_peg": (
        None,
        "formulation='plan' / optimal_prefix=True only: fix the goal peg (0/1/2); None = sampled "
        "from the seed. The goal is always a single peg holding every disk, which is what makes "
        "the shortest solution unique.",
    ),
}

_MAX_DEPTH = 20000  # guard: the move list is materialised


# --------------------------------------------------------------------------------------------
# published wording (vendored / recorded verbatim in published_trace.txt)
# --------------------------------------------------------------------------------------------

_TRACE_PATH = Path(__file__).resolve().parent / "published_trace.txt"


def _load_published_sections() -> dict:
    """Split published_trace.txt on its '=== ... ===' banners. Returns {} if the file is absent."""
    try:
        raw = _TRACE_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    sections, name, buf = {}, None, []
    for line in raw.splitlines():
        if line.startswith("=== ") and line.rstrip().endswith("==="):
            if name is not None:
                sections[name] = "\n".join(buf).strip("\n")
            name, buf = line.strip("= ").strip(), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        sections[name] = "\n".join(buf).strip("\n")
    return sections


_SECTIONS = _load_published_sections()


def _section(keyword: str) -> str:
    for key, value in _SECTIONS.items():
        if keyword.lower() in key.lower():
            return value
    return ""


APPLE_SYSTEM_PROMPT = _section("SYSTEM PROMPT")
APPLE_USER_PROMPT_TEMPLATE = _section("USER PROMPT TEMPLATE")
_PUBLISHED_TRACE_TEXT = _section("WORKED EXAMPLE")
PUBLISHED_EXEMPLAR_MOVES = [
    (int(a), int(b), int(c))
    for a, b, c in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", _PUBLISHED_TRACE_TEXT)
]


# --------------------------------------------------------------------------------------------
# state helpers
# --------------------------------------------------------------------------------------------

def _pegs_from_pos(pos: list) -> list:
    """pos[i] = peg of disk i+1  ->  three lists of disks, bottom (largest) first."""
    pegs = [[], [], []]
    for disk in range(len(pos), 0, -1):
        pegs[pos[disk - 1]].append(disk)
    return pegs


def _config_str(pegs: list) -> str:
    """Published notation, e.g. '[[3, 2, 1], [], []]'."""
    return "[" + ", ".join("[" + ", ".join(str(d) for d in peg) + "]" for peg in pegs) + "]"


def _canon(pegs: list) -> str:
    """Canonical form: each peg sorted bottom-to-top (descending), so disk order can't disagree."""
    return _config_str([sorted(peg, reverse=True) for peg in pegs])


def _triple(move: tuple) -> str:
    disk, src, dst = move
    return f"[{disk}, {src}, {dst}]"


def _move_line(move: tuple) -> str:
    """One move inside the published `moves = [...]` list literal."""
    return "  " + _triple(move) + ","


def _legal_moves(pos: list):
    """Yield (disk, src, dst) for every legal single move out of `pos`."""
    tops = [None, None, None]
    for disk in range(len(pos), 0, -1):
        tops[pos[disk - 1]] = disk          # smallest disk seen last wins => top of peg
    for src in range(3):
        disk = tops[src]
        if disk is None:
            continue
        for dst in range(3):
            if dst != src and (tops[dst] is None or tops[dst] > disk):
                yield disk, src, dst


def _applied(pos: list, move: tuple) -> list:
    disk, _src, dst = move
    nxt = list(pos)
    nxt[disk - 1] = dst
    return nxt


# --------------------------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------------------------

# --- execute formulation (the default) ---

_EXEC_INIT = "Initial configuration: "
_EXEC_MOVES_OPEN = "moves = [\n"
_EXEC_MOVES_CLOSE = "\n]\n"
_EXEC_QUESTION = "Report the configuration of the three pegs after these moves."

_EXEC_RULES = [
    "Rules:",
    "",
    "• Only one disk can be moved at a time.",
    "",
    "• Only the top disk from any stack can be moved.",
    "",
    "• A larger disk may not be placed on top of a smaller disk.",
]


def _build_exec_prompt(pegs: list, n: int, moves: list) -> str:
    lines = [
        f"I have a puzzle with {n} disks of different sizes on three pegs. The disks are numbered "
        f"from 1 (smallest) to {n} (largest). The positions are 0-indexed (the leftmost peg is 0). "
        "A configuration is written as three lists — the disks on peg 0, then peg 1, then peg 2 — "
        "each list running from the bottom of that peg to the top.",
        "",
        _EXEC_INIT + _config_str(pegs),
        "",
    ]
    lines += _EXEC_RULES
    lines += [
        "",
        "I made the following moves, in order, each written as [disk_id, from_peg, to_peg]:",
        "",
        "moves = [",
    ]
    lines += [_move_line(m) for m in moves]
    lines += ["]", "", _EXEC_QUESTION]
    return "\n".join(lines)


def _split_exec_prompt(prompt: str) -> tuple:
    """(i, j, a, b): char span of the initial configuration and char span of the move lines."""
    i = prompt.index(_EXEC_INIT) + len(_EXEC_INIT)
    j = prompt.index("\n", i)
    a = prompt.index(_EXEC_MOVES_OPEN, j) + len(_EXEC_MOVES_OPEN)
    b = prompt.index(_EXEC_MOVES_CLOSE, a)
    return i, j, a, b


def _parse_exec_prompt(prompt: str) -> tuple:
    """Read the start configuration and the move list back out of the prompt text."""
    i, j, a, b = _split_exec_prompt(prompt)
    pegs = _parse_config(prompt[i:j])
    if pegs is None:
        raise ValueError("could not parse the initial configuration out of the prompt")
    moves = []
    for line in prompt[a:b].split("\n"):
        nums = re.findall(r"-?\d+", line)
        if len(nums) != 3:
            raise ValueError(f"could not parse move line {line!r}")
        moves.append(tuple(int(x) for x in nums))
    return pegs, moves


# --- plan formulation (batch 1, kept as a knob) ---

def _render_peg(peg: list) -> str:
    if not peg:
        return "(empty)"
    if len(peg) == 1:
        return str(peg[0])
    parts = [f"{peg[0]} (bottom)"] + [str(d) for d in peg[1:-1]] + [f"{peg[-1]} (top)"]
    return ", ".join(parts)


_PLAN_INIT_HEAD = "Initial configuration:\n"
_PLAN_GOAL_HEAD = "\nGoal configuration:"


def _build_plan_prompt(pegs: list, n: int, goal_peg: int, depth: int) -> str:
    goal = [[], [], []]
    goal[goal_peg] = list(range(n, 0, -1))
    lines = [f"I have a puzzle with {n} disks of different sizes with", "Initial configuration:", ""]
    for p in range(3):
        lines += [f"• Peg {p}: {_render_peg(pegs[p])}", ""]
    lines += ["Goal configuration:", ""]
    for p in range(3):
        lines += [f"• Peg {p}: {_render_peg(goal[p])}", ""]
    lines += ["Rules:", ""]
    lines += [line for line in _EXEC_RULES[2:]]
    lines += [
        "",
        "There is exactly one shortest sequence of moves that transforms the initial configuration "
        "into the goal configuration. The positions are 0-indexed (the leftmost peg is 0). Report "
        f"the configuration of the three pegs after the first {depth} "
        f"{'move' if depth == 1 else 'moves'} of that sequence.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------------

def _tower(i: int, src: int, dst: int, pos: list, moves: list) -> None:
    """Move a perfect tower of disks 1..i from src to dst."""
    if i == 0:
        return
    via = 3 - src - dst
    _tower(i - 1, src, via, pos, moves)
    moves.append((i, src, dst))
    pos[i - 1] = dst
    _tower(i - 1, via, dst, pos, moves)


def _relocate(i: int, target: int, pos: list, moves: list) -> None:
    """Gather disks 1..i (wherever they are) onto peg `target`, optimally."""
    if i == 0:
        return
    here = pos[i - 1]
    if here == target:
        _relocate(i - 1, target, pos, moves)
        return
    other = 3 - target - here
    _relocate(i - 1, other, pos, moves)
    moves.append((i, here, target))
    pos[i - 1] = target
    _tower(i - 1, other, target, pos, moves)


def _min_disks_for(depth: int) -> int:
    """Smallest n with 2**n - 1 > depth: guarantees a start config strictly farther than `depth`."""
    n = 1
    while (1 << n) - 1 <= depth:
        n += 1
    return n


def _sample_far_start(rng: random.Random, n: int, goal_peg: int, depth: int) -> tuple:
    """
    Sample a start configuration uniformly among those whose optimal distance to the goal is > depth.

    The distance D of a configuration decomposes bitwise: bit (i-1) of D is 1 exactly when disk i is
    NOT on the running target peg, and a 1-bit leaves two choices of peg. So there are
    2**popcount(D) configurations at distance D, and sampling D with that weight then filling in the
    free choices is uniform over the eligible set.
    """
    candidates = [d for d in range(depth + 1, (1 << n))]
    if not candidates:
        raise ValueError(f"no start configuration with {n} disks is more than {depth} moves away")
    weights = [1 << bin(d).count("1") for d in candidates]
    distance = rng.choices(candidates, weights=weights, k=1)[0]
    pos = [0] * n
    target = goal_peg
    for i in range(n, 0, -1):
        if (distance >> (i - 1)) & 1:
            choice = rng.choice([p for p in range(3) if p != target])
            pos[i - 1] = choice
            target = 3 - target - choice
        else:
            pos[i - 1] = target
    return pos, distance


def _random_walk(rng: random.Random, pos: list, depth: int, allow_undo: bool) -> list:
    """`depth` legal moves, never immediately undoing the previous one unless allow_undo."""
    moves, prev = [], None
    for _ in range(depth):
        options = sorted(_legal_moves(pos))
        if prev is not None and not allow_undo:
            undo = (prev[0], prev[2], prev[1])
            options = [m for m in options if m != undo]
        if not options:                  # unreachable (>= 2 legal moves always exist), but be loud
            raise AssertionError("no legal non-undo move available")
        move = rng.choice(options)
        moves.append(move)
        pos[move[0] - 1] = move[2]
        prev = move
    return moves


def _exec_step_line(index: int, move: tuple, state: str) -> str:
    return f"move {index}: {_triple(move)} -> {state}"


#: Structure of a rendered step line, per formulation (used by --selftest and by corrupt_step()).
EXEC_STEP_RE = re.compile(r"^move (\d+): \[(\d+), (\d+), (\d+)\] -> (\[\[.*\]\])$")
PLAN_STEP_RE = re.compile(r"^  \[(\d+), (\d+), (\d+)\],$")


def generate(depth: int, seed: int, **knobs) -> Instance:
    if depth < 1:
        raise ValueError("depth must be >= 1 (at least one move)")
    if depth > _MAX_DEPTH:
        raise ValueError(f"depth must be <= {_MAX_DEPTH}")
    formulation = knobs.get("formulation", KNOBS["formulation"][0])
    if formulation not in ("execute", "plan"):
        raise ValueError("formulation must be 'execute' or 'plan'")
    disks = knobs.get("disks", KNOBS["disks"][0])
    optimal_prefix = knobs.get("optimal_prefix", KNOBS["optimal_prefix"][0])
    allow_undo = knobs.get("allow_undo", KNOBS["allow_undo"][0])
    fixed_goal = knobs.get("goal_peg", KNOBS["goal_peg"][0])
    optimal = bool(optimal_prefix) or formulation == "plan"

    rng = random.Random(f"hanoi|{formulation}|{int(bool(optimal_prefix))}|{depth}|{seed}")
    if disks is not None:
        n = int(disks)
        if n < 1:
            raise ValueError("disks must be >= 1")
    else:
        n = max(3, _min_disks_for(depth)) if optimal else DEFAULT_DISKS

    meta = {"n": n, "formulation": formulation, "seed": seed,
            "allow_undo": bool(allow_undo), "optimal_prefix": bool(optimal_prefix)}

    if optimal:
        goal_peg = fixed_goal if fixed_goal is not None else rng.randrange(3)
        start_pos, distance = _sample_far_start(rng, n, goal_peg, depth)
        pos, moves = list(start_pos), []
        _relocate(n, goal_peg, pos, moves)
        moves = moves[:depth]
        meta["goal_peg"] = goal_peg
        meta["distance"] = distance
    else:
        start_pos = [rng.randrange(3) for _ in range(n)]
        moves = _random_walk(rng, list(start_pos), depth, allow_undo)

    # replay the move list to get the state after every move
    pos = list(start_pos)
    states = []
    for disk, _src, dst in moves:
        pos[disk - 1] = dst
        states.append(_config_str(_pegs_from_pos(pos)))

    start_pegs = _pegs_from_pos(start_pos)
    if formulation == "execute":
        prompt = _build_exec_prompt(start_pegs, n, moves)
        steps = [_exec_step_line(i + 1, m, s) for i, (m, s) in enumerate(zip(moves, states))]
    else:
        prompt = _build_plan_prompt(start_pegs, n, meta["goal_peg"], depth)
        steps = [_move_line(m) for m in moves]

    meta.update({
        "start_pos": list(start_pos),
        "start": _config_str(start_pegs),
        "moves": [list(m) for m in moves],
    })
    return Instance(prompt=prompt, steps=steps, states=states, answer=states[-1],
                    depth=depth, meta=meta)


# --------------------------------------------------------------------------------------------
# solve — independent of generate()
# --------------------------------------------------------------------------------------------

def _distance(pos: list, target: int) -> int:
    """Optimal number of moves from `pos` to 'every disk on peg target'."""
    d, t = 0, target
    for i in range(len(pos), 0, -1):
        p = pos[i - 1]
        if p != t:
            d += 1 << (i - 1)
            t = 3 - t - p
    return d


def solve(inst: Instance) -> str:
    """
    Reference solver, written independently of generate().

    execute: re-reads the start configuration and the move list out of the PROMPT TEXT and applies
    them with an explicit stack simulation (push/pop with precondition checks), where generate()
    works from an internal disk->peg array it never re-parses.

    plan: generate() builds the move list by the recursive divide-and-conquer algorithm; solve()
    never recurses — it computes the closed-form distance-to-goal of every legal neighbour state
    and takes the unique move that decreases it (the shortest path to a single-peg goal is unique).
    """
    if inst.meta.get("formulation", "execute") == "execute":
        stacks, moves = _parse_exec_prompt(inst.prompt)
        stacks = [list(p) for p in stacks]
        for disk, src, dst in moves:
            if not 0 <= src <= 2 or not 0 <= dst <= 2 or src == dst:
                raise ValueError(f"bad move {[disk, src, dst]}")
            if not stacks[src] or stacks[src][-1] != disk:
                raise ValueError(f"disk {disk} is not the top disk of peg {src}")
            if stacks[dst] and stacks[dst][-1] < disk:
                raise ValueError(f"disk {disk} cannot go on top of disk {stacks[dst][-1]}")
            stacks[dst].append(stacks[src].pop())
        return _config_str(stacks)

    pos = list(inst.meta["start_pos"])
    goal_peg = inst.meta["goal_peg"]
    d = _distance(pos, goal_peg)
    for _ in range(inst.depth):
        if d == 0:
            raise ValueError("already at the goal: no move number %d exists" % inst.depth)
        forward = [m for m in _legal_moves(pos) if _distance(_applied(pos, m), goal_peg) == d - 1]
        if len(forward) != 1:
            raise AssertionError(f"expected a unique distance-decreasing move, got {len(forward)}")
        disk, _src, dst = forward[0]
        pos[disk - 1] = dst
        d -= 1
    return _config_str(_pegs_from_pos(pos))


# --------------------------------------------------------------------------------------------
# rendering / checking
# --------------------------------------------------------------------------------------------

def format_cot(inst: Instance) -> str:
    """Gold trace + the harness-imposed final `Answer:` line."""
    if inst.meta.get("formulation", "execute") == "execute":
        return "\n".join(inst.steps) + "\nAnswer: " + inst.answer
    return "moves = [\n" + "\n".join(inst.steps) + "\n]\nAnswer: " + inst.answer


def step_spans(inst: Instance) -> list:
    """Char offsets (start, end) of each step line within format_cot()."""
    text = format_cot(inst)
    spans, cursor = [], 0
    for step in inst.steps:
        start = text.index(step, cursor)
        spans.append((start, start + len(step)))
        cursor = start + len(step)
    return spans


_ANSWER_LINE = re.compile(r"^\s*\**\s*answer\s*\**\s*:?\s*(.*)$", re.IGNORECASE)
_INNER_LIST = re.compile(r"\[([^\[\]]*)\]")
_PEG_FIELD = re.compile(r"peg\s*([0-2])\s*[:=\-]\s*([^\n;|]*)", re.IGNORECASE)


def _extract_answer_text(completion: str) -> str:
    lines = completion.splitlines()
    for line in reversed(lines):
        m = _ANSWER_LINE.match(line)
        if m:
            return m.group(1).strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


def _parse_config(text: str):
    """Tolerant parse of a peg configuration. Returns three lists of ints, or None."""
    groups = _INNER_LIST.findall(text)
    if len(groups) >= 3:
        groups = groups[-3:]
        try:
            return [[int(x) for x in re.findall(r"-?\d+", g)] for g in groups]
        except ValueError:
            return None
    found = {}
    for peg, body in _PEG_FIELD.findall(text):
        found[int(peg)] = [] if re.search(r"empty|none", body, re.IGNORECASE) \
            else [int(x) for x in re.findall(r"\d+", body)]
    if set(found) == {0, 1, 2}:
        return [found[0], found[1], found[2]]
    return None


def check(inst: Instance, completion: str) -> bool:
    """Extract the last 'Answer:' line (else the last non-empty line) and exact-match after
    canonicalisation (per-peg disk order is forced by the rules, so either order is accepted)."""
    parsed = _parse_config(_extract_answer_text(completion))
    if parsed is None or len(parsed) != 3:
        return False
    return _canon(parsed) == _canon(_parse_config(inst.answer))


# --------------------------------------------------------------------------------------------
# prompt redaction (AMENDMENT 4)
# --------------------------------------------------------------------------------------------

REDACTION_MEANINGFUL = True

#: Visible placeholder left behind for every removed item (or span).
REDACTED = "[…]"


def redact_prompt(inst: Instance, k: int) -> str:
    """Return `inst.prompt` with everything needed to *recompute* the state after step k removed.

    execute (the default): the initial configuration becomes one placeholder and each of the first
    `k` move lines becomes one placeholder — `k + 1` placeholders in all. Kept: the disk count, the
    notation description, the three rules (static, not state), moves k+1..depth and the question —
    everything needed to *continue* from step k+1 given the trace so far.

    plan: the model is given no moves at all, so the only redactable thing is the initial
    configuration; it is replaced by a single placeholder for any k > 0 (the goal, the rules and
    the question stay). Redaction is coarse here, which is one more reason 'plan' is not the
    default.

    `k == 0` returns the prompt unchanged; `k == inst.depth` leaves only static material and the
    question.
    """
    if not 0 <= k <= inst.depth:
        raise ValueError(f"k must be in 0..{inst.depth}, got {k}")
    if k == 0:
        return inst.prompt

    prompt = inst.prompt
    if inst.meta.get("formulation", "execute") == "plan":
        i = prompt.index(_PLAN_INIT_HEAD) + len(_PLAN_INIT_HEAD)
        j = prompt.index(_PLAN_GOAL_HEAD, i)
        return prompt[:i] + "\n" + REDACTED + "\n" + prompt[j:]

    i, j, _a, _b = _split_exec_prompt(prompt)
    prompt = prompt[:i] + REDACTED + prompt[j:]
    _i, _j, a, b = _split_exec_prompt(prompt)
    lines = prompt[a:b].split("\n")
    if len(lines) != inst.depth:
        raise ValueError(f"prompt has {len(lines)} move lines, expected {inst.depth}")
    lines[:k] = ["  " + REDACTED + ","] * k
    return prompt[:a] + "\n".join(lines) + prompt[b:]


# --------------------------------------------------------------------------------------------
# step corruption (AMENDMENT 6)
# --------------------------------------------------------------------------------------------

def _pegs_with_disk_on(pegs: list, disk: int, peg: int) -> list:
    """`pegs` with `disk` lifted off wherever it sits and dropped on `peg`, canonically ordered."""
    out = [[d for d in p if d != disk] for p in pegs]
    out[peg].append(disk)
    return [sorted(p, reverse=True) for p in out]


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple:
    """Rewrite step k (1-based) so that the STATE it reports is plausibly wrong.

    Returns `(step_text, corrupted_state)`; `corrupted_state` is that configuration in the same
    bracket form as `inst.states`, and is always a well-formed configuration of the same n disks
    (every peg bottom-to-top, largest first) that differs from the true `inst.states[k-1]`.

    execute (the default): the **move triple is left exactly as written** — only the configuration
    reported after the arrow changes. One of two wrong-but-legal-looking configurations is chosen
    from `(inst, k, seed)`:

    * *wrong peg* — the moved disk is put on the third peg instead of `to_peg`. Offered only when
      that peg is empty or carries a larger disk, so the reported configuration is one a legal move
      out of the previous configuration could actually have produced.
    * *move not applied* — the configuration from before move k is reported again (the classic
      dropped-update), which is legal by construction.

    plan: the step lines are bare `[disk, from, to]` triples and report no state at all, so there is
    nothing after an arrow to corrupt. Per AMENDMENT 6's "apply to every format the task supports",
    the **destination peg of the triple** is corrupted instead — to the third peg, the only other
    choice — and `corrupted_state` is the configuration that altered move reaches. The corrupted
    move may be illegal under the size rule (the third peg is whichever one the optimal move avoids,
    and it often holds a smaller disk); the *configuration* is still well formed, since in Hanoi any
    assignment of disks to pegs is a legal state. This is the one place where the action text moves,
    and it is documented in README.md under "Corruption".
    """
    if not 1 <= k <= inst.depth:
        raise ValueError(f"k must be in 1..{inst.depth}, got {k}")
    disk, src, dst = tuple(inst.meta["moves"][k - 1])
    other = 3 - src - dst
    before = inst.meta["start"] if k == 1 else inst.states[k - 2]
    before_pegs = _parse_config(before)
    if before_pegs is None:
        raise ValueError(f"could not parse the configuration before step {k}")

    wrong_peg = _config_str(_pegs_with_disk_on(before_pegs, disk, other))
    if inst.meta.get("formulation", "execute") == "plan":
        return _move_line((disk, src, other)), wrong_peg

    candidates = []
    top = before_pegs[other][-1] if before_pegs[other] else None
    if top is None or top > disk:
        candidates.append(wrong_peg)
    candidates.append(_canon(before_pegs))
    rng = random.Random(f"hanoi|corrupt|{seed}|{k}|{inst.meta['start']}|{inst.depth}")
    wrong = rng.choice(candidates)
    return _exec_step_line(k, (disk, src, dst), wrong), wrong


# --------------------------------------------------------------------------------------------
# exemplars
# --------------------------------------------------------------------------------------------

def _published_exemplar(formulation: str = "execute") -> Instance:
    """
    The paper's own worked example: 3 disks, standard start [[3, 2, 1], [], []], goal peg 2, and the
    7-move optimal solution printed verbatim in the appendix system prompt, re-cast as an execution
    instance (the moves are given; the resulting configuration is asked for). The move triples are
    parsed out of published_trace.txt, not recomputed.
    """
    moves = PUBLISHED_EXEMPLAR_MOVES or [
        (1, 0, 2), (2, 0, 1), (1, 2, 1), (3, 0, 2), (1, 1, 0), (2, 1, 2), (1, 0, 2)
    ]
    n, goal_peg = 3, 2
    start_pos = [0] * n
    pos = list(start_pos)
    states = []
    for disk, _src, dst in moves:
        pos[disk - 1] = dst
        states.append(_config_str(_pegs_from_pos(pos)))
    start_pegs = _pegs_from_pos(start_pos)
    if formulation == "execute":
        prompt = _build_exec_prompt(start_pegs, n, moves)
        steps = [_exec_step_line(i + 1, m, s) for i, (m, s) in enumerate(zip(moves, states))]
    else:
        prompt = _build_plan_prompt(start_pegs, n, goal_peg, len(moves))
        steps = [_move_line(m) for m in moves]
    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=states[-1],
        depth=len(moves),
        meta={
            "n": n,
            "formulation": formulation,
            "goal_peg": goal_peg,
            "start_pos": start_pos,
            "start": _config_str(start_pegs),
            "distance": len(moves),
            "moves": [list(m) for m in moves],
            "seed": None,
            "allow_undo": False,
            "optimal_prefix": True,
            "published_exemplar": True,
            "source": "arXiv:2506.06941 appendix (Tower of Hanoi system prompt); "
                      "see published_trace.txt",
        },
    )


def exemplars(k: int, seed: int) -> list:
    """exemplars(k, seed)[0] is the paper's published worked example (re-cast as an execution
    instance); the rest are generated at modest depth."""
    if k <= 0:
        return []
    out = [_published_exemplar()]
    for i in range(1, k):
        out.append(generate(depth=3 + (i % 4), seed=10_000 + 100 * seed + i))
    return out


# --------------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------------

def _bfs_answer(inst: Instance) -> str:
    """Brute-force cross-check for the plan formulation: BFS over the whole 3**n state graph from
    the goal, then walk the unique distance-decreasing path."""
    n, goal_peg = inst.meta["n"], inst.meta["goal_peg"]
    goal = tuple([goal_peg] * n)
    dist = {goal: 0}
    frontier = [goal]
    while frontier:
        nxt = []
        for state in frontier:
            for move in _legal_moves(list(state)):
                cand = tuple(_applied(list(state), move))
                if cand not in dist:
                    dist[cand] = dist[state] + 1
                    nxt.append(cand)
        frontier = nxt
    pos = tuple(inst.meta["start_pos"])
    for _ in range(inst.depth):
        options = [tuple(_applied(list(pos), m)) for m in _legal_moves(list(pos))]
        options = [o for o in options if dist[o] == dist[pos] - 1]
        assert len(options) == 1, "shortest path to a perfect state should be unique"
        pos = options[0]
    return _config_str(_pegs_from_pos(list(pos)))


def _wrong_answer(inst: Instance) -> str:
    pegs = [list(p) for p in _parse_config(inst.answer)]
    src = next(i for i, p in enumerate(pegs) if p)
    dst = (src + 1) % 3
    disk = pegs[src].pop()
    pegs[dst].append(disk)
    return _canon(pegs)


def _common_checks(inst: Instance, depth: int, tag: str, failures: list) -> None:
    if solve(inst) != inst.answer:
        failures.append(f"{tag}: solve() != answer ({solve(inst)!r} vs {inst.answer!r})")
    if not check(inst, format_cot(inst)):
        failures.append(f"{tag}: check() rejected the gold completion")
    if check(inst, "Answer: " + _wrong_answer(inst)):
        failures.append(f"{tag}: check() accepted a wrong answer")
    if not (len(inst.steps) == len(inst.states) == inst.depth == depth):
        failures.append(f"{tag}: len(steps)={len(inst.steps)} len(states)={len(inst.states)} "
                        f"depth={inst.depth} (expected {depth})")
    cot = format_cot(inst)
    if [cot[a:b] for a, b in step_spans(inst)] != inst.steps:
        failures.append(f"{tag}: step_spans() misaligned")
    # legality + state agreement, simulated with stacks (independent of generate's disk->peg array)
    stacks = [list(p) for p in _parse_config(inst.meta["start"])]
    for idx, ((disk, src, dst), state) in enumerate(zip(inst.meta["moves"], inst.states)):
        if not stacks[src] or stacks[src][-1] != disk or (stacks[dst] and stacks[dst][-1] < disk):
            failures.append(f"{tag}: illegal move {idx + 1} {[disk, src, dst]}")
            break
        stacks[dst].append(stacks[src].pop())
        if _config_str(stacks) != state:
            failures.append(f"{tag}: state mismatch after move {idx + 1}")
            break


def _selftest() -> int:
    rng = random.Random(20260918)
    failures = []

    # ---- default (execute) formulation, 200 random (depth, seed) pairs ---------------------
    for trial in range(200):
        depth = rng.choice(DEPTHS + [rng.randint(1, 50)])
        seed = rng.randrange(10 ** 6)
        inst = generate(depth, seed)
        tag = f"execute depth={depth} seed={seed}"
        _common_checks(inst, depth, tag, failures)

        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer) != \
           (inst.prompt, inst.steps, inst.states, inst.answer):
            failures.append(f"{tag}: generate() is not deterministic")

        # no immediate undo by default
        moves = [tuple(m) for m in inst.meta["moves"]]
        for x, y in zip(moves, moves[1:]):
            if y == (x[0], x[2], x[1]):
                failures.append(f"{tag}: move list contains an immediate undo {list(y)}")
                break

        # the prompt really does carry the start and every move, in the published notation
        if _EXEC_INIT + inst.meta["start"] not in inst.prompt:
            failures.append(f"{tag}: start configuration missing from the prompt")
        for m in moves:
            if _move_line(m) not in inst.prompt:
                failures.append(f"{tag}: move {list(m)} missing from the prompt")
                break
        if inst.steps[-1] != f"move {depth}: {_triple(moves[-1])} -> {inst.answer}":
            failures.append(f"{tag}: last step line is not in the documented format")
        if trial == 0:
            print(f"  sample: {tag} n={inst.meta['n']} answer={inst.answer}")

    # ---- knobs: optimal_prefix, disks, allow_undo, and the 'plan' formulation --------------
    for trial in range(60):
        depth = rng.choice([1, 2, 5, 8, 16, 32])
        seed = rng.randrange(10 ** 6)

        inst = generate(depth, seed, optimal_prefix=True)
        tag = f"optimal_prefix depth={depth} seed={seed}"
        _common_checks(inst, depth, tag, failures)
        if inst.meta["distance"] <= depth:
            failures.append(f"{tag}: start distance {inst.meta['distance']} not strictly > depth")
        pos = list(inst.meta["start_pos"])
        for m in [tuple(x) for x in inst.meta["moves"]]:
            pos = _applied(pos, m)
        if _distance(pos, inst.meta["goal_peg"]) != inst.meta["distance"] - depth:
            failures.append(f"{tag}: prefix is not optimal")

        pinst = generate(depth, seed, formulation="plan")
        ptag = f"plan depth={depth} seed={seed}"
        _common_checks(pinst, depth, ptag, failures)
        if pinst.meta["n"] <= 8 and trial % 5 == 0 and _bfs_answer(pinst) != pinst.answer:
            failures.append(f"{ptag}: BFS cross-check disagrees")
        if _config_str(_pegs_from_pos([pinst.meta["goal_peg"]] * pinst.meta["n"])) == pinst.answer:
            failures.append(f"{ptag}: answer is the goal configuration (trivially guessable)")
        if "moves = [" not in format_cot(pinst):
            failures.append(f"{ptag}: plan trace is not in the published move-list format")

        nd = rng.choice([1, 2, 3, 5, 6])
        dinst = generate(depth, seed, disks=nd, allow_undo=True)
        _common_checks(dinst, depth, f"disks={nd} allow_undo depth={depth} seed={seed}", failures)
        if dinst.meta["n"] != nd:
            failures.append(f"disks knob ignored (got n={dinst.meta['n']}, wanted {nd})")

    # ---- redaction (AMENDMENT 4) ----------------------------------------------------------
    if not REDACTION_MEANINGFUL:
        failures.append("REDACTION_MEANINGFUL should be True for this task")
    red_insts = [_published_exemplar()] + [
        generate(rng.choice(DEPTHS + [1, 3, 5, 7]), rng.randrange(1 << 30),
                 disks=rng.choice([3, 4, 5]))
        for _ in range(20)
    ]
    for inst in red_insts:
        depth = inst.depth
        for k in sorted({0, 1, depth // 2, depth}):
            if not 0 <= k <= depth:
                continue
            tag = f"(redact depth={depth} n={inst.meta['n']} k={k})"
            red = redact_prompt(inst, k)
            if k == 0:
                if red != inst.prompt:
                    failures.append(f"redact_prompt(inst, 0) != inst.prompt {tag}")
                continue
            if red == inst.prompt:
                failures.append(f"redact_prompt did not change the prompt {tag}")
            if REDACTED not in red:
                failures.append(f"redacted prompt has no {REDACTED!r} placeholder {tag}")
            if red.count(REDACTED) != k + 1:
                failures.append(f"expected {k + 1} placeholders, got {red.count(REDACTED)} {tag}")
            if _EXEC_INIT + REDACTED not in red:
                failures.append(f"initial configuration not redacted in place {tag}")
            if inst.meta["start"] in red:
                failures.append(f"initial configuration survived redaction {tag}")
            if _EXEC_QUESTION not in red:
                failures.append(f"the question was removed {tag}")
            for rule in _EXEC_RULES:
                if rule and rule not in red:
                    failures.append(f"static rule text was redacted {tag}")
                    break
            a = red.index(_EXEC_MOVES_OPEN) + len(_EXEC_MOVES_OPEN)
            b = red.index(_EXEC_MOVES_CLOSE, a)
            lines = red[a:b].split("\n")
            expected = ["  " + REDACTED + ","] * k + \
                       [_move_line(tuple(m)) for m in inst.meta["moves"][k:]]
            if lines != expected:
                failures.append(f"move block mis-redacted {tag}")
            if k == depth:
                for m in inst.meta["moves"]:
                    if _triple(tuple(m)) in red:
                        failures.append(f"move {m} survived k=depth {tag}")
                        break
                if re.search(r"\[\s*\d+\s*,", red):
                    failures.append(f"a bracketed integer list survived k=depth {tag}")

    plan_inst = generate(8, 3, formulation="plan")
    if redact_prompt(plan_inst, 0) != plan_inst.prompt:
        failures.append("plan: redact_prompt(inst, 0) != inst.prompt")
    pred = redact_prompt(plan_inst, 4)
    if REDACTED not in pred:
        failures.append("plan: initial configuration not redacted")
    if "• Peg 0:" in pred.split("Goal configuration:")[0]:
        failures.append("plan: initial peg bullets survived redaction")
    if "Goal configuration:" not in pred:
        failures.append("plan: the goal was removed")

    bad = False
    for k in (-1, red_insts[0].depth + 1):
        try:
            redact_prompt(red_insts[0], k)
        except ValueError:
            continue
        bad = True
    if bad:
        failures.append("redact_prompt accepted an out-of-range k")

    # ---- step corruption (AMENDMENT 6) ------------------------------------------------------
    cor_insts = [generate(rng.choice(DEPTHS + [1, 3, 5, 7]), rng.randrange(1 << 30),
                          disks=rng.choice([3, 4, 5])) for _ in range(20)]
    cor_insts += [generate(rng.choice([2, 5, 8, 16]), rng.randrange(1 << 30), formulation="plan")
                  for _ in range(5)]
    cor_insts.append(_published_exemplar())
    for inst in cor_insts:
        depth, n = inst.depth, inst.meta["n"]
        plan = inst.meta.get("formulation", "execute") == "plan"
        for k in sorted({1, depth // 2, depth}):
            if not 1 <= k <= depth:
                continue
            cseed = 100 * k + depth
            tag = f"(corrupt {inst.meta['formulation']} depth={depth} n={n} k={k})"
            text, state = corrupt_step(inst, k, cseed)
            if state == inst.states[k - 1]:
                failures.append(f"corrupted_state == states[k-1] {tag}")
            if text == inst.steps[k - 1]:
                failures.append(f"step_text == steps[k-1] {tag}")
            if corrupt_step(inst, k, cseed) != (text, state):
                failures.append(f"corrupt_step is not deterministic {tag}")
            if inst.meta.get("seed") is not None and corrupt_step(
                    generate(depth, inst.meta["seed"], disks=n,
                             formulation=inst.meta["formulation"]), k, cseed) != (text, state):
                failures.append(f"corrupt_step differs across identical instances {tag}")
            # structure: same step-line shape as the gold trace
            m = (PLAN_STEP_RE if plan else EXEC_STEP_RE).match(text)
            if m is None:
                failures.append(f"corrupted step line is malformed: {text!r} {tag}")
                continue
            disk, s, d = tuple(inst.meta["moves"][k - 1])
            if plan:
                if (int(m.group(1)), int(m.group(2))) != (disk, s):
                    failures.append(f"plan: disk/source were altered {tag}")
                if int(m.group(3)) == d:
                    failures.append(f"plan: destination peg was not corrupted {tag}")
                if int(m.group(3)) not in (0, 1, 2):
                    failures.append(f"plan: destination peg is not a peg index {tag}")
            else:
                if (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))) != \
                   (k, disk, s, d):
                    failures.append(f"execute: the move triple was altered {tag}")
                if m.group(5) != state:
                    failures.append(f"execute: step text and corrupted_state disagree {tag}")

            # the wrong state is a well-formed configuration of the same disks
            pegs = _parse_config(state)
            if pegs is None or len(pegs) != 3:
                failures.append(f"corrupted_state does not parse {tag}")
                continue
            if sorted(d for peg in pegs for d in peg) != list(range(1, n + 1)):
                failures.append(f"corrupted_state is not a configuration of {n} disks {tag}")
            if any(peg != sorted(peg, reverse=True) for peg in pegs):
                failures.append(f"corrupted_state has a larger disk on a smaller one {tag}")
            if _config_str(pegs) != state:
                failures.append(f"corrupted_state is not in canonical bracket form {tag}")

            # and it is one of the two documented corruptions of the previous configuration
            before = inst.meta["start"] if k == 1 else inst.states[k - 2]
            bpegs = _parse_config(before)
            allowed = {_canon(bpegs), _config_str(_pegs_with_disk_on(bpegs, disk, 3 - s - d))}
            if state not in allowed:
                failures.append(f"corrupted_state is not a documented corruption {tag}")
            if not plan:
                third = bpegs[3 - s - d]
                if state != _canon(bpegs) and third and third[-1] < disk:
                    failures.append(f"execute: wrong-peg corruption used a blocked peg {tag}")

    bad = False
    for k in (0, cor_insts[0].depth + 1):
        try:
            corrupt_step(cor_insts[0], k, 0)
        except ValueError:
            continue
        bad = True
    if bad:
        failures.append("corrupt_step accepted an out-of-range k")

    # ---- exemplars and published wording ---------------------------------------------------
    ex = exemplars(3, 0)
    if len(ex) != 3:
        failures.append("exemplars(3, 0) did not return 3 instances")
    else:
        pub = ex[0]
        if not pub.meta.get("published_exemplar"):
            failures.append("exemplars(3, 0)[0] is not the published exemplar")
        if [tuple(m) for m in pub.meta["moves"]] != [
                (1, 0, 2), (2, 0, 1), (1, 2, 1), (3, 0, 2), (1, 1, 0), (2, 1, 2), (1, 0, 2)]:
            failures.append("published exemplar moves do not match the paper's worked example")
        if pub.answer != "[[], [], [3, 2, 1]]":
            failures.append(f"published exemplar answer is {pub.answer!r}")
        for inst in ex:
            if solve(inst) != inst.answer or not check(inst, format_cot(inst)):
                failures.append("an exemplar failed solve()/check()")
            if not inst.prompt or not format_cot(inst):
                failures.append("an exemplar did not render")
    if not APPLE_SYSTEM_PROMPT.startswith("You are a helpful assistant."):
        failures.append("APPLE_SYSTEM_PROMPT not loaded from published_trace.txt")
    if "{N} disks" not in APPLE_USER_PROMPT_TEMPLATE:
        failures.append("APPLE_USER_PROMPT_TEMPLATE not loaded from published_trace.txt")

    # ---- tolerant-parse spot checks --------------------------------------------------------
    probe = generate(5, 7)
    variants = [
        "Answer: " + probe.answer,
        "blah\n**Answer:** " + probe.answer.replace(" ", ""),
        format_cot(probe),
        "\n".join(probe.steps),                                        # no Answer line at all
        _canon([sorted(p) for p in _parse_config(probe.answer)]),      # top-to-bottom order
    ]
    for v in variants:
        if not check(probe, v):
            failures.append(f"tolerant parse failed on {v!r}")
    if check(probe, "Answer: I don't know") or check(probe, "Answer: [[1], [2]]"):
        failures.append("check() accepted an unparseable/short answer")

    if failures:
        print(f"FAIL ({len(failures)} problems)")
        for f in failures[:20]:
            print("  -", f)
        return 1
    print("selftest: 200 execute instances OK (solve==answer, check(gold), check(wrong)==False, "
          "determinism, len(steps)==len(states)==depth, legality, no-undo, prompt round-trip)")
    print("selftest: 60 x {optimal_prefix, plan, disks/allow_undo} OK "
          "(plan cross-checked against BFS over the full state graph)")
    print("selftest: redaction OK for k in {0, 1, depth//2, depth} over 21 instances "
          "(k+1 placeholders, static rules kept, nothing recomputable left at k=depth)")
    print("selftest: corruption OK for k in {1, depth//2, depth} over 26 instances "
          "(wrong-but-legal configuration, move triple untouched in 'execute', destination peg "
          "corrupted in 'plan', step-line structure preserved, determinism)")
    print("selftest: exemplars(3, 0) OK; published exemplar = paper's 3-disk 7-move solution")
    print("PASS")
    return 0


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for seed in range(3):
            inst = generate(depth, seed)
            print("=" * 88)
            print(f"### depth={inst.depth}  seed={seed}  n={inst.meta['n']}  "
                  f"formulation={inst.meta['formulation']}  start={inst.meta['start']}")
            print("-" * 88)
            print("PROMPT:")
            print(inst.prompt)
            print("-" * 88)
            print("GOLD:")
            print(format_cot(inst))
            print()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        _demo()
    else:
        print(__doc__)
        print(f"ANSWER_FORMAT: {ANSWER_FORMAT}")
        print(f"DEPTHS: {DEPTHS}")
        print("usage: python3 task.py [--selftest | --demo]")
