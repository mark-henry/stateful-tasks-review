#!/usr/bin/env python3
"""blocksworld -- Blocks World *state tracking* (given a plan, report the resulting state).

Published format: Stechly, Valmeekam & Kambhampati 2024, "Chain of Thoughtlessness? An
Analysis of CoT in Planning" (arXiv:2405.04776), the `blocksworld_state_tracking` domain of
the companion repo karthikv792/cot-planning, prompt file
`example_query_prompts/blocksworld_state_tracking/upb.txt` (the "Blocksworld Universal
Algorithm" CoT exemplar).  A step is the numbered four-part block

    <n>. Current State: <predicates>
       Action: <action>
       Reason: The above action is applicable ...
       Resulting State: <predicates>

The problem statement wording is PlanBench's plan-execution task
(vendor/LLMs-Planning/plan-bench/prompts/blocksworld/task_7_plan_execution.json:
`[STATEMENT] / [ACTION SEQUENCE] / [RESULTING STATE]`), which is the published *tracking*
(as opposed to planning) formulation of this domain.  See README.md.

Pure stdlib.  `python3 task.py --selftest`, `python3 task.py --demo`.
"""

import argparse
import hashlib
import random
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------------------


@dataclass
class Instance:
    prompt: str
    steps: list[str]
    states: list[str]
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = (
    "the resulting state, written as a list of conditions in the domain's wording, e.g. "
    "`Block B is clear, the hand is empty, Block A is on top of Block C and Block C is on the table`"
)

# Sweep rationale in README.md.  4 blocks (the published exemplars use 3 and 4).
DEPTHS = [2, 4, 6, 10, 16, 24]

KNOBS = {
    "blocks": (4, "number of blocks in the domain (2-26); the paper's own knob, swept 3-20"),
    "include_rules": (True, "prepend the published domain intro (the action rules) to the prompt"),
    "allow_undo": (
        False,
        "allow the random walk to immediately reverse the previous action (trivialises steps)",
    ),
}

# --------------------------------------------------------------------------------------
# Published wording (verbatim from the vendored repos)
# --------------------------------------------------------------------------------------

# vendor/cot-planning/configs/blocksworld_state_tracking.yaml : domain_intro
DOMAIN_INTRO = """I am playing with a set of blocks where I need to arrange the blocks into stacks. Here are the actions I can do

Pick up a block
Unstack a block from on top of another block
Put down a block
Stack a block on top of another block

I have the following restrictions on my actions:
I can only pick up or unstack one block at a time.
I can only pick up or unstack a block if my hand is empty.
I can only pick up a block if the block is on the table and the block is clear. A block is clear if the block has no other blocks on top of it and if the block is not picked up.
I can only unstack a block from on top of another block if the block I am unstacking was really on top of the other block.
I can only unstack a block from on top of another block if the block I am unstacking is clear.
Once I pick up or unstack a block, I am holding the block.
I can only put down a block that I am holding.
I can only stack a block on top of another block if I am holding the block being stacked.
I can only stack a block on top of another block if the block onto which I am stacking the block is clear.
Once I put down or stack a block, my hand becomes empty.
Once you stack a block on top of a second block, the second block is no longer clear."""

REASON_TEMPLATE = (
    "The above action is applicable in the current state because its preconditions; "
    "{preconds}, are satisfied in the current state."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _clear(x: str) -> str:
    return f"Block {x} is clear"


def _ontable(x: str) -> str:
    return f"Block {x} is on the table"


def _on(x: str, y: str) -> str:
    return f"Block {x} is on top of Block {y}"


def _holding(x: str) -> str:
    return f"the hand is currently holding Block {x}"


HANDEMPTY = "the hand is empty"


def _join(items: list[str]) -> str:
    """Published conjunction style: 'a, b, c and d'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _action_nl(action: tuple) -> str:
    kind = action[0]
    if kind == "pickup":
        return f"pick up the Block {action[1]}"
    if kind == "putdown":
        return f"put down the Block {action[1]}"
    if kind == "stack":
        return f"stack the Block {action[1]} on top of the Block {action[2]}"
    if kind == "unstack":
        return f"unstack the Block {action[1]} from on top of the Block {action[2]}"
    raise ValueError(f"unknown action {action!r}")


def _reason_nl(action: tuple) -> str:
    kind = action[0]
    if kind == "pickup":
        pre = [_clear(action[1]), HANDEMPTY, _ontable(action[1])]
    elif kind == "unstack":
        pre = [_clear(action[1]), HANDEMPTY, _on(action[1], action[2])]
    elif kind == "putdown":
        pre = [_holding(action[1])]
    elif kind == "stack":
        pre = [_clear(action[2]), _holding(action[1])]
    else:
        raise ValueError(f"unknown action {action!r}")
    return REASON_TEMPLATE.format(preconds=_join(pre))


# --------------------------------------------------------------------------------------
# Generator-side world model: stacks (bottom-to-top lists) + one held block
# --------------------------------------------------------------------------------------


class _World:
    """Stack-list representation.  Used by generate() only."""

    def __init__(self, stacks: list[list[str]], held: str | None = None):
        self.stacks = [list(s) for s in stacks]
        self.held = held

    def clone(self) -> "_World":
        return _World(self.stacks, self.held)

    def tops(self) -> list[str]:
        return [s[-1] for s in self.stacks]

    def legal_actions(self) -> list[tuple]:
        acts: list[tuple] = []
        if self.held is None:
            for s in self.stacks:
                top = s[-1]
                if len(s) == 1:
                    acts.append(("pickup", top))
                else:
                    acts.append(("unstack", top, s[-2]))
        else:
            acts.append(("putdown", self.held))
            for t in self.tops():
                acts.append(("stack", self.held, t))
        return sorted(acts)

    def apply(self, action: tuple) -> None:
        kind = action[0]
        if kind == "pickup":
            x = action[1]
            for i, s in enumerate(self.stacks):
                if s == [x]:
                    del self.stacks[i]
                    self.held = x
                    return
            raise ValueError(f"illegal {action!r}")
        if kind == "unstack":
            x, y = action[1], action[2]
            for s in self.stacks:
                if len(s) >= 2 and s[-1] == x and s[-2] == y:
                    s.pop()
                    self.held = x
                    return
            raise ValueError(f"illegal {action!r}")
        if kind == "putdown":
            if self.held != action[1]:
                raise ValueError(f"illegal {action!r}")
            self.stacks.append([self.held])
            self.held = None
            return
        if kind == "stack":
            x, y = action[1], action[2]
            if self.held != x:
                raise ValueError(f"illegal {action!r}")
            for s in self.stacks:
                if s[-1] == y:
                    s.append(x)
                    self.held = None
                    return
            raise ValueError(f"illegal {action!r}")
        raise ValueError(f"unknown action {action!r}")

    # --- rendering ---------------------------------------------------------------

    def nl(self) -> str:
        clears = sorted(self.tops())
        conds = [_clear(x) for x in clears]
        conds.append(HANDEMPTY if self.held is None else _holding(self.held))
        ons = []
        tables = []
        for s in self.stacks:
            tables.append(s[0])
            for lower, upper in zip(s, s[1:]):
                ons.append((upper, lower))
        conds += [_on(u, l) for u, l in sorted(ons)]
        conds += [_ontable(x) for x in sorted(tables)]
        return _join(conds)

    def compact(self) -> str:
        """Short canonical state string: stacks bottom-to-top, sorted, '/' held-or-dash."""
        towers = "|".join(sorted("".join(s) for s in self.stacks))
        return f"{towers}/{self.held or '-'}"


# --------------------------------------------------------------------------------------
# Solver-side world model: PDDL predicate set (independent of _World)
# --------------------------------------------------------------------------------------

_PRE = {
    "pickup": lambda a: {("clear", a[1]), ("handempty",), ("ontable", a[1])},
    "unstack": lambda a: {("clear", a[1]), ("handempty",), ("on", a[1], a[2])},
    "putdown": lambda a: {("holding", a[1])},
    "stack": lambda a: {("clear", a[2]), ("holding", a[1])},
}
_DEL = {
    "pickup": lambda a: {("clear", a[1]), ("ontable", a[1]), ("handempty",)},
    "unstack": lambda a: {("on", a[1], a[2]), ("clear", a[1]), ("handempty",)},
    "putdown": lambda a: {("holding", a[1])},
    "stack": lambda a: {("holding", a[1]), ("clear", a[2])},
}
_ADD = {
    "pickup": lambda a: {("holding", a[1])},
    "unstack": lambda a: {("holding", a[1]), ("clear", a[2])},
    "putdown": lambda a: {("clear", a[1]), ("ontable", a[1]), ("handempty",)},
    "stack": lambda a: {("clear", a[1]), ("handempty",), ("on", a[1], a[2])},
}


def _pddl_apply(preds: set, action: tuple) -> set:
    kind = action[0]
    if kind not in _PRE:
        raise ValueError(f"unknown action {action!r}")
    missing = _PRE[kind](action) - preds
    if missing:
        raise ValueError(f"precondition {sorted(missing)} unmet for {action!r}")
    return (preds - _DEL[kind](action)) | _ADD[kind](action)


def _pddl_nl(preds: set) -> str:
    """Render a predicate set in the published prose order."""
    conds = [_clear(x) for (_, x) in sorted(p for p in preds if p[0] == "clear")]
    holding = [p for p in preds if p[0] == "holding"]
    if holding:
        conds.append(_holding(holding[0][1]))
    else:
        conds.append(HANDEMPTY)
    conds += [_on(x, y) for (_, x, y) in sorted(p for p in preds if p[0] == "on")]
    conds += [_ontable(x) for (_, x) in sorted(p for p in preds if p[0] == "ontable")]
    return _join(conds)


# --------------------------------------------------------------------------------------
# Prompt assembly / parsing
# --------------------------------------------------------------------------------------

_INIT_MARKER = "As initial conditions I have that, "
_SEQ_OPEN = "[ACTION SEQUENCE]\n"
_SEQ_CLOSE = "\n[ACTION SEQUENCE END]"

_ACTION_RES = [
    (re.compile(r"^pick up the Block ([A-Z])$"), "pickup"),
    (re.compile(r"^put down the Block ([A-Z])$"), "putdown"),
    (re.compile(r"^stack the Block ([A-Z]) on top of the Block ([A-Z])$"), "stack"),
    (re.compile(r"^unstack the Block ([A-Z]) from on top of the Block ([A-Z])$"), "unstack"),
]


def _build_prompt(init_nl: str, actions: list[tuple], include_rules: bool) -> str:
    body = (
        "[STATEMENT]\n"
        f"{_INIT_MARKER}{init_nl}.\n"
        " I have executed the following action sequence:\n"
        "\n"
        f"{_SEQ_OPEN}"
        + "\n".join(_action_nl(a) for a in actions)
        + _SEQ_CLOSE
        + "\n[RESULTING STATE]"
    )
    if include_rules:
        return DOMAIN_INTRO + "\n\n" + body
    return body


def _parse_conditions_to_preds(text: str) -> set:
    """Strict parse of a published-style state sentence into a PDDL predicate set."""
    text = text.strip().rstrip(".")
    clauses = [c.strip() for c in re.split(r",\s*|\s+and\s+", text) if c.strip()]
    preds: set = set()
    for c in clauses:
        m = re.fullmatch(r"Block ([A-Z]) is clear", c)
        if m:
            preds.add(("clear", m.group(1)))
            continue
        m = re.fullmatch(r"Block ([A-Z]) is on the table", c)
        if m:
            preds.add(("ontable", m.group(1)))
            continue
        m = re.fullmatch(r"Block ([A-Z]) is on top of Block ([A-Z])", c)
        if m:
            preds.add(("on", m.group(1), m.group(2)))
            continue
        if c == HANDEMPTY:
            preds.add(("handempty",))
            continue
        m = re.fullmatch(r"the hand is currently holding Block ([A-Z])", c)
        if m:
            preds.add(("holding", m.group(1)))
            continue
        raise ValueError(f"cannot parse condition {c!r}")
    return preds


def _parse_prompt(prompt: str) -> tuple[set, list[tuple]]:
    i = prompt.index(_INIT_MARKER) + len(_INIT_MARKER)
    j = prompt.index("\n I have executed", i)
    preds = _parse_conditions_to_preds(prompt[i:j])

    a = prompt.index(_SEQ_OPEN) + len(_SEQ_OPEN)
    b = prompt.index(_SEQ_CLOSE, a)
    actions: list[tuple] = []
    for line in prompt[a:b].split("\n"):
        line = line.strip()
        if not line:
            continue
        for rx, kind in _ACTION_RES:
            m = rx.fullmatch(line)
            if m:
                actions.append((kind,) + m.groups())
                break
        else:
            raise ValueError(f"cannot parse action {line!r}")
    return preds, actions


# --------------------------------------------------------------------------------------
# generate / solve / check / format_cot / exemplars
# --------------------------------------------------------------------------------------


def _rng(*parts) -> random.Random:
    """Deterministic RNG: sha256 of the parameter tuple seeds a Mersenne Twister.

    Hashing first (rather than seeding Random with the near-identical parameter string)
    keeps neighbouring (depth, seed) streams obviously independent.
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(), "big"))


def _random_initial_world(rng: random.Random, n: int) -> _World:
    blocks = list(LETTERS[:n])
    rng.shuffle(blocks)
    stacks: list[list[str]] = []
    for b in blocks:
        if not stacks or rng.random() < 0.45:
            stacks.append([b])
        else:
            stacks[rng.randrange(len(stacks))].append(b)
    return _World(stacks)


def _undo_of(action: tuple) -> tuple | None:
    kind = action[0]
    if kind == "pickup":
        return ("putdown", action[1])
    if kind == "putdown":
        return ("pickup", action[1])
    if kind == "stack":
        return ("unstack", action[1], action[2])
    if kind == "unstack":
        return ("stack", action[1], action[2])
    return None


def generate(depth: int, seed: int, **knobs) -> Instance:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    blocks = int(knobs.get("blocks", KNOBS["blocks"][0]))
    include_rules = bool(knobs.get("include_rules", KNOBS["include_rules"][0]))
    allow_undo = bool(knobs.get("allow_undo", KNOBS["allow_undo"][0]))
    if not 2 <= blocks <= 26:
        raise ValueError("blocks must be in 2..26")
    unknown = set(knobs) - set(KNOBS)
    if unknown:
        raise ValueError(f"unknown knobs: {sorted(unknown)}")

    rng = _rng("blocksworld", depth, seed, blocks, allow_undo)
    world = _random_initial_world(rng, blocks)
    init_nl = world.nl()
    init_compact = world.compact()

    actions: list[tuple] = []
    steps: list[str] = []
    states: list[str] = []
    prev: tuple | None = None
    for i in range(depth):
        legal = world.legal_actions()
        if not allow_undo and prev is not None:
            banned = _undo_of(prev)
            filtered = [a for a in legal if a != banned]
            if filtered:
                legal = filtered
        action = legal[rng.randrange(len(legal))]
        before = world.nl()
        world.apply(action)
        steps.append(
            f"{i + 1}. Current State: {before}\n"
            f"   Action: {_action_nl(action)}\n"
            f"   Reason: {_reason_nl(action)}\n"
            f"   Resulting State: {world.nl()}"
        )
        states.append(world.compact())
        actions.append(action)
        prev = action

    answer = world.nl()
    return Instance(
        prompt=_build_prompt(init_nl, actions, include_rules),
        steps=steps,
        states=states,
        answer=answer,
        depth=depth,
        meta={
            "seed": seed,
            "blocks": blocks,
            "include_rules": include_rules,
            "allow_undo": allow_undo,
            "initial_state": init_compact,
            "final_state": world.compact(),
            "actions": [_action_nl(a) for a in actions],
            "published_exemplar": False,
        },
    )


def solve(inst: Instance) -> str:
    """Independent reference solver: re-parse the prompt, simulate with PDDL predicate sets."""
    preds, actions = _parse_prompt(inst.prompt)
    for action in actions:
        preds = _pddl_apply(preds, action)
    return _pddl_nl(preds)


# --- tolerant answer checking ----------------------------------------------------------

_CORE_ON = re.compile(r"^(\w+) is on top of (\w+)$")
_CORE_TABLE = re.compile(r"^(\w+) is (?:on table|on floor|ontable)$")
_CORE_HOLD = re.compile(r"^hand is (?:currently )?holding (\w+)$")
_CORE_HOLD2 = re.compile(r"^(?:i am |i'm |hand[: ]+)?holding[: ]*\s*(\w+)$")


def _core(text: str) -> tuple:
    """Tolerant parse -> (frozenset of on-pairs, frozenset of on-table blocks, held-or-None).

    `clear` and `handempty` are derivable from the core configuration and are ignored, so a
    model is not penalised for omitting (or over-listing) them.  See README.
    """
    s = text.strip()
    s = re.sub(r"\bblock\b", " ", s, flags=re.I)
    s = re.sub(r"\bthe\b", " ", s, flags=re.I)
    s = s.lower().replace("_", " ")
    clauses = [c.strip(" .\t*`-") for c in re.split(r"[,;\n]|\band\b", s)]
    ons: set = set()
    tables: set = set()
    held = None
    for c in clauses:
        c = re.sub(r"\s+", " ", c).strip()
        if not c:
            continue
        m = _CORE_ON.match(c)
        if m:
            x, y = m.group(1), m.group(2)
            if y == "table":
                tables.add(x)
            else:
                ons.add((x, y))
            continue
        m = _CORE_TABLE.match(c)
        if m:
            tables.add(m.group(1))
            continue
        m = _CORE_HOLD.match(c) or _CORE_HOLD2.match(c)
        if m:
            held = m.group(1)
            continue
    return frozenset(ons), frozenset(tables), held


def _extract_answer(completion: str) -> str:
    lines = completion.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = re.search(r"answer\s*:", lines[i], flags=re.I)
        if m:
            tail = lines[i][m.end() :].strip(" *`\t")
            if tail:
                return tail
            return "\n".join(lines[i + 1 :]).strip()
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return ""


def check(inst: Instance, completion: str) -> bool:
    try:
        got = _core(_extract_answer(completion))
        want = _core(inst.answer)
    except Exception:
        return False
    if not got[0] and not got[1] and got[2] is None:
        return False
    return got == want


def format_cot(inst: Instance) -> str:
    return (
        "\n\n".join(inst.steps)
        + f"\n\nFinal State: {inst.answer}\n[PLAN END]\nAnswer: {inst.answer}"
    )


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    text = format_cot(inst)
    spans = []
    cursor = 0
    for step in inst.steps:
        i = text.index(step, cursor)
        spans.append((i, i + len(step)))
        cursor = i + len(step)
    return spans


# --- prompt redaction (AMENDMENT 4) ----------------------------------------------------

REDACTION_MEANINGFUL = True

#: Visible placeholder left behind for every removed item (or span).
REDACTED = "[…]"


def redact_prompt(inst: Instance, k: int) -> str:
    """Return `inst.prompt` with everything needed to *recompute* the state after step k removed.

    Removed: the initial-conditions sentence (one placeholder for the whole span) and the first
    `k` lines of the `[ACTION SEQUENCE]` block (one placeholder each).  Kept: the domain intro
    (static rules, not state), the `[STATEMENT]` / `[ACTION SEQUENCE]` / `[RESULTING STATE]`
    scaffolding, and actions k+1..depth -- everything needed to *continue* from step k+1.

    `k == 0` returns the prompt unchanged; `k == inst.depth` leaves only the question.
    """
    if not 0 <= k <= inst.depth:
        raise ValueError(f"k must be in 0..{inst.depth}, got {k}")
    if k == 0:
        return inst.prompt

    prompt = inst.prompt

    # 1. the initial state: one placeholder for the removed span, trailing '.' kept.
    i = prompt.index(_INIT_MARKER) + len(_INIT_MARKER)
    j = prompt.index("\n I have executed", i)
    prompt = prompt[:i] + REDACTED + "." + prompt[j:]

    # 2. the first k action lines: one placeholder each.
    a = prompt.index(_SEQ_OPEN) + len(_SEQ_OPEN)
    b = prompt.index(_SEQ_CLOSE, a)
    lines = prompt[a:b].split("\n")
    if len(lines) != inst.depth:
        raise ValueError(f"prompt has {len(lines)} action lines, expected {inst.depth}")
    lines[:k] = [REDACTED] * k
    return prompt[:a] + "\n".join(lines) + prompt[b:]


# --- exemplars -------------------------------------------------------------------------

# Example 1 of published_trace.txt (== vendor/cot-planning/example_query_prompts/
# blocksworld_state_tracking/upb.txt): 3 blocks, C on A, B on table; 6 actions.
_PUBLISHED_STACKS = [["A", "C"], ["B"]]
_PUBLISHED_ACTIONS = [
    ("unstack", "C", "A"),
    ("putdown", "C"),
    ("pickup", "A"),
    ("stack", "A", "C"),
    ("pickup", "B"),
    ("stack", "B", "A"),
]


def published_exemplar(include_rules: bool = True) -> Instance:
    """The verbatim published worked example, re-cast as a tracking (given-actions) instance."""
    world = _World(_PUBLISHED_STACKS)
    init_nl = world.nl()
    steps, states = [], []
    for i, action in enumerate(_PUBLISHED_ACTIONS):
        before = world.nl()
        world.apply(action)
        steps.append(
            f"{i + 1}. Current State: {before}\n"
            f"   Action: {_action_nl(action)}\n"
            f"   Reason: {_reason_nl(action)}\n"
            f"   Resulting State: {world.nl()}"
        )
        states.append(world.compact())
    return Instance(
        prompt=_build_prompt(init_nl, _PUBLISHED_ACTIONS, include_rules),
        steps=steps,
        states=states,
        answer=world.nl(),
        depth=len(_PUBLISHED_ACTIONS),
        meta={
            "seed": None,
            "blocks": 3,
            "include_rules": include_rules,
            "allow_undo": True,
            "initial_state": _World(_PUBLISHED_STACKS).compact(),
            "final_state": world.compact(),
            "actions": [_action_nl(a) for a in _PUBLISHED_ACTIONS],
            "published_exemplar": True,
            "source": "arXiv:2405.04776 companion repo, blocksworld_state_tracking/upb.txt, example 1",
        },
    )


def exemplars(k: int, seed: int) -> list[Instance]:
    if k <= 0:
        return []
    out = [published_exemplar()]
    rng = _rng("blocksworld-exemplars", seed)
    for _ in range(k - 1):
        out.append(generate(depth=4, seed=rng.randrange(1 << 30)))
    return out[:k]


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------


def _published_trace_steps() -> list[str] | None:
    """Step blocks of example 1 in published_trace.txt, if the file is present."""
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_trace.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index("[PLAN]\n") + len("[PLAN]\n")
    end = text.index("\nFinal State:", start)
    return [b.rstrip("\n") for b in text[start:end].strip("\n").split("\n\n")]


def _selftest() -> int:
    rng = random.Random(20260916)
    failures: list[str] = []
    n = 200

    # 0. the published exemplar must round-trip verbatim
    pub = published_exemplar()
    trace = _published_trace_steps()
    if trace is None:
        print("[warn] published_trace.txt not found; skipping verbatim-format check")
    else:
        if trace != pub.steps:
            for a, b in zip(trace, pub.steps):
                if a != b:
                    failures.append(f"published trace mismatch:\n  want {a!r}\n  got  {b!r}")
                    break
            else:
                failures.append(f"published trace length {len(trace)} != {len(pub.steps)}")
        else:
            print(f"published exemplar: {len(pub.steps)}/{len(trace)} step blocks match verbatim")
    if solve(pub) != pub.answer:
        failures.append("solve() != answer on the published exemplar")
    if not check(pub, format_cot(pub)):
        failures.append("check() failed on the published exemplar")

    # tolerance battery: surface variants a model plausibly emits
    accept = [
        "Answer: " + pub.answer,
        "Answer: " + pub.answer + ".",
        "**Answer:** " + pub.answer,
        "answer: Block A is on top of Block C, Block B is on top of Block A "
        "and Block C is on the table",
        "Answer: B is on top of A, A is on top of C, C is on the table",
        "Final State: ...\nAnswer:\n" + pub.answer,
        pub.answer,
    ]
    reject = [
        "Answer: Block A is on top of Block B, Block B is on top of Block C "
        "and Block C is on the table",
        "Answer: Block A is on top of Block C and Block C is on the table",
        "Answer: the hand is empty",
        "Answer: I don't know",
        "",
    ]
    for v in accept:
        if not check(pub, v):
            failures.append(f"check rejected an acceptable variant: {v[:60]!r}")
    for v in reject:
        if check(pub, v):
            failures.append(f"check accepted a wrong answer: {v[:60]!r}")

    for t in range(n):
        depth = rng.choice(DEPTHS + [1, 3, 5, 7, 9, 13, 20, 30])
        seed = rng.randrange(1 << 30)
        blocks = rng.choice([2, 3, 4, 5, 6, 8, 12])
        inst = generate(depth, seed, blocks=blocks)
        tag = f"(depth={depth}, seed={seed}, blocks={blocks})"

        if len(inst.steps) != depth or len(inst.states) != depth:
            failures.append(f"len(steps)/len(states) != depth {tag}")
        if inst.depth != depth:
            failures.append(f"inst.depth != depth {tag}")

        got = solve(inst)
        if got != inst.answer:
            failures.append(f"solve() != answer {tag}\n  want {inst.answer!r}\n  got  {got!r}")

        if not check(inst, format_cot(inst)):
            failures.append(f"check(format_cot) is False {tag}")

        # a wrong answer must be rejected
        for bump in range(1, 40):
            other = generate(depth, seed + bump, blocks=blocks)
            if _core(other.answer) != _core(inst.answer):
                if check(inst, f"Answer: {other.answer}"):
                    failures.append(f"check accepted a wrong answer {tag}")
                break
        else:
            failures.append(f"could not build a wrong answer {tag}")

        # determinism
        again = generate(depth, seed, blocks=blocks)
        if (again.prompt, again.steps, again.states, again.answer) != (
            inst.prompt,
            inst.steps,
            inst.states,
            inst.answer,
        ):
            failures.append(f"generate is not deterministic {tag}")

        # step_spans point at the steps
        text = format_cot(inst)
        for (lo, hi), step in zip(step_spans(inst), inst.steps):
            if text[lo:hi] != step:
                failures.append(f"step_spans misaligned {tag}")
                break

        # tolerance: reordered / clear-free / lowercase-ish restatement is still accepted
        preds, actions = _parse_prompt(inst.prompt)
        for a in actions:
            preds = _pddl_apply(preds, a)
        core_only = [
            _on(x, y) for (_, x, y) in sorted(p for p in preds if p[0] == "on")
        ] + [_ontable(x) for (_, x) in sorted(p for p in preds if p[0] == "ontable")]
        held = [p for p in preds if p[0] == "holding"]
        if held:
            core_only.append(_holding(held[0][1]))
        core_only.reverse()
        if not check(inst, "Answer: " + _join(core_only)):
            failures.append(f"check rejected a clear-free restatement {tag}")

        if t == 0:
            for extra in ("", "   \n\n", "nonsense"):
                if check(inst, extra):
                    failures.append("check accepted empty/nonsense output")

    # --- redaction (AMENDMENT 4) ------------------------------------------------------
    n_red = 0
    red_insts = [published_exemplar()] + [
        generate(
            rng.choice(DEPTHS + [1, 3, 5, 7]),
            rng.randrange(1 << 30),
            blocks=rng.choice([2, 3, 4, 6, 8]),
        )
        for _ in range(20)
    ]
    for inst in red_insts:
        depth = inst.depth
        for k in sorted({0, 1, depth // 2, depth}):
            if not 0 <= k <= depth:
                continue
            n_red += 1
            tag = f"(redact depth={depth}, blocks={inst.meta['blocks']}, k={k})"
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
                failures.append(
                    f"expected {k + 1} placeholders, got {red.count(REDACTED)} {tag}"
                )

            # the initial state is gone, its scaffolding is not
            if _INIT_MARKER + REDACTED + "." not in red:
                failures.append(f"initial state not redacted in place {tag}")
            if "[RESULTING STATE]" not in red:
                failures.append(f"the question was removed {tag}")
            if inst.meta["include_rules"] and DOMAIN_INTRO not in red:
                failures.append(f"static rule table was redacted {tag}")

            # the action block: k placeholders then actions k+1..depth verbatim
            a = red.index(_SEQ_OPEN) + len(_SEQ_OPEN)
            b = red.index(_SEQ_CLOSE, a)
            lines = red[a:b].split("\n")
            if lines != [REDACTED] * k + inst.meta["actions"][k:]:
                failures.append(f"action block mis-redacted {tag}")

            # k == depth: no initial-state or operator tokens survive anywhere
            if k == depth:
                leak = re.search(r"Block [A-Z]", red)
                if leak:
                    failures.append(f"block token {leak.group(0)!r} survived k=depth {tag}")
                for act in inst.meta["actions"]:
                    if act in red:
                        failures.append(f"operator {act!r} survived k=depth {tag}")
                        break

    bad = False
    for k in (-1, red_insts[0].depth + 1):
        try:
            redact_prompt(red_insts[0], k)
        except ValueError:
            continue
        bad = True
    if bad:
        failures.append("redact_prompt accepted an out-of-range k")

    ex = exemplars(3, 0)
    if len(ex) != 3:
        failures.append("exemplars(3, 0) did not return 3 instances")
    else:
        if not ex[0].meta.get("published_exemplar"):
            failures.append("exemplars(3, 0)[0] is not the published exemplar")
        for e in ex:
            if not e.prompt or not format_cot(e):
                failures.append("exemplar failed to render")
            if solve(e) != e.answer:
                failures.append("exemplar solve() != answer")

    print(
        f"ran {n} random (depth, seed, blocks) instances + {len(ex)} exemplars "
        f"+ {n_red} redactions over {len(red_insts)} instances"
    )
    if failures:
        print(f"FAIL: {len(failures)} failure(s)")
        for f in failures[:10]:
            print("  - " + f)
        return 1
    print("selftest OK")
    return 0


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, seed=1000 + i)
            print("=" * 88)
            print(f"### depth={depth}  seed={1000 + i}  blocks={inst.meta['blocks']}")
            print("=" * 88)
            print("--- PROMPT ---")
            print(inst.prompt)
            print("--- GOLD CoT ---")
            print(format_cot(inst))
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
