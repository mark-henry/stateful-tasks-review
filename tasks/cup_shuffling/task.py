#!/usr/bin/env python3
"""cup_shuffling -- BBH `tracking_shuffled_objects` (three / five / seven objects).

Published source: Suzgun et al. 2022, "Challenging BIG-Bench Tasks and Whether
Chain-of-Thought Can Solve Them" (arXiv:2210.09261), github.com/suzgunmirac/BIG-Bench-Hard
commit 9ee07bd481feebf959a6b59d61ea57bdcf30964d.  Underlying BIG-bench task:
`tracking_shuffled_objects` by James Simon (google/BIG-bench).

The published CoT format (vendor/bbh/cot-prompts/*.txt, all three subtasks share the
same 3-shot prefix) is reproduced verbatim:

    (0) At the start: Alice: yellow, Bob: blue, Claire: pink.
    (1) Claire and Alice swap balls: Alice: pink, Bob: blue, Claire: yellow.
    ...
    At the end of the game, Bob has the yellow ball. So the answer is (A).

Pure python + stdlib.  No network at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR_BBH = os.path.join(HERE, "vendor", "bbh")
TRACE_PATH = os.path.join(HERE, "published_trace.txt")

# --------------------------------------------------------------------------------------
# interface dataclass
# --------------------------------------------------------------------------------------


@dataclass
class Instance:
    prompt: str          # problem statement only, published BBH wording (incl. Options block)
    steps: list          # gold trace, one element per swap, published format
    states: list         # tracked state after each step (len == len(steps))
    answer: str          # exact-match target, the BBH option letter e.g. "(A)"
    depth: int           # number of swaps
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a multiple-choice option letter in parentheses, like (A)"

# Redaction is meaningful here: the initial assignment and each swap clause are separable
# pieces of the prompt, and steps 1..k consume exactly the first k swap clauses.
REDACTION_MEANINGFUL = True

# depth = number of pairwise swaps.  BBH itself only ever uses depth == n (3, 5, 7).
DEPTHS = [2, 3, 5, 8, 12, 16, 24]

KNOBS = {
    "objects": (3, "number of people/objects n, 3-7 (BBH publishes n=3, 5 and 7)"),
    "scenario": (
        "mixed",
        "one of 'ball', 'gift', 'book', 'dance', 'soccer', or 'mixed' to sample one "
        "of the five published contexts per instance (as the BBH files themselves do)",
    ),
}

# --------------------------------------------------------------------------------------
# the five published contexts, transcribed from vendor/bbh/data/*.json
# --------------------------------------------------------------------------------------

NAMES = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude"]

SCENARIOS = {
    "ball": {
        "intro": "{people} are playing a game. At the start of the game, they are each holding a ball: {assign}.",
        "assign": "{p} has {a_item}",
        "conn": "As the game progresses, pairs of players trade balls.",
        "swap": "{a} and {b} swap balls",
        "end": "At the end of the game, {p} has the",
        "items": [
            "black ball", "blue ball", "brown ball", "green ball", "orange ball",
            "pink ball", "purple ball", "red ball", "white ball", "yellow ball",
        ],
    },
    "gift": {
        "intro": "{people} are holding a white elephant gift exchange. At the start of the event, "
                 "they are each holding a present of a different color: {assign}.",
        "assign": "{p} has {a_item}",
        "conn": "As the event progresses, pairs of people swap gifts.",
        "swap": "{a} and {b} swap their gifts",
        "end": "At the end of the event, {p} has the",
        "items": [
            "black present", "blue present", "brown present", "green present", "orange present",
            "pink present", "purple present", "red present", "white present", "yellow present",
        ],
    },
    "book": {
        "intro": "{people} are friends and avid readers who occasionally trade books. At the start "
                 "of the semester, they each buy one new book: {assign}.",
        "assign": "{p} gets {item}",
        "conn": "As the semester proceeds, they start trading around the new books.",
        "swap": "{a} and {b} swap books",
        "end": "At the end of the semester, {p} has",
        "items": [
            "Catch-22", "Frankenstein", "Hound of the Baskervilles", "Lolita", "Moby Dick",
            "The Fellowship of the Ring", "The Great Gatsby", "The Odyssey", "The Pearl", "Ulysses",
        ],
    },
    "dance": {
        "intro": "{people} are dancers at a square dance. At the start of a song, they each have a "
                 "partner: {assign}.",
        "assign": "{p} is dancing with {item}",
        "conn": "Throughout the song, the dancers often trade partners.",
        "swap": "{a} and {b} switch partners",
        "end": "At the end of the dance, {p} is dancing with",
        "items": [
            "Helga", "Izzi", "Jamie", "Karl", "Lola",
            "Melissa", "Ophelia", "Patrick", "Rodrigo", "Sam",
        ],
    },
    "soccer": {
        "intro": "{people} are on the same team in a soccer match. At the start of the match, they "
                 "are each assigned to a position: {assign}.",
        "assign": "{p} is playing {item}",
        "conn": "As the game progresses, pairs of players occasionally swap positions.",
        "swap": "{a} and {b} trade positions",
        "end": "At the end of the match, {p} is playing",
        "items": [
            "benchwarmer", "center midfielder", "cheerleader", "fullback", "goalkeeper",
            "left midfielder", "left winger", "right midfielder", "right winger", "striker",
        ],
    },
}

LETTERS = "ABCDEFG"


def _short(item: str) -> str:
    """Short state token for an item, exactly as the published scratchpad writes it.

    'yellow ball' -> 'yellow' ; 'red present' -> 'red' ; 'Lola' -> 'Lola'.
    """
    for noun in (" ball", " present"):
        if item.endswith(noun):
            return item[: -len(noun)]
    return item


def _article(item: str) -> str:
    return "an " if item[0].lower() in "aeiou" else "a "


def _join_people(names) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return "%s and %s" % tuple(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _join_clauses(clauses) -> str:
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return "%s and %s" % tuple(clauses)
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1]


def _ordinals(k: int):
    """BBH's swap-sentence connectives: First, Then, ..., Finally."""
    if k == 1:
        return ["First"]
    return ["First"] + ["Then"] * (k - 2) + ["Finally"]


# --------------------------------------------------------------------------------------
# instance construction (shared by the generator and the vendored-BBH loader)
# --------------------------------------------------------------------------------------


def _build(people, items, swap_texts, pairs, end_clause, query_idx, meta):
    """Assemble an Instance from already-rendered pieces.

    `people`      list of names, in initial-assignment order (option k <- people[k]'s item)
    `items`       full item phrases, in the same order (items[k] is option LETTERS[k])
    `swap_texts`  rendered 'X and Y swap balls' strings, one per swap
    `pairs`       (i, j) index pairs matching swap_texts
    `end_clause`  'At the end of the game, Bob has the'  (no trailing item)
    `query_idx`   index into `people` of the person being asked about
    """
    n = len(people)
    holding = list(range(n))  # holding[p] = index of the item person p currently holds
    steps, states = [], []
    initial_state = ", ".join("%s: %s" % (people[p], _short(items[holding[p]])) for p in range(n))
    for k, ((i, j), text) in enumerate(zip(pairs, swap_texts), start=1):
        holding[i], holding[j] = holding[j], holding[i]
        state = ", ".join("%s: %s" % (people[p], _short(items[holding[p]])) for p in range(n))
        steps.append("(%d) %s: %s." % (k, text, state))
        states.append(" ".join(_short(items[holding[p]]) for p in range(n)))

    final_item = items[holding[query_idx]]
    answer = "(%s)" % LETTERS[holding[query_idx]]

    prompt_lines = meta.pop("_prompt_lines")
    prompt = "\n".join(prompt_lines)

    meta = dict(meta)
    meta.update(
        {
            "people": list(people),
            "items": list(items),
            "pairs": [list(p) for p in pairs],
            "query_person": people[query_idx],
            "answer_object": final_item,
            "answer_object_short": _short(final_item),
            "initial_line": "(0) At the start: %s." % initial_state,
            "closing": "%s %s." % (end_clause, final_item),
            "options": {LETTERS[k]: items[k] for k in range(n)},
        }
    )
    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=answer,
        depth=len(steps),
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------


def _sample_swaps(rng, n, depth):
    """BIG-bench's two swap-sequence constraints (README 'Swap sequence generation'):

    1. every person is involved in at least one swap  (only enforceable when
       depth >= ceil(n/2); relaxed below that, see README caveats)
    2. the same two people never swap twice in a row
    """
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    feasible = depth * 2 >= n
    for _ in range(4000):
        pairs, prev = [], None
        for _ in range(depth):
            p = rng.choice([q for q in all_pairs if q != prev])
            pairs.append(p)
            prev = p
        if not feasible or len({x for p in pairs for x in p}) == n:
            return pairs
    raise RuntimeError("could not sample a swap sequence for n=%d depth=%d" % (n, depth))


def generate(depth: int, seed: int, **knobs) -> Instance:
    objects = int(knobs.get("objects", KNOBS["objects"][0]))
    scenario = knobs.get("scenario", KNOBS["scenario"][0])
    unknown = set(knobs) - set(KNOBS)
    if unknown:
        raise TypeError("unknown knobs: %s" % sorted(unknown))
    if depth < 1:
        raise ValueError("depth must be >= 1")
    if not 3 <= objects <= len(NAMES):
        raise ValueError("objects must be in 3..%d" % len(NAMES))
    if scenario != "mixed" and scenario not in SCENARIOS:
        raise ValueError("scenario must be 'mixed' or one of %s" % sorted(SCENARIOS))

    rng = random.Random("%d|%d|%d|%s" % (seed, depth, objects, scenario))

    name = scenario if scenario != "mixed" else rng.choice(sorted(SCENARIOS))
    sc = SCENARIOS[name]
    n = objects
    people = NAMES[:n]
    items = rng.sample(sc["items"], n)
    pairs = _sample_swaps(rng, n, depth)
    query_idx = rng.randrange(n)

    assign = _join_clauses(
        [
            sc["assign"].format(p=people[k], item=items[k], a_item=_article(items[k]) + items[k])
            for k in range(n)
        ]
    )
    intro = sc["intro"].format(people=_join_people(people), assign=assign)
    swap_texts = [sc["swap"].format(a=people[i], b=people[j]) for i, j in pairs]
    sentences = " ".join(
        "%s, %s." % (word, text) for word, text in zip(_ordinals(depth), swap_texts)
    )
    end_clause = sc["end"].format(p=people[query_idx])
    line2 = "%s %s %s" % (sc["conn"], sentences, end_clause)
    options = ["(%s) %s" % (LETTERS[k], items[k]) for k in range(n)]

    return _build(
        people,
        items,
        swap_texts,
        pairs,
        end_clause,
        query_idx,
        {
            "_prompt_lines": [intro, line2, "Options:"] + options,
            "scenario": name,
            "objects": n,
            "seed": seed,
            "source": "generated",
        },
    )


# --------------------------------------------------------------------------------------
# solve -- independent reference solver, parses the prompt and traces BACKWARDS
# --------------------------------------------------------------------------------------

_SWAP_RE = re.compile(r"(?:First|Then|Finally), ([A-Z][a-z]+) and ([A-Z][a-z]+) ")
_END_RE = re.compile(r"At the end of [^,]+, ([A-Z][a-z]+) ")


def solve(inst: Instance) -> str:
    """Reference solver.  Reads ONLY `inst.prompt` (never meta / steps / states), and uses
    a different algorithm from generate(): instead of simulating the permutation forward,
    it walks the swap list in reverse from the queried person to whoever started with the
    object that person ends up with.  Option letter k is, by BBH convention, the object
    initially held by the k-th person listed.
    """
    lines = inst.prompt.split("\n")
    people = re.split(r",\s+and\s+|,\s+|\s+and\s+", lines[0].split(" are ")[0])
    people = [p.strip() for p in people if p.strip()]

    pairs = [(people.index(a), people.index(b)) for a, b in _SWAP_RE.findall(lines[1])]
    query = _END_RE.findall(lines[1])[-1]

    pos = people.index(query)
    for i, j in reversed(pairs):
        if pos == i:
            pos = j
        elif pos == j:
            pos = i
    return "(%s)" % LETTERS[pos]


# --------------------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------------------

_ANSWER_LINE_RE = re.compile(r"^\s*Answer\s*:\s*(.*)$", re.IGNORECASE)
_LETTER_ONLY_RE = re.compile(r"\(?\s*([A-Ga-g])\s*\)?\s*\.?")
_SO_THE_ANSWER_RE = re.compile(r"answer\s+is\s*\(?\s*([A-Ga-g])\s*\)?", re.IGNORECASE)
_LEADING_OPTION_RE = re.compile(r"^\(([A-Ga-g])\)")


def _extract(completion: str):
    """Harness normalization: last `Answer:` line, else the last non-empty line."""
    tail = None
    for line in completion.split("\n"):
        m = _ANSWER_LINE_RE.match(line)
        if m:
            tail = m.group(1)
    if tail is None:
        nonempty = [l for l in completion.split("\n") if l.strip()]
        if not nonempty:
            return None
        tail = nonempty[-1]
    return tail.strip()


def check(inst: Instance, completion: str) -> bool:
    tail = _extract(completion)
    if not tail:
        return False
    letter = None
    m = _LETTER_ONLY_RE.fullmatch(tail)          # "(A)" / "A" / "A." / "(A)."
    if m is None:
        m = _SO_THE_ANSWER_RE.search(tail)       # "...So the answer is (A)."
    if m is None:
        m = _LEADING_OPTION_RE.match(tail)       # "(A) yellow ball"
    if m is not None:
        letter = m.group(1).upper()
    else:
        # last resort: the model named the object instead of the letter
        opts = inst.meta.get("options", {})
        hits = [k for k, v in opts.items() if v.lower() == tail.lower().rstrip(".")]
        if len(hits) == 1:
            letter = hits[0]
    if letter is None:
        return False
    return "(%s)" % letter == inst.answer


# --------------------------------------------------------------------------------------
# format_cot / step_spans
# --------------------------------------------------------------------------------------


def format_cot(inst: Instance) -> str:
    body = "\n".join([inst.meta["initial_line"]] + list(inst.steps))
    return "%s\n%s So the answer is %s.\nAnswer: %s" % (
        body,
        inst.meta["closing"],
        inst.answer,
        inst.answer,
    )


def step_spans(inst: Instance):
    """Char offsets (start, end) of each step within format_cot(inst)."""
    text = format_cot(inst)
    spans, cursor = [], 0
    for step in inst.steps:
        start = text.index(step, cursor)
        spans.append((start, start + len(step)))
        cursor = start + len(step)
    return spans


# --------------------------------------------------------------------------------------
# redact_prompt -- prompt blinding (SPEC AMENDMENT 4)
# --------------------------------------------------------------------------------------

PLACEHOLDER = "[\u2026]"  # the literal string "[...]" with a horizontal ellipsis

# a whole swap clause, including its "First, " / "Then, " / "Finally, " connective and the
# trailing period: "Then, Alice and Bob swap balls."
_SWAP_CLAUSE_RE = re.compile(
    r"(?:First|Then|Finally),\s+[A-Z][a-z]+ and [A-Z][a-z]+ [a-z ]+\."
)


def redact_prompt(inst: Instance, k: int) -> str:
    """Return `inst.prompt` with everything needed to RECOMPUTE the state after swap k removed.

    Removed (one PLACEHOLDER per removed item):
      * the initial assignment clause -- who starts with which object -- i.e. everything after
        the colon in the opening sentence ("... they are each holding a ball: [...].");
      * the first `k` swap clauses in the second line.

    Kept verbatim: the cast and the scenario lead-in, swap clauses k+1..depth, the question
    ("At the end of the game, Bob has the") and the whole `Options:` block. The option list is
    static material -- it is the answer alphabet, and by BBH convention its ORDER encodes the
    initial assignment, but nothing in the prompt states that convention once the assignment
    sentence is gone.

    `redact_prompt(inst, 0)` returns `inst.prompt` unchanged; `redact_prompt(inst, inst.depth)`
    leaves no initial state and no operators at all.
    """
    if not 0 <= k <= inst.depth:
        raise ValueError("k must be in 0..%d, got %d" % (inst.depth, k))
    if k == 0:
        return inst.prompt

    lines = inst.prompt.split("\n")
    if len(lines) < 3 or not lines[2].startswith("Options:"):
        raise ValueError("unexpected prompt shape: no Options: block on line 3")

    # (a) the initial assignment: everything after the first ": " of the opening line.
    intro = lines[0]
    colon = intro.find(": ")
    if colon < 0 or not intro.endswith("."):
        raise ValueError("unexpected opening sentence: %r" % intro)
    lines[0] = "%s %s." % (intro[: colon + 1], PLACEHOLDER)

    # (b) the first k swap clauses of the second line.
    swaps = lines[1]
    spans = [m.span() for m in _SWAP_CLAUSE_RE.finditer(swaps)]
    if len(spans) != inst.depth:
        raise ValueError("found %d swap clauses, expected depth=%d" % (len(spans), inst.depth))
    out, cursor = [], 0
    for start, end in spans[:k]:
        out.append(swaps[cursor:start])
        out.append(PLACEHOLDER)
        cursor = end
    out.append(swaps[cursor:])
    lines[1] = "".join(out)

    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# corrupt_step -- mistake propagation (SPEC AMENDMENT 6)
# --------------------------------------------------------------------------------------

# "(3) Alice and Claire swap balls: Alice: pink, Bob: blue, Claire: yellow."
#  \_______ action, left untouched ______/  \________ reported state ________/
_STEP_RE = re.compile(r"^\((\d+)\) (.+?): ((?:[A-Z][a-z]+: [^,]+)(?:, [A-Z][a-z]+: [^,]+)*)\.$")


def _parse_step(step: str):
    """Split a published step line into (action_prefix, [names], [short items]).

    The action clause ('Alice and Claire swap balls', '... switch partners', '... trade
    positions') never contains a colon in any of the five published scenarios, so the first
    ': ' is the action/state boundary.  Short items may contain spaces ('center midfielder')
    but never commas, so the assignment splits on ', '.
    """
    m = _STEP_RE.match(step)
    if m is None:
        raise ValueError("unrecognized step line: %r" % step)
    prefix = "(%s) %s" % (m.group(1), m.group(2))
    names, shorts = [], []
    for part in m.group(3).split(", "):
        who, what = part.split(": ", 1)
        names.append(who)
        shorts.append(what)
    return prefix, names, shorts


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple:
    """Return (step_text, corrupted_state) for step `k` (1-based) with a plausible wrong state.

    The minimal edit for a permutation state is a transposition: two people's objects are
    exchanged relative to the truth, so the reported assignment is still a legal assignment
    (same people, same objects, each held exactly once) but is the wrong one.  This is also
    the natural human/model slip here -- applying the swap to the wrong pair, or forgetting
    to apply it (the pair being swapped is one of the candidate transpositions).

    The action clause '(k) X and Y swap balls:' is left byte-identical; only the reported
    assignment after it changes.  `corrupted_state` is in the same canonical form as
    `inst.states` (short item names, space-joined, in the prompt's person order).

    Deterministic in (inst, k, seed).
    """
    if not 1 <= k <= inst.depth:
        raise ValueError("k must be in 1..%d, got %d" % (inst.depth, k))

    prefix, names, shorts = _parse_step(inst.steps[k - 1])
    n = len(shorts)

    # candidate transpositions: any pair holding different objects (all pairs, in practice,
    # since every instance deals out distinct objects -- the filter is belt-and-braces so a
    # degenerate instance can never yield corrupted_state == states[k-1]).
    cands = [(i, j) for i in range(n) for j in range(i + 1, n) if shorts[i] != shorts[j]]
    if not cands:
        raise ValueError("step %d reports %d identical objects; nothing to corrupt" % (k, n))

    rng = random.Random("corrupt|%s|%d|%d" % (inst.prompt, k, seed))
    i, j = rng.choice(cands)
    shorts = list(shorts)
    shorts[i], shorts[j] = shorts[j], shorts[i]

    step_text = "%s: %s." % (
        prefix,
        ", ".join("%s: %s" % (names[p], shorts[p]) for p in range(n)),
    )
    return step_text, " ".join(shorts)


# --------------------------------------------------------------------------------------
# vendored fixed BBH set + the verbatim published exemplar
# --------------------------------------------------------------------------------------

_BBH_FILES = {
    3: "tracking_shuffled_objects_three_objects.json",
    5: "tracking_shuffled_objects_five_objects.json",
    7: "tracking_shuffled_objects_seven_objects.json",
}


def _instance_from_bbh_input(text: str, meta: dict) -> Instance:
    """Rebuild a full Instance (with gold CoT) from a raw BBH `input` string."""
    lines = text.split("\n")
    people = re.split(r",\s+and\s+|,\s+|\s+and\s+", lines[0].split(" are ")[0])
    people = [p.strip() for p in people if p.strip()]
    items = [re.sub(r"^\([A-G]\)\s*", "", l) for l in lines[3:] if l.strip()]
    swap_texts, pairs = [], []
    for m in re.finditer(r"(?:First|Then|Finally), ([A-Z][a-z]+) and ([A-Z][a-z]+) ([a-z ]+)\.",
                         lines[1]):
        a, b, verb = m.group(1), m.group(2), m.group(3)
        swap_texts.append("%s and %s %s" % (a, b, verb))
        pairs.append((people.index(a), people.index(b)))
    end_clause = "At the end of" + lines[1].split(". At the end of")[-1]
    query = _END_RE.findall(lines[1])[-1]
    meta = dict(meta)
    meta["_prompt_lines"] = lines
    return _build(people, items, swap_texts, pairs, end_clause, people.index(query), meta)


def load_bbh(objects: int = 3):
    """The vendored, FIXED BBH eval set (contaminated; use generate() for fresh instances).

    Returns a list of Instances; gold CoT is reconstructed in the published format, and
    `meta['bbh_target']` carries BBH's own published answer letter.
    """
    path = os.path.join(VENDOR_BBH, "data", _BBH_FILES[objects])
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for idx, ex in enumerate(data["examples"]):
        out.append(
            _instance_from_bbh_input(
                ex["input"],
                {"source": "bbh", "objects": objects, "bbh_index": idx,
                 "bbh_target": ex["target"]},
            )
        )
    return out


def _published_exemplar() -> Instance:
    """exemplars()[0]: the first BBH 3-shot CoT exemplar, verbatim from published_trace.txt."""
    with open(TRACE_PATH, encoding="utf-8") as fh:
        raw = fh.read()
    body = raw.split("\n---\n", 1)[1]
    lines = body.rstrip("\n").split("\n")
    qi = next(i for i, l in enumerate(lines) if l.startswith("Q: "))
    ai = next(i for i, l in enumerate(lines) if l.startswith("A: "))
    prompt_lines = [lines[qi][3:]] + lines[qi + 1:ai]
    cot = lines[ai + 1:]

    initial_line = cot[0]
    steps = [l for l in cot[1:] if re.match(r"^\(\d+\)", l)]
    closing_full = cot[len(steps) + 1]
    closing, tail = closing_full.rsplit(" So the answer is ", 1)
    answer = tail.rstrip(".")

    people = [m.group(1) for m in re.finditer(r"([A-Z][a-z]+): ", initial_line)]
    options = {}
    for l in prompt_lines:
        m = re.match(r"^\(([A-G])\)\s*(.*)$", l)
        if m:
            options[m.group(1)] = m.group(2)
    states = []
    for s in steps:
        state = s.split(": ", 1)[1].rstrip(".")
        states.append(" ".join(part.split(": ")[1] for part in state.split(", ")))
    answer_object = options[answer.strip("()")]

    return Instance(
        prompt="\n".join(prompt_lines),
        steps=steps,
        states=states,
        answer=answer,
        depth=len(steps),
        meta={
            "source": "published_trace.txt (BBH cot-prompts, verbatim)",
            "people": people,
            "items": [options[LETTERS[k]] for k in range(len(people))],
            "query_person": re.search(r"At the end of [^,]+, ([A-Z][a-z]+) ", closing).group(1),
            "answer_object": answer_object,
            "answer_object_short": _short(answer_object),
            "initial_line": initial_line,
            "closing": closing,
            "options": options,
            "objects": len(people),
            "scenario": "ball",
            "verbatim": True,
        },
    )


def exemplars(k: int, seed: int):
    """Few-shot exemplars.  [0] is the verbatim first BBH CoT exemplar; the rest are
    freshly generated at the published depth (3 swaps, 3 objects)."""
    out = [_published_exemplar()]
    i = 0
    while len(out) < k:
        out.append(generate(3, seed * 1000 + i, objects=3, scenario="mixed"))
        i += 1
    return out[:k]


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------


def _render(inst: Instance) -> str:
    return "PROMPT:\n%s\n\nGOLD:\n%s\n" % (inst.prompt, format_cot(inst))


def _selftest() -> int:
    rng = random.Random(20260916)
    fails = []
    n_cases = 200
    for t in range(n_cases):
        depth = rng.choice(DEPTHS + [1, 4, 6, 7, 9, 10, 20])
        seed = rng.randrange(10**6)
        objects = rng.choice([3, 3, 3, 5, 5, 7])
        scenario = rng.choice(["mixed", "mixed"] + sorted(SCENARIOS))
        inst = generate(depth, seed, objects=objects, scenario=scenario)

        got = solve(inst)
        if got != inst.answer:
            fails.append("solve mismatch d=%d s=%d n=%d: %s != %s" % (depth, seed, objects, got, inst.answer))
        if not check(inst, format_cot(inst)):
            fails.append("check(gold) false d=%d s=%d" % (depth, seed))
        wrong = "(%s)" % LETTERS[(LETTERS.index(inst.answer[1]) + 1) % objects]
        if check(inst, "Answer: %s" % wrong):
            fails.append("check(wrong) true d=%d s=%d" % (depth, seed))
        again = generate(depth, seed, objects=objects, scenario=scenario)
        if (again.prompt, again.steps, again.answer) != (inst.prompt, inst.steps, inst.answer):
            fails.append("nondeterministic d=%d s=%d" % (depth, seed))
        if not (len(inst.steps) == len(inst.states) == depth):
            fails.append("length mismatch d=%d s=%d" % (depth, seed))
        spans = step_spans(inst)
        text = format_cot(inst)
        if [text[a:b] for a, b in spans] != inst.steps:
            fails.append("step_spans mismatch d=%d s=%d" % (depth, seed))
    print("generated instances: %d cases, %d failures" % (n_cases, len(fails)))

    # tolerant-parser probes
    ex = generate(4, 7, objects=3, scenario="ball")
    probes = [
        (format_cot(ex), True),
        ("%s So the answer is %s." % (ex.meta["closing"], ex.answer), True),
        ("blah\nAnswer: %s" % ex.answer[1], True),
        ("Answer: %s\nAnswer: %s" % ("(B)" if ex.answer != "(B)" else "(C)", ex.answer), True),
        ("Answer: %s" % ex.meta["answer_object"], True),
        ("", False),
        ("Answer: banana", False),
    ]
    for text, want in probes:
        if check(ex, text) != want:
            fails.append("parser probe failed: %r -> %r" % (text, not want))
    print("parser probes: %d" % len(probes))

    # differential check against the vendored, published BBH answers (750 items)
    bbh_total = bbh_bad = 0
    for n in (3, 5, 7):
        for inst in load_bbh(n):
            bbh_total += 1
            if solve(inst) != inst.meta["bbh_target"] or inst.answer != inst.meta["bbh_target"]:
                bbh_bad += 1
                if bbh_bad < 4:
                    fails.append("BBH mismatch n=%d idx=%d: solve=%s answer=%s target=%s"
                                 % (n, inst.meta["bbh_index"], solve(inst), inst.answer,
                                    inst.meta["bbh_target"]))
    print("vendored BBH items: %d checked, %d mismatches" % (bbh_total, bbh_bad))

    # exemplars
    exs = exemplars(3, 0)
    if len(exs) != 3:
        fails.append("exemplars(3,0) returned %d" % len(exs))
    if not exs[0].meta.get("verbatim"):
        fails.append("exemplars()[0] is not the published exemplar")
    with open(TRACE_PATH, encoding="utf-8") as fh:
        pub = fh.read().split("\n---\n", 1)[1].rstrip("\n")
    rebuilt = "Q: %s\nA: Let's think step by step.\n%s\n%s\n%s So the answer is %s." % (
        exs[0].prompt, exs[0].meta["initial_line"], "\n".join(exs[0].steps),
        exs[0].meta["closing"], exs[0].answer)
    if rebuilt != pub:
        fails.append("exemplar round-trip is NOT verbatim:\n--got--\n%s\n--want--\n%s" % (rebuilt, pub))
    if solve(exs[0]) != exs[0].answer:
        fails.append("solve() disagrees with the published exemplar answer")
    for e in exs:
        _render(e)
    print("exemplars: %d rendered, [0] verbatim round-trip OK" % len(exs))

    # knob validation
    for bad in (lambda: generate(3, 0, objects=9), lambda: generate(3, 0, objects=2),
                lambda: generate(3, 0, scenario="x"),
                lambda: generate(0, 0), lambda: generate(3, 0, nope=1)):
        try:
            bad()
            fails.append("bad knob accepted")
        except (ValueError, TypeError):
            pass

    # ---- redaction (SPEC AMENDMENT 4) ----
    if REDACTION_MEANINGFUL is not True:
        fails.append("REDACTION_MEANINGFUL should be True for cup_shuffling")
    rrng = random.Random(20260918)
    n_red = 20
    for t in range(n_red):
        depth = rrng.choice([1, 2, 3, 4, 5, 7, 8, 12, 16, 24])
        inst = generate(depth, rrng.randrange(10**6),
                        objects=rrng.choice([3, 5, 7]),
                        scenario=rrng.choice(["mixed"] + sorted(SCENARIOS)))
        tail = inst.prompt[inst.prompt.index("\nOptions:"):]
        swap_line = inst.prompt.split("\n")[1]
        matches = list(_SWAP_CLAUSE_RE.finditer(swap_line))
        clauses = [m.group(0) for m in matches]
        question = swap_line[matches[-1].end():].strip()  # "At the end of the game, Bob has the"
        tag = "d=%d s=%d" % (depth, inst.meta["seed"])
        for k in sorted({0, 1, depth // 2, depth}):
            red = redact_prompt(inst, k)
            if k == 0:
                if red != inst.prompt:
                    fails.append("redact k=0 changed the prompt (%s)" % tag)
                if PLACEHOLDER in red:
                    fails.append("redact k=0 inserted a placeholder (%s)" % tag)
                continue
            if red == inst.prompt:
                fails.append("redact k=%d is a no-op (%s)" % (k, tag))
            if PLACEHOLDER not in red:
                fails.append("redact k=%d has no placeholder (%s)" % (k, tag))
            if red.count(PLACEHOLDER) != k + 1:
                fails.append("redact k=%d: %d placeholders, want %d (%s)"
                             % (k, red.count(PLACEHOLDER), k + 1, tag))
            # question + options block survive untouched
            if not red.endswith(tail):
                fails.append("redact k=%d damaged the Options block (%s)" % (k, tag))
            if question not in red:
                fails.append("redact k=%d dropped the question (%s)" % (k, tag))
            head = red[:red.index("\nOptions:")]
            # the initial assignment is gone for every k >= 1
            for item in inst.meta["items"]:
                if item in head:
                    fails.append("redact k=%d leaks initial item %r (%s)" % (k, item, tag))
            # swaps 1..k gone, swaps k+1..depth intact and in order
            left = _SWAP_CLAUSE_RE.findall(red.split("\n")[1])
            if left != clauses[k:]:
                fails.append("redact k=%d kept the wrong swap clauses (%s)" % (k, tag))
            if k == depth and left:
                fails.append("redact k=depth still has %d swap clauses (%s)" % (len(left), tag))
        # the fully redacted prompt must not contain ANY operator or initial-state token
        full = redact_prompt(inst, depth)
        head = full[:full.index("\nOptions:")]
        for word in ("First,", "Then,", "Finally,"):
            if word in head:
                fails.append("redact k=depth leaks connective %r (%s)" % (word, tag))
        for name in inst.meta["people"]:
            # names may only survive in the cast sentence and the question
            if head.count(name) > (2 if name == inst.meta["query_person"] else 1):
                fails.append("redact k=depth leaks %r in a swap clause (%s)" % (name, tag))
        for bad_k in (-1, inst.depth + 1):
            try:
                redact_prompt(inst, bad_k)
                fails.append("redact accepted k=%d (%s)" % (bad_k, tag))
            except ValueError:
                pass
    # the verbatim published exemplar and vendored BBH items redact too
    for inst in [exemplars(1, 0)[0]] + [load_bbh(n)[0] for n in (3, 5, 7)]:
        red = redact_prompt(inst, inst.depth // 2)
        if red.count(PLACEHOLDER) != inst.depth // 2 + 1 or redact_prompt(inst, 0) != inst.prompt:
            fails.append("redaction failed on %s" % inst.meta.get("source"))
    print("redaction: %d instances x k in {0, 1, depth//2, depth}" % n_red)

    # ---- step corruption (SPEC AMENDMENT 6) ----
    crng = random.Random(20260918)
    n_cor = 20
    for t in range(n_cor):
        depth = crng.choice([1, 2, 3, 4, 5, 7, 8, 12, 16, 24])
        n = crng.choice([3, 5, 7])
        sc_knob = crng.choice(["mixed"] + sorted(SCENARIOS))
        gseed = crng.randrange(10**6)
        inst = generate(depth, gseed, objects=n, scenario=sc_knob)
        tag = "d=%d s=%d n=%d" % (depth, gseed, n)
        cseed = crng.randrange(10**6)
        for k in sorted({1, max(1, depth // 2), depth}):
            text, state = corrupt_step(inst, k, cseed)
            true_text, true_state = inst.steps[k - 1], inst.states[k - 1]
            if state == true_state:
                fails.append("corrupt k=%d: state unchanged (%s)" % (k, tag))
            if text == true_text:
                fails.append("corrupt k=%d: step text unchanged (%s)" % (k, tag))
            # structure: same step-line shape, same action clause, legal assignment
            m = _STEP_RE.match(text)
            if m is None:
                fails.append("corrupt k=%d: step line malformed %r (%s)" % (k, text, tag))
                continue
            if int(m.group(1)) != k:
                fails.append("corrupt k=%d: wrong step index (%s)" % (k, tag))
            prefix, names, shorts = _parse_step(text)
            true_prefix, true_names, true_shorts = _parse_step(true_text)
            if prefix != true_prefix:
                fails.append("corrupt k=%d: action clause changed (%s)" % (k, tag))
            if names != true_names or names != inst.meta["people"][:n]:
                fails.append("corrupt k=%d: people list changed (%s)" % (k, tag))
            if sorted(shorts) != sorted(true_shorts):
                fails.append("corrupt k=%d: not a permutation of the true objects (%s)" % (k, tag))
            if len(set(shorts)) != n:
                fails.append("corrupt k=%d: an object is held twice (%s)" % (k, tag))
            if state != " ".join(shorts):
                fails.append("corrupt k=%d: state disagrees with the step line (%s)" % (k, tag))
            if state.count(" ") != true_state.count(" "):
                fails.append("corrupt k=%d: state shape changed (%s)" % (k, tag))
            # exactly one transposition away from the truth (minimal edit)
            diff = [p for p in range(n) if shorts[p] != true_shorts[p]]
            if len(diff) != 2:
                fails.append("corrupt k=%d: %d positions differ, want 2 (%s)" % (k, len(diff), tag))
            # determinism, and sensitivity to the seed arguments
            if corrupt_step(inst, k, cseed) != (text, state):
                fails.append("corrupt k=%d: nondeterministic (%s)" % (k, tag))
            again = generate(depth, gseed, objects=n, scenario=sc_knob)
            if corrupt_step(again, k, cseed) != (text, state):
                fails.append("corrupt k=%d: not a pure function of (inst, k, seed) (%s)" % (k, tag))
        for bad_k in (0, -1, inst.depth + 1):
            try:
                corrupt_step(inst, bad_k, 0)
                fails.append("corrupt accepted k=%d (%s)" % (bad_k, tag))
            except ValueError:
                pass
    # different seeds must be able to produce different corruptions (n=5: 10 transpositions)
    inst = generate(6, 11, objects=5, scenario="ball")
    if len({corrupt_step(inst, 3, s)[1] for s in range(20)}) < 2:
        fails.append("corrupt: seed has no effect")
    # the verbatim published exemplar and vendored BBH items corrupt too
    for inst in [exemplars(1, 0)[0]] + [load_bbh(n)[0] for n in (3, 5, 7)]:
        text, state = corrupt_step(inst, 1, 0)
        if state == inst.states[0] or text == inst.steps[0] or _STEP_RE.match(text) is None:
            fails.append("corruption failed on %s" % inst.meta.get("source"))
    print("corruption: %d instances x k in {1, depth//2, depth}" % n_cor)


    if fails:
        print("\nFAILURES (%d):" % len(fails))
        for f in fails[:20]:
            print("  -", f)
        print("SELFTEST FAILED")
        return 1
    print("SELFTEST PASSED")
    return 0


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, 100 + i, objects=3, scenario="mixed")
            print("=" * 78)
            print("depth=%d  objects=%d  scenario=%s  seed=%d  answer=%s (%s)"
                  % (inst.depth, inst.meta["objects"], inst.meta["scenario"], inst.meta["seed"],
                     inst.answer, inst.meta["answer_object"]))
            print("=" * 78)
            print(_render(inst))


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
