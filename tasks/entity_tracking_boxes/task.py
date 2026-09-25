"""entity_tracking_boxes -- the "boxes" entity-tracking task.

Primary source
--------------
Najoung Kim and Sebastian Schuster, "Entity Tracking in Language Models",
ACL 2023, pp. 3835-3855.  https://aclanthology.org/2023.acl-long.213/
Repo: https://github.com/sebschu/entity-tracking-lms @ 8400de051ef4ad9483cc37dee3017b37a48dafd7

A world of N boxes holds objects drawn from a 100-word BNC-frequency vocabulary.
The prompt states the initial contents of every box, then applies a sequence of
`move` / `remove` / `put` operations.  The model must report the contents of one
queried box after all operations.

The paper has NO chain-of-thought / scratchpad variant (desk.json:
trace_available=false), so the gold trace format here is chosen, not published.
See README.md "Format decision".

Pure python stdlib.  Object vocabulary is read from vendor/ at import time.
"""

import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# vendored data
# --------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_VOCAB_FILES = {
    "bnc": os.path.join(
        _HERE, "vendor", "entity-tracking-lms", "data", "objects_with_bnc_frequency.csv"
    ),
    "disjoint": os.path.join(
        _HERE, "vendor", "entity-tracking-lms", "data", "objects_not_in_bnc.csv"
    ),
}
_VOCAB_CACHE: dict[str, list[str]] = {}


def _load_vocab(name: str) -> list[str]:
    """Object names, in the file's own (frequency-sorted) order."""
    if name not in _VOCAB_FILES:
        raise ValueError(f"unknown vocab {name!r}; expected one of {sorted(_VOCAB_FILES)}")
    if name not in _VOCAB_CACHE:
        path = _VOCAB_FILES[name]
        with open(path, encoding="utf-8-sig", newline="") as f:
            _VOCAB_CACHE[name] = [row["object_name"] for row in csv.DictReader(f)]
    return _VOCAB_CACHE[name]


# --------------------------------------------------------------------------
# published surface strings (verbatim templates from generate_boxes_data.py)
# --------------------------------------------------------------------------

_OPERATIONS_DICT = {
    "move": "Move {content} from Box {box1} to Box {box2}.",
    "remove": "Remove {content} from Box {box1}.",
    "put": "Put {content} into Box {box1}.",
}
_OPERATIONS = ("move", "remove", "put")


def _content_phrase(items) -> str:
    """'the apple and the key' -- the paper's conjunction rendering."""
    return " and ".join(f"the {c}" for c in sorted(items))


def _box_phrase(box: int, items) -> str:
    """'Box 3 contains the apple and the key' / 'Box 3 contains nothing'.

    Matches WorldState._describe_box with zero_shot=True, which is the form used
    in the paper's few-shot prompt file (prompt_incontext.txt).
    """
    if not items:
        return f"Box {box} contains nothing"
    return f"Box {box} contains {_content_phrase(items)}"


def _canonical(items) -> str:
    """The exact-match answer string."""
    return ", ".join(sorted(items)) if items else "nothing"


def _state_string(boxes: list[set]) -> str:
    """Compact full world state: '0:boat,shoe|1:-|2:cake'."""
    return "|".join(
        f"{i}:" + (",".join(sorted(b)) if b else "-") for i, b in enumerate(boxes)
    )


# --------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------

ANSWER_FORMAT = (
    'the contents of the queried box: object names in alphabetical order separated '
    'by ", " (for example "apple, key"), or the single word "nothing" if the box is empty'
)

# Redaction is meaningful here: the initial box contents and the already-applied
# operations are exactly the state a model would otherwise re-derive from the prompt.
REDACTION_MEANINGFUL = True

# Paper default scenario length is 12 operations; difficulty in the paper (Fig. 2)
# is read out as the number of ops that touched the queried box, which grows with
# total ops.  Sweep spans trivial (2) through well past the paper's default.
DEPTHS = [2, 4, 8, 12, 18, 26, 36]

KNOBS = {
    "num_boxes": (7, "number of boxes in the world (paper default 7)"),
    "max_items_per_box": (3, "hard cap on objects per box (paper default 3)"),
    "expected_items_per_box": (
        2,
        "mean of the Poisson used for initial per-box occupancy (paper default 2)",
    ),
    "vocab": (
        "bnc",
        "'bnc' = objects_with_bnc_frequency.csv (100 words, paper default); "
        "'disjoint' = objects_not_in_bnc.csv (the paper's held-out vocabulary split)",
    ),
    "query_policy": (
        "most_changed",
        "which box is queried: 'most_changed' (a box touched by the most operations, "
        "so depth actually bites), 'uniform' (any box, as the paper's test set does), or "
        "'most_changed_nonempty' (as most_changed but restricted to boxes that end up "
        "non-empty, which removes the guessable 'nothing' majority class)",
    ),
}


@dataclass
class Instance:
    prompt: str
    steps: list
    states: list
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's Poisson sampler -- stdlib stand-in for numpy.random.poisson."""
    limit = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def _nonempty_subset(rng: random.Random, items: set) -> list:
    """Coin-flip subset, resampled until non-empty (as in the vendored generator)."""
    ordered = sorted(items)
    while True:
        out = [c for c in ordered if rng.randint(0, 1) == 0]
        if out:
            return out


def generate(depth: int, seed: int, **knobs) -> Instance:
    if depth < 0:
        raise ValueError("depth must be >= 0")
    cfg = {k: v[0] for k, v in KNOBS.items()}
    unknown = set(knobs) - set(cfg)
    if unknown:
        raise ValueError(f"unknown knobs: {sorted(unknown)}")
    cfg.update(knobs)

    num_boxes = int(cfg["num_boxes"])
    cap = int(cfg["max_items_per_box"])
    exp_items = int(cfg["expected_items_per_box"])
    vocab = _load_vocab(str(cfg["vocab"]))
    policy = str(cfg["query_policy"])
    if num_boxes < 1:
        raise ValueError("num_boxes must be >= 1")
    if cap < 1:
        raise ValueError("max_items_per_box must be >= 1")

    rng = random.Random(
        "entity_tracking_boxes|%d|%d|%d|%d|%d|%s|%s"
        % (depth, seed, num_boxes, cap, exp_items, cfg["vocab"], policy)
    )

    # --- initial world state -------------------------------------------------
    boxes = [set() for _ in range(num_boxes)]
    placed: set = set()
    while True:
        counts = [min(_poisson(rng, exp_items), cap) for _ in range(num_boxes)]
        if sum(counts) <= len(vocab):
            break
    for i, n in enumerate(counts):
        free = [o for o in vocab if o not in placed]
        chosen = rng.sample(free, n)
        boxes[i].update(chosen)
        placed.update(chosen)

    initial_boxes = [set(b) for b in boxes]

    # --- operations ----------------------------------------------------------
    ops = [o for o in _OPERATIONS if not (o == "move" and num_boxes < 2)]
    op_records = []       # (kind, box1, box2|None, sorted contents)
    op_sentences = []
    states = []
    changed_counts = [0] * num_boxes

    for _ in range(depth):
        guard = 0
        while True:
            guard += 1
            if guard > 10000:
                raise RuntimeError("could not sample a legal operation")
            kind = rng.choice(ops)
            box1 = rng.randrange(num_boxes)
            if kind == "move":
                if not boxes[box1]:
                    continue
                others = [b for b in range(num_boxes) if b != box1]
                box2 = rng.choice(others)
                content = _nonempty_subset(rng, boxes[box1])
                if len(boxes[box2]) + len(content) > cap:
                    continue
                boxes[box1].difference_update(content)
                boxes[box2].update(content)
                op_records.append(("move", box1, box2, sorted(content)))
                changed_counts[box1] += 1
                changed_counts[box2] += 1
                break
            if kind == "remove":
                if not boxes[box1]:
                    continue
                content = _nonempty_subset(rng, boxes[box1])
                boxes[box1].difference_update(content)
                placed.difference_update(content)
                op_records.append(("remove", box1, None, sorted(content)))
                changed_counts[box1] += 1
                break
            # put
            n = _poisson(rng, max(1, exp_items // 2))
            if n < 1 or len(boxes[box1]) + n > cap:
                continue
            free = [o for o in vocab if o not in placed]
            if len(free) < n:
                continue
            content = sorted(rng.sample(free, n))
            boxes[box1].update(content)
            placed.update(content)
            op_records.append(("put", box1, None, content))
            changed_counts[box1] += 1
            break

        kind, box1, box2, content = op_records[-1]
        op_sentences.append(
            _OPERATIONS_DICT[kind].format(
                content=_content_phrase(content), box1=box1, box2=box2
            )
        )
        states.append(_state_string(boxes))

    # --- queried box ---------------------------------------------------------
    if policy == "uniform":
        candidates = list(range(num_boxes))
    elif policy in ("most_changed", "most_changed_nonempty"):
        pool = list(range(num_boxes))
        if policy == "most_changed_nonempty":
            nonempty = [i for i in pool if boxes[i]]
            if nonempty:
                pool = nonempty
        best = max(changed_counts[i] for i in pool)
        candidates = [i for i in pool if changed_counts[i] == best]
    else:
        raise ValueError(f"unknown query_policy {policy!r}")
    query = rng.choice(candidates)

    # --- rendering -----------------------------------------------------------
    description = (
        ", ".join(_box_phrase(i, initial_boxes[i]) for i in range(num_boxes)) + "."
    )
    if op_sentences:
        description += " " + " ".join(op_sentences)
    prompt = description + f"\nQuestion: What does Box {query} contain?"

    # gold trace: one line per operation, restating only the affected box(es)
    replay = [set(b) for b in initial_boxes]
    steps = []
    for idx, (kind, box1, box2, content) in enumerate(op_records, start=1):
        if kind == "move":
            replay[box1].difference_update(content)
            replay[box2].update(content)
            touched = [box1, box2]
        elif kind == "remove":
            replay[box1].difference_update(content)
            touched = [box1]
        else:
            replay[box1].update(content)
            touched = [box1]
        body = ", ".join(_box_phrase(b, replay[b]) for b in touched)
        steps.append(f"After op {idx}: {body}.")

    answer = _canonical(boxes[query])
    closing = _box_phrase(query, boxes[query]) + "."

    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=answer,
        depth=depth,
        meta={
            "seed": seed,
            "query_box": query,
            "num_boxes": num_boxes,
            "max_items_per_box": cap,
            "expected_items_per_box": exp_items,
            "vocab": cfg["vocab"],
            "query_policy": policy,
            "numops_queried_box": changed_counts[query],
            "initial_state": _state_string(initial_boxes),
            "closing": closing,
            "operations": [
                {"op": k, "box1": b1, "box2": b2, "contents": c}
                for (k, b1, b2, c) in op_records
            ],
        },
    )


# --------------------------------------------------------------------------
# prompt redaction (AMENDMENT 4)
# --------------------------------------------------------------------------

REDACTION_PLACEHOLDER = "[…]"


def _split_description(prompt: str) -> tuple:
    """(initial-contents sentence, [operation sentences]) -- both with trailing '.'.

    Splits the prompt's description exactly the way ``solve()`` does.  Neither the
    box-contents clause nor an operation sentence can contain a '.', so
    ``" ".join(parts)`` reconstructs the description byte-for-byte.
    """
    description = prompt.split("\nQuestion:")[0]
    parts = [p.strip() + "." for p in description.split(".") if p.strip()]
    if not parts:
        raise ValueError("prompt has no description")
    return parts[0], parts[1:]


def redact_prompt(inst: Instance, k: int) -> str:
    """The prompt with the first ``k`` operation sentences replaced by ``[…]``.

    Removed: one placeholder per redacted operation sentence.  Kept: the initial
    box-contents description, operations ``k+1..depth`` verbatim, and the question
    line.  ``k == 0`` is the identity; ``k == inst.depth`` leaves the initial
    description, ``depth`` placeholders and the question.

    The initial description is deliberately kept even though it is nominally
    "initial state": the published trace format restates only the boxes an
    operation touched, so a box no operation has reached by step ``k`` is described
    nowhere in the trace prefix.  Dropping the description would make the answer
    under-determined rather than merely un-recomputable.  Removing the operators
    for steps 1..k is already sufficient to stop a model re-deriving the state
    after step ``k`` from the prompt alone.
    """
    if k < 0 or k > inst.depth:
        raise ValueError(f"k must be in [0, {inst.depth}], got {k}")
    if k == 0:
        return inst.prompt
    initial, ops = _split_description(inst.prompt)
    if len(ops) != inst.depth:
        raise ValueError(f"prompt has {len(ops)} operations, expected {inst.depth}")
    head = [initial] + [REDACTION_PLACEHOLDER] * k + ops[k:]
    tail = inst.prompt[len(inst.prompt.split("\nQuestion:")[0]):]
    return " ".join(head) + tail


# --------------------------------------------------------------------------
# reference solver -- parses the prompt and replays it, independent of generate()
# --------------------------------------------------------------------------

_RE_QUERY = re.compile(r"What does Box (\d+) contain")
_RE_INIT = re.compile(r"^Box (\d+) contains (.+)$")
_RE_MOVE = re.compile(r"^Move (.+) from Box (\d+) to Box (\d+)$")
_RE_REMOVE = re.compile(r"^Remove (.+) from Box (\d+)$")
_RE_PUT = re.compile(r"^Put (.+) into Box (\d+)$")


def _parse_items(phrase: str) -> list:
    phrase = phrase.strip()
    if phrase == "nothing":
        return []
    return [p.strip()[4:] if p.strip().startswith("the ") else p.strip()
            for p in phrase.split(" and ")]


def solve(inst: Instance) -> str:
    text = inst.prompt
    qm = _RE_QUERY.search(text)
    if qm is None:
        raise ValueError("no query found in prompt")
    query = int(qm.group(1))
    description = text.split("\nQuestion:")[0]

    sentences = [s.strip() for s in description.split(".") if s.strip()]
    world: dict = {}
    for part in sentences[0].split(", "):
        m = _RE_INIT.match(part.strip())
        if m is None:
            raise ValueError(f"unparsable initial state clause: {part!r}")
        world[int(m.group(1))] = set(_parse_items(m.group(2)))

    for sent in sentences[1:]:
        m = _RE_MOVE.match(sent)
        if m:
            items, src, dst = _parse_items(m.group(1)), int(m.group(2)), int(m.group(3))
            for it in items:
                world[src].discard(it)
                world[dst].add(it)
            continue
        m = _RE_REMOVE.match(sent)
        if m:
            for it in _parse_items(m.group(1)):
                world[int(m.group(2))].discard(it)
            continue
        m = _RE_PUT.match(sent)
        if m:
            for it in _parse_items(m.group(1)):
                world[int(m.group(2))].add(it)
            continue
        raise ValueError(f"unparsable operation: {sent!r}")

    return _canonical(world[query])


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

_EMPTY_WORDS = {"nothing", "empty", "isempty", "none", ""}


def _normalize_answer(raw: str) -> str:
    s = raw.strip().lower().rstrip(".").strip()
    s = re.sub(r"^box\s+\d+\s+(contains|has|is)\b", "", s).strip()
    s = re.sub(r"^\s*(contains|has)\b", "", s).strip()
    s = s.strip('"').strip("'").strip()
    parts = [p.strip() for p in re.split(r",\s*and\s+|\s+and\s+|,|;", s)]
    items = []
    for p in parts:
        p = p.strip().strip(".").strip()
        if p.startswith("the "):
            p = p[4:].strip()
        elif p.startswith("a ") or p.startswith("an "):
            p = p.split(" ", 1)[1].strip()
        if p in _EMPTY_WORDS or p == "is empty":
            continue
        if p:
            items.append(p)
    if not items:
        return "nothing"
    return ", ".join(sorted(set(items)))


def check(inst: Instance, completion: str) -> bool:
    lines = completion.splitlines()
    picked = None
    for line in reversed(lines):
        if "answer:" in line.lower():
            picked = line[line.lower().rindex("answer:") + len("answer:"):]
            break
    if picked is None:
        for line in reversed(lines):
            if line.strip():
                picked = line
                break
    if picked is None:
        return False
    return _normalize_answer(picked) == _normalize_answer(inst.answer)


# --------------------------------------------------------------------------
# gold completion
# --------------------------------------------------------------------------


def format_cot(inst: Instance) -> str:
    body = "\n".join(inst.steps)
    closing = inst.meta.get("closing") or f"Box ? contains {inst.answer}."
    if body:
        return f"{body}\n{closing}\nAnswer: {inst.answer}"
    return f"{closing}\nAnswer: {inst.answer}"


def step_spans(inst: Instance) -> list:
    text = format_cot(inst)
    spans = []
    cursor = 0
    for step in inst.steps:
        start = text.index(step, cursor)
        end = start + len(step)
        spans.append((start, end))
        cursor = end
    return spans


# --------------------------------------------------------------------------
# few-shot exemplars
# --------------------------------------------------------------------------

_EXEMPLAR_DEPTHS = [6, 4, 4, 5, 3, 6, 4, 5]


def exemplars(k: int, seed: int) -> list:
    """Generated exemplars.

    No published CoT exemplar exists for this task (desk.json trace_available=false);
    the paper's few-shot prompt (prompt_incontext.txt, inside the authors'
    password-protected zip) is a direct-answer exemplar with 6 operations and no
    reasoning trace, and the authors ask that the uncompressed files not be
    redistributed.  exemplars(k, seed)[0] therefore mirrors that exemplar's shape
    (6 operations, paper-default world) rather than reproducing it verbatim.
    """
    out = []
    for i in range(k):
        depth = _EXEMPLAR_DEPTHS[i % len(_EXEMPLAR_DEPTHS)]
        # exemplar 0 is forced non-empty so the first demonstration shows the
        # full answer format; the rest use the default policy, so an empty-box
        # exemplar can still appear and the shot set is not biased against it.
        knobs = {"query_policy": "most_changed_nonempty"} if i == 0 else {}
        out.append(generate(depth, seed * 1000 + i + 1, **knobs))
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _selftest(n: int = 200) -> int:
    rng = random.Random(20260916)
    failures = []
    for trial in range(n):
        depth = rng.choice(DEPTHS + [rng.randrange(0, 40)])
        seed = rng.randrange(10**6)
        inst = generate(depth, seed)

        got = solve(inst)
        if got != inst.answer:
            failures.append(f"[{trial}] solve mismatch d={depth} s={seed}: {got!r} != {inst.answer!r}")
        if not check(inst, format_cot(inst)):
            failures.append(f"[{trial}] check(gold) False d={depth} s={seed}")
        wrong = "nothing" if inst.answer != "nothing" else "book"
        if check(inst, f"Answer: {wrong}"):
            failures.append(f"[{trial}] check(wrong) True d={depth} s={seed}")
        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer) != (
            inst.prompt, inst.steps, inst.states, inst.answer
        ):
            failures.append(f"[{trial}] nondeterministic d={depth} s={seed}")
        if not (len(inst.steps) == len(inst.states) == depth):
            failures.append(
                f"[{trial}] length mismatch d={depth} steps={len(inst.steps)} states={len(inst.states)}"
            )
        spans = step_spans(inst)
        text = format_cot(inst)
        if len(spans) != depth or any(text[a:b] != s for (a, b), s in zip(spans, inst.steps)):
            failures.append(f"[{trial}] bad step_spans d={depth} s={seed}")
        if not text.rstrip().splitlines()[-1].startswith("Answer: "):
            failures.append(f"[{trial}] gold does not end with Answer: line")

    # --- redaction (AMENDMENT 4) -------------------------------------------
    if REDACTION_MEANINGFUL is not True:
        failures.append("REDACTION_MEANINGFUL should be True for this task")
    for trial in range(20):
        depth = rng.choice(DEPTHS)
        seed = rng.randrange(10**6)
        inst = generate(depth, seed)
        tag = f"[redact {trial}] d={depth} s={seed}"
        question = "\nQuestion:" + inst.prompt.split("\nQuestion:")[1]
        initial, op_sentences = _split_description(inst.prompt)
        if " ".join([initial] + op_sentences) + question != inst.prompt:
            failures.append(f"{tag} description does not round-trip")

        if redact_prompt(inst, 0) != inst.prompt:
            failures.append(f"{tag} redact_prompt(inst, 0) != inst.prompt")
        for k in sorted({1, depth // 2, depth}):
            if not 0 < k <= depth:
                continue
            red = redact_prompt(inst, k)
            if red == inst.prompt:
                failures.append(f"{tag} k={k} redaction did not change the prompt")
            if REDACTION_PLACEHOLDER not in red:
                failures.append(f"{tag} k={k} redaction has no placeholder")
            # one placeholder per redacted operation; the initial description stays
            if red.count(REDACTION_PLACEHOLDER) != k:
                failures.append(f"{tag} k={k} expected {k} placeholders, "
                                f"got {red.count(REDACTION_PLACEHOLDER)}")
            if not red.startswith(initial + " "):
                failures.append(f"{tag} k={k} initial description not preserved")
            if not red.endswith(question):
                failures.append(f"{tag} k={k} question line not preserved")
            # operations k+1..depth survive verbatim, in order
            cursor = 0
            for sent in op_sentences[k:]:
                idx = red.find(sent, cursor)
                if idx < 0:
                    failures.append(f"{tag} k={k} lost surviving operation {sent!r}")
                    break
                cursor = idx + len(sent)
            # redacted operations are gone (a later op may legitimately repeat an
            # earlier sentence, so only check the ones that do not recur)
            for sent in op_sentences[:k]:
                if sent not in op_sentences[k:] and sent in red:
                    failures.append(f"{tag} k={k} kept redacted operation {sent!r}")

        # k == depth: the initial description survives, every operator is gone
        full = redact_prompt(inst, depth)
        head = full.split("\nQuestion:")[0]
        expected_head = " ".join([initial] + [REDACTION_PLACEHOLDER] * depth)
        if head != expected_head:
            failures.append(f"{tag} k=depth head is not description + placeholders: {head!r}")
        # object names from the initial description may (must) survive; names that
        # only ever entered the world through an operation must not.
        initial_objects = set()
        for spec in inst.meta["initial_state"].split("|"):
            body = spec.split(":", 1)[1]
            if body != "-":
                initial_objects.update(body.split(","))
        op_objects = set()
        for op in inst.meta["operations"]:
            op_objects.update(op["contents"])
        leaked = sorted(
            o for o in op_objects - initial_objects
            if re.search(r"\b%s\b" % re.escape(o), head)
        )
        if leaked:
            failures.append(f"{tag} k=depth leaked operation-only object tokens {leaked}")
        for token in ("Move", "Remove", "Put", " from ", " into "):
            if token in head:
                failures.append(f"{tag} k=depth leaked operator token {token!r}")

    for bad in (-1, generate(4, 1).depth + 1):
        try:
            redact_prompt(generate(4, 1), bad)
        except ValueError:
            pass
        else:
            failures.append(f"redact_prompt accepted out-of-range k={bad}")

    # tolerant-parser spot checks on the last instance
    inst = generate(12, 7)
    variants = [
        f"Answer: {inst.answer}",
        f"blah blah\nAnswer:  {inst.answer} .",
        f"Box {inst.meta['query_box']} contains " + (
            "nothing." if inst.answer == "nothing"
            else " and ".join(f"the {c}" for c in reversed(inst.answer.split(", "))) + "."
        ),
    ]
    for v in variants:
        if not check(inst, v):
            failures.append(f"tolerant parse failed on {v!r}")

    try:
        ex = exemplars(3, 0)
        for e in ex:
            format_cot(e)
            if solve(e) != e.answer:
                failures.append("exemplar solve mismatch")
        if len(ex) != 3:
            failures.append("exemplars(3, 0) wrong length")
    except Exception as exc:  # pragma: no cover
        failures.append(f"exemplars raised {exc!r}")

    # knob sanity
    for kn in ({"num_boxes": 3}, {"max_items_per_box": 1}, {"vocab": "disjoint"},
               {"query_policy": "uniform"}, {"query_policy": "most_changed_nonempty"},
               {"expected_items_per_box": 1}):
        for sd in range(5):
            inst = generate(10, sd, **kn)
            if solve(inst) != inst.answer:
                failures.append(f"knob {kn} solve mismatch seed={sd}")
    for sd in range(20):
        inst = generate(14, sd, query_policy="most_changed_nonempty")
        if inst.answer == "nothing":
            failures.append(f"most_changed_nonempty produced empty answer seed={sd}")

    if failures:
        print(f"SELFTEST FAILED: {len(failures)} problem(s)")
        for f in failures[:20]:
            print("  " + f)
        return 1
    print(f"ran {n} random (depth, seed) pairs")
    print("solve()==answer, check(gold)=True, check(wrong)=False, determinism, "
          "len(steps)==len(states)==depth, step_spans, exemplars, knobs: all OK")
    print("redact_prompt: identity at k=0, placeholders and surviving ops at "
          "k in {1, depth//2, depth}, initial description kept and every operator "
          "gone at k=depth: OK")
    print("SELFTEST PASSED")
    return 0


def _demo() -> None:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, 1000 + i)
            print("=" * 78)
            print(f"depth={inst.depth}  seed={inst.meta['seed']}  "
                  f"query_box={inst.meta['query_box']}  "
                  f"ops_touching_queried_box={inst.meta['numops_queried_box']}")
            print("-" * 78)
            print("PROMPT:")
            print(inst.prompt)
            print("-" * 78)
            print("GOLD COT:")
            print(format_cot(inst))
            print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.demo:
        _demo()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
