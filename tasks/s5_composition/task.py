"""
s5_composition -- the word problem over S_n (symmetric group), in the published
permutation-composition format of belindal/state-tracking.

Published format (see published_trace.txt, and vendor/state-tracking/permutation_task.py
at commit fc63e2db262265f42e9d2ac5c05888284f843b4c, MIT):

    * A "story" is a space-separated sequence of actions; each action is a permutation of
      1..n written in one-line notation as an n-digit string, e.g. "53124".
    * The state is itself an n-digit string, starting at the identity "12345".
    * Composition, verbatim from PermutationState.apply_action:
          new_perm = tuple(self.permutation[action[j]-1] for j in range(n))
      i.e. new_state[i] = state[action[i]], 1-indexed.
    * The state is NEVER written inline in the published format: the model reads only the
      action sequence. States are recorded separately by the generator (state_seq in the
      emitted JSON) for supervision/probing.

Theory: Barrington 1989 -- the word problem for any non-solvable finite group is
NC1-complete under projections; S_n (n>=5) and A_n (n>=5) are non-solvable, S_3/S_4 and the
cyclic groups Z_n are solvable. Liu, Ash, Goel, Krishnamurthy & Zhang, "Transformers Learn
Shortcuts to Automata" (arXiv:2210.10749) is the primary source for this task as a
transformer benchmark; it publishes no textual trace, which is why the trace and the
notation come from the follow-up "(How) Do Language Models Track State?" (arXiv:2503.02854).

AMENDMENT 5 adds the `format` knob. `format="published"` (default) is the above.
`format="ergonomic"` is an SFT-free variant: the same instances, answers and depth
semantics, but a plain-language prompt and a step line that writes the permutation,
the position lookups and the resulting arrangement out in full, e.g.

    step 3: apply 51243 to 21543 -> take positions 5,1,2,4,3 of 21543 -> 3 2 1 4 5 -> 32145

Pure python stdlib. See README.md for the format decision and caveats.
"""

from __future__ import annotations

import itertools
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
    prompt: str          # problem statement only, published notation, no scratchpad instruction
    steps: list[str]     # gold trace, one element per composed permutation
    states: list[str]    # running product after each step (len == len(steps))
    answer: str          # final permutation as an n-digit string
    depth: int           # number of permutations composed
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a permutation of the digits 1-5 written as a single 5-digit string, e.g. 53124"

# depth = number of permutations composed. See README "Depth semantics".
DEPTHS = [2, 4, 8, 12, 16, 24, 32]

KNOBS = {
    "group": (
        "S5",
        "group the actions are drawn from: 'S5' (default, non-solvable, NC1-complete), "
        "'S3' (solvable contrast), 'A5' (non-solvable, alternating), or 'Zn' (cyclic "
        "rotations of 1..n, abelian/solvable contrast). Case-insensitive.",
    ),
    "n": (
        None,
        "number of items, 2..9 (digits must stay single characters). None = the group's "
        "natural n (S5/A5 -> 5, S3 -> 3, Zn -> 5).",
    ),
    "format": (
        "published",
        "trace/prompt format: 'published' (default, belindal/state-tracking notation, "
        "taught by fine-tuning) or 'ergonomic' (AMENDMENT 5: plain-language prompt, "
        "state written out in full at every step, meant to be followed from 3 exemplars "
        "by a prompted instruct model). The instance distribution, answer, depth "
        "semantics, solve() and check() are identical either way.",
    ),
}

FORMATS = ("published", "ergonomic")


def _resolve_format(fmt: str) -> str:
    f = str(fmt).strip().lower()
    if f not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; use 'published' or 'ergonomic'")
    return f


# --------------------------------------------------------------------------------------
# group machinery
# --------------------------------------------------------------------------------------


def _resolve_group(group: str, n):
    """-> (canonical group name, n). Raises ValueError on bad knobs."""
    g = str(group).strip().upper()
    if re.fullmatch(r"[SA][2-9]", g):
        natural = int(g[1])
        if n is not None and int(n) != natural:
            raise ValueError(f"group {g} fixes n={natural}; pass group='{g[0]}{n}' instead")
        n = natural
    elif g in ("Z", "ZN", "CYCLIC"):
        g = "Zn"
        n = 5 if n is None else int(n)
    elif re.fullmatch(r"Z[2-9]", g):
        n = int(g[1])
        g = "Zn"
    else:
        raise ValueError(f"unknown group {group!r}; use S5, S3, A5, S<n>, A<n> or Zn")
    if not 2 <= n <= 9:
        raise ValueError("n must be in 2..9 so each item is a single digit")
    return g, n


def _parity(perm: tuple[int, ...]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return inv % 2


def _group_elements(group: str, n: int) -> list[tuple[int, ...]]:
    """Action vocabulary, in a deterministic order. 1-indexed one-line notation."""
    if group.startswith("S"):
        return list(itertools.permutations(range(1, n + 1)))
    if group.startswith("A"):
        return [p for p in itertools.permutations(range(1, n + 1)) if _parity(p) == 0]
    if group == "Zn":
        base = list(range(1, n + 1))
        return [tuple(base[k:] + base[:k]) for k in range(n)]
    raise ValueError(group)


def _compose(state: tuple[int, ...], action: tuple[int, ...]) -> tuple[int, ...]:
    """Published composition: new_state[i] = state[action[i]] (1-indexed)."""
    return tuple(state[action[i] - 1] for i in range(len(state)))


def _s(perm: tuple[int, ...]) -> str:
    return "".join(str(x) for x in perm)


def _p(text: str) -> tuple[int, ...]:
    return tuple(int(c) for c in text)


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------


def _worked_example(n: int) -> str:
    """One-line worked example of the convention, in the ergonomic step style, for this n."""
    ident = tuple(range(1, n + 1))
    action = (2, 1) + ident[2:]          # swap the first two positions
    arrangement = tuple(reversed(ident))  # something that is not the identity
    new = _compose(arrangement, action)
    return (
        f"For example, applying {_s(action)} to {_s(arrangement)} means taking positions "
        f"{','.join(str(x) for x in action)} of {_s(arrangement)}, which is "
        f"{' '.join(str(x) for x in new)}, so the new arrangement is {_s(new)}."
    )


def _render_prompt(start: str, actions: list[str], fmt: str = "published") -> str:
    n = len(start)
    if fmt == "ergonomic":
        return (
            f"Start with the arrangement {start}.\n"
            f"Apply these {len(actions)} permutations to it, one at a time, in the order "
            f"given:\n"
            f"{' '.join(actions)}\n"
            f"Each permutation is a list of positions: to apply one, read off the digits of "
            f"the current arrangement at those positions, in that order, and that is the new "
            f"arrangement. {_worked_example(n)}\n"
            f"What is the final arrangement?"
        )
    return (
        f"Start: {start}\n"
        f"Apply the following {len(actions)} permutations to the starting arrangement, "
        f"in order:\n"
        f"{' '.join(actions)}\n"
        f"Each permutation is written in one-line notation over the positions 1-{n}: to apply "
        f"a permutation a to the current arrangement s, the new arrangement's i-th digit is "
        f"the digit that s has in position a_i.\n"
        f"What is the final arrangement?"
    )


def _render_step(k: int, action: str, prev: str, state: str, fmt: str = "published") -> str:
    if fmt == "ergonomic":
        return (
            f"step {k}: apply {action} to {prev} -> take positions "
            f"{','.join(action)} of {prev} -> {' '.join(state)} -> {state}"
        )
    return f"step {k}: {action} -> {state}"


def _build(start: str, actions: list[str], meta: dict, fmt: str = "published") -> Instance:
    """Shared constructor: runs the published composition forward over `actions`.

    `fmt` changes only the rendering of `prompt` and `steps`; `states`, `answer` and
    `depth` are format-independent by construction.
    """
    fmt = _resolve_format(fmt)
    state = _p(start)
    steps, states = [], []
    for k, a in enumerate(actions, start=1):
        prev = _s(state)
        state = _compose(state, _p(a))
        states.append(_s(state))
        steps.append(_render_step(k, a, prev, _s(state), fmt))
    answer = _s(state)
    meta = dict(meta)
    meta.update(
        {
            "start": start,
            "actions": list(actions),
            "n": len(start),
            "format": fmt,
            "answer_format": (
                f"a permutation of the digits 1-{len(start)} written as a single "
                f"{len(start)}-digit string, e.g. {actions[0] if actions else start}"
            ),
        }
    )
    return Instance(
        prompt=_render_prompt(start, actions, fmt),
        steps=steps,
        states=states,
        answer=answer,
        depth=len(actions),
        meta=meta,
    )


# --------------------------------------------------------------------------------------
# interface
# --------------------------------------------------------------------------------------


def generate(
    depth: int, seed: int, group: str = "S5", n=None, format: str = "published", **_ignored
) -> Instance:
    """Pure function of (depth, seed, knobs).

    The instance drawn is a function of (depth, seed, group, n) ONLY: `format` changes
    how it is written down, never which one it is.
    """
    depth = int(depth)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    fmt = _resolve_format(format)
    group, n = _resolve_group(group, n)
    vocab = _group_elements(group, n)
    rng = random.Random(f"s5_composition|{group}|{n}|{depth}|{seed}")
    actions = [_s(vocab[rng.randrange(len(vocab))]) for _ in range(depth)]
    start = _s(tuple(range(1, n + 1)))
    return _build(start, actions, {"seed": seed, "group": group, "source": "generated"}, fmt)


_START_RE = re.compile(r"^Start(?::|\s+with\s+the\s+arrangement)\s+([1-9]+)\.?$")


def _parse_prompt(prompt: str):
    """
    (lines, start_i, start, story_i, actions) read out of the prompt TEXT.

    Format-agnostic on purpose: both the published and the ergonomic prompt open with a
    `Start...` line and carry the story on its own line of equal-length digit strings, so
    solve() and redact_prompt() need no knowledge of which format they were handed (and
    stay independent of inst.meta).
    """
    lines = prompt.splitlines()
    start = start_i = story_i = None
    for i, line in enumerate(lines):
        m = _START_RE.match(line.strip())
        if m is not None and start is None:
            start, start_i = m.group(1), i
            continue
        toks = line.split()
        if (
            story_i is None
            and start is not None
            and toks
            and all(t.isdigit() and "0" not in t and len(t) == len(start) for t in toks)
        ):
            story_i = i
    if start is None or story_i is None:
        raise ValueError("could not parse prompt")
    return lines, start_i, start, story_i, lines[story_i].split()


def solve(inst: Instance) -> str:
    """
    Independent reference solver.

    Independent of generate() in two ways: (a) it re-reads the problem from inst.prompt
    (the text the model sees), never from inst.meta or inst.states; (b) it folds the word
    from the RIGHT into a single composed permutation and applies it once, instead of
    walking the state forward one action at a time. Works on either format.
    """
    _, _, start, _, actions = _parse_prompt(inst.prompt)

    # fold right-to-left: w = a_1 . a_2 . ... . a_d, with (x . y)[i] = x[y[i]]
    word = _p(actions[-1])
    for a in reversed(actions[:-1]):
        x = _p(a)
        word = tuple(x[word[i] - 1] for i in range(len(word)))
    s0 = _p(start)
    return _s(tuple(s0[word[i] - 1] for i in range(len(word))))


def format_cot(inst: Instance) -> str:
    """Gold trace in the published format + the harness-imposed final Answer line."""
    return "\n".join(inst.steps) + "\nAnswer: " + inst.answer


def _extract(completion: str) -> str:
    """Last 'Answer:' line; else last non-empty line."""
    chosen = None
    for line in completion.splitlines():
        m = re.search(r"answer\s*[:=]\s*(.*)$", line, re.IGNORECASE)
        if m:
            chosen = m.group(1)
    if chosen is None:
        for line in completion.splitlines():
            if line.strip():
                chosen = line
    return "" if chosen is None else chosen


def _normalize(text: str) -> str:
    return "".join(re.findall(r"\d", text))


def check(inst: Instance, completion: str) -> bool:
    got = _normalize(_extract(completion))
    want = _normalize(inst.answer)
    return bool(want) and got == want


def step_spans(inst: Instance) -> list[tuple[int, int]]:
    """Char offsets of each step within format_cot() (for per-step attention blinding)."""
    text = format_cot(inst)
    spans, pos = [], 0
    for step in inst.steps:
        start = text.index(step, pos)
        spans.append((start, start + len(step)))
        pos = start + len(step)
    return spans


# --- AMENDMENT 4: prompt redaction ------------------------------------------------------
# The start arrangement IS the initial state, and each permutation in the story is the
# operator consumed by one step, so both are redactable; the composition rule is a static
# rule table and the question is the task, so both stay.
REDACTION_MEANINGFUL = True

_PLACEHOLDER = "[\u2026]"


def redact_prompt(inst: Instance, k: int) -> str:
    """
    Problem statement with everything needed to RECOMPUTE the state after step k removed:
    the start arrangement and the first k permutations become `[\u2026]` (one placeholder per
    removed item). Permutations k+1..depth, the composition rule and the question survive.
    k=0 returns the prompt unchanged; k>=depth removes the start and every permutation.
    Reads the prompt text (not meta), like solve().
    """
    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return inst.prompt
    k = min(k, inst.depth)

    lines, start_i, start, story_i, actions = _parse_prompt(inst.prompt)
    out = list(lines)
    # the start arrangement, wherever it sits in the sentence ("Start: 12345" /
    # "Start with the arrangement 12345.")
    out[start_i] = lines[start_i].replace(start, _PLACEHOLDER, 1)
    out[story_i] = " ".join([_PLACEHOLDER] * k + actions[k:])
    return "\n".join(out)


# --- AMENDMENT 6: step corruption -------------------------------------------------------
# The state is a permutation, so the minimal same-shape lie is a transposition: swap two
# positions of the arrangement the step reports. The result is always a different, legal
# arrangement (all digits are distinct), and the step's action text is untouched -- only the
# reported result changes.


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple[str, str]:
    """
    (step_text, corrupted_state) for step k (1-based), with the state it reports replaced by
    a plausible wrong one: `inst.states[k-1]` with two positions swapped.

    Deterministic in (inst, k, seed). Honours `inst.meta["format"]`:

      published:  step 2: 31254 -> 15324
      ergonomic:  step 2: apply 31254 to 53124 -> take positions 3,1,2,5,4 of 53124
                    -> 1 5 3 2 4 -> 15324

    (the published exemplar at k=2, seed=7; the true state after step 2 is 15342)

    In the ergonomic rendering both the spaced digits and the final arrangement are rewritten
    to the corrupted state (they must agree, or the lie is not a lie but a typo); the operand,
    the permutation and the `take positions` list are left exactly as the gold step has them.
    """
    k = int(k)
    if not 1 <= k <= inst.depth:
        raise ValueError(f"k must be in 1..{inst.depth}")
    fmt = _resolve_format(inst.meta.get("format", "published"))
    true = inst.states[k - 1]
    n = len(true)
    if n < 2:  # pragma: no cover - n >= 2 is enforced by _resolve_group
        raise ValueError("cannot corrupt a 1-item arrangement")

    rng = random.Random(
        "s5_composition|corrupt|{}|{}|{}|{}".format(
            inst.meta["start"], " ".join(inst.meta["actions"]), k, seed
        )
    )
    pairs = list(itertools.combinations(range(n), 2))
    i, j = pairs[rng.randrange(len(pairs))]
    digits = list(true)
    digits[i], digits[j] = digits[j], digits[i]
    corrupted = "".join(digits)

    prev = inst.meta["start"] if k == 1 else inst.states[k - 2]
    action = inst.meta["actions"][k - 1]
    return _render_step(k, action, prev, corrupted, fmt), corrupted


# --- the published exemplar ------------------------------------------------------------
# Verbatim from published_trace.txt (belindal/state-tracking, PermutationTask(num_items=5),
# random.seed(0)). Embedded so task.py never depends on file layout; the selftest
# cross-checks these constants against published_trace.txt when that file is present.
_PUBLISHED_START = "12345"
_PUBLISHED_ACTIONS = "53124 31254 51243 53421 31542 12543 23451 34521".split()
_PUBLISHED_STATES = "53124 15342 21543 35412 43215 43512 35124 12453".split()


def _published_exemplar(fmt: str = "published") -> Instance:
    inst = _build(
        _PUBLISHED_START,
        _PUBLISHED_ACTIONS,
        {"seed": 0, "group": "S5", "source": "published_trace.txt"},
        fmt,
    )
    if inst.states != _PUBLISHED_STATES:  # pragma: no cover - guards a code edit
        raise AssertionError("composition disagrees with the published state sequence")
    return inst


def exemplars(k: int, seed: int, **knobs) -> list[Instance]:
    """
    exemplars(k, seed)[0] is the published story; the rest are generated.

    AMENDMENT 5: knobs (notably `format`) are forwarded, so the exemplars the harness
    shows are written in the same format as the instance being solved. Under
    format="ergonomic" element 0 is still the published *story* (same actions, same
    states), re-rendered in the ergonomic trace format.
    """
    k = int(k)
    if k <= 0:
        return []
    fmt = _resolve_format(knobs.get("format", "published"))
    gen_knobs = {kk: vv for kk, vv in knobs.items() if kk in ("group", "n")}
    group, n = _resolve_group(gen_knobs.get("group", "S5"), gen_knobs.get("n"))
    out = [_published_exemplar(fmt)] if (group, n) == ("S5", 5) else []
    rng = random.Random(f"s5_composition|exemplars|{seed}")
    depths = [4, 6, 5, 8, 3]
    while len(out) < k:
        out.append(
            generate(
                depths[(len(out) - 1) % len(depths)],
                rng.randrange(10**9),
                format=fmt,
                **gen_knobs,
            )
        )
    return out[:k]


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------


def _wrong(answer: str) -> str:
    return answer[1] + answer[0] + answer[2:] if len(answer) >= 2 else "9"


def _check_published_file() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "published_trace.txt")
    if not os.path.exists(path):
        return "published_trace.txt absent, skipped"
    with open(path) as f:
        text = f.read()
    assert " ".join(_PUBLISHED_ACTIONS) in text, "embedded story != published_trace.txt"
    for i, st in enumerate(_PUBLISHED_STATES, start=1):
        assert f"step {i}: {st}" in text, f"embedded state {i} != published_trace.txt"
    return "published_trace.txt matches embedded exemplar"


def _selftest() -> int:
    rng = random.Random(20260916)
    groups = ["S5", "S5", "S5", "S3", "A5", "Zn"]  # S5 weighted: it is the default
    extra_depths = [1, 3, 5, 7, 40]
    n_cases = 200
    for _ in range(n_cases):
        depth = rng.choice(DEPTHS + extra_depths)
        seed = rng.randrange(10**9)
        group = groups[rng.randrange(len(groups))]
        inst = generate(depth, seed, group=group)

        assert solve(inst) == inst.answer, f"solve mismatch {group} d={depth} s={seed}"
        assert check(inst, format_cot(inst)), "check() rejected the gold completion"
        assert not check(inst, "Answer: " + _wrong(inst.answer)), "check() accepted a wrong answer"
        assert len(inst.steps) == len(inst.states) == depth, "len(steps)/len(states) != depth"
        again = generate(depth, seed, group=group)
        assert (again.prompt, again.steps, again.answer) == (
            inst.prompt,
            inst.steps,
            inst.answer,
        ), "generate() is not deterministic"

        text = format_cot(inst)
        for (a, b), step in zip(step_spans(inst), inst.steps):
            assert text[a:b] == step, "step_spans misaligned"
        assert "step" not in inst.prompt, "prompt leaked a scratchpad instruction"
        assert "Answer:" not in inst.prompt, "prompt leaked an answer instruction"

        # --- AMENDMENT 5: the ergonomic format is the SAME instance, written differently -
        erg = generate(depth, seed, group=group, format="ergonomic")
        assert erg.meta["format"] == "ergonomic" and inst.meta["format"] == "published"
        assert (erg.answer, erg.depth, erg.states) == (inst.answer, inst.depth, inst.states), (
            "format changed the instance"
        )
        assert (erg.meta["start"], erg.meta["actions"]) == (
            inst.meta["start"],
            inst.meta["actions"],
        ), "format changed the drawn word"
        assert solve(erg) == erg.answer == solve(inst), "solve() differs across formats"
        assert check(erg, format_cot(erg)), "check() rejected the ergonomic gold completion"
        assert not check(erg, "Answer: " + _wrong(erg.answer)), "check() accepted a wrong answer"
        assert check(erg, format_cot(inst)) and check(inst, format_cot(erg)), (
            "check() is not format-independent"
        )
        assert len(erg.steps) == len(erg.states) == depth, "len(steps)/len(states) != depth"
        assert erg.prompt != inst.prompt and erg.steps != inst.steps, "formats render alike"
        assert generate(depth, seed, group=group, format="ergonomic").steps == erg.steps, (
            "generate() is not deterministic under format='ergonomic'"
        )
        for (a, b), step in zip(step_spans(erg), erg.steps):
            assert format_cot(erg)[a:b] == step, "step_spans misaligned (ergonomic)"
        assert "step" not in erg.prompt and "Answer:" not in erg.prompt
        for j, (st, prev) in enumerate(zip(erg.steps, [erg.meta["start"]] + erg.states), start=1):
            assert st.startswith(f"step {j}: apply {inst.meta['actions'][j - 1]} to {prev} -> ")
            assert st.endswith(" -> " + erg.states[j - 1])

    ex = exemplars(3, 0)
    assert len(ex) == 3 and ex[0].meta["source"] == "published_trace.txt"
    assert ex[0].answer == _PUBLISHED_STATES[-1]
    ex_erg = exemplars(3, 0, format="ergonomic")
    assert len(ex_erg) == 3 and ex_erg[0].meta["source"] == "published_trace.txt"
    for e, g in zip(ex, ex_erg):
        for inst_ in (e, g):
            assert solve(inst_) == inst_.answer and check(inst_, format_cot(inst_))
            assert format_cot(inst_).strip()
        assert g.meta["format"] == "ergonomic" and e.meta["format"] == "published"
        assert (g.answer, g.depth, g.states) == (e.answer, e.depth, e.states), (
            "exemplars() drew different instances per format"
        )
        assert " -> take positions " in g.steps[0], "exemplars() did not forward format"

    # --- AMENDMENT 4/5: redaction, under BOTH formats ------------------------------------
    n_redact = 20
    for _ in range(n_redact):
        depth = rng.choice(DEPTHS + extra_depths)
        seed = rng.randrange(10**9)
        group = groups[rng.randrange(len(groups))]
        for fmt in FORMATS:
            inst = generate(depth, seed, group=group, format=fmt)
            start = inst.meta["start"]
            actions = inst.meta["actions"]
            p_lines, start_i, _, story_i, _ = _parse_prompt(inst.prompt)

            assert redact_prompt(inst, 0) == inst.prompt, "redact_prompt(inst, 0) != prompt"

            for k in sorted({0, 1, depth // 2, depth}):
                red = redact_prompt(inst, k)
                r_lines = red.splitlines()
                assert len(r_lines) == len(p_lines), "redaction changed the line count"
                toks = r_lines[story_i].split()
                assert len(toks) == depth, "redaction changed the permutation count"

                if k == 0:
                    assert red == inst.prompt, "k=0 must be a no-op"
                    continue

                assert red != inst.prompt, "k>0 did not change the prompt"
                assert _PLACEHOLDER in red, "k>0 produced no placeholder"
                assert red.count(_PLACEHOLDER) == k + 1, (
                    "expected one placeholder per removed permutation plus the start"
                )
                assert _PLACEHOLDER in r_lines[start_i] and start not in r_lines[start_i], (
                    "start arrangement not redacted"
                )
                assert toks[:k] == [_PLACEHOLDER] * k, "first k permutations not redacted"
                assert toks[k:] == actions[k:], "later permutations must survive intact"
                # static material survives (the rule/worked example and the question)
                assert p_lines[-1] == r_lines[-1], "the question was redacted"
                assert p_lines[-2] == r_lines[-2], "the composition rule was redacted"

            # k=depth: no initial-state or operator token is left in the instance-specific
            # part of the prompt (the start line and the story line). The rule line is
            # static -- identical for every instance of this n -- so a digit string that
            # happens to appear there carries no information about THIS instance.
            f_lines = redact_prompt(inst, depth).splitlines()
            assert f_lines[story_i].split() == [_PLACEHOLDER] * depth, "k=depth left a permutation"
            assert _PLACEHOLDER in f_lines[start_i], "k=depth left the start arrangement"
            recoverable = f_lines[start_i] + " " + f_lines[story_i]
            for tok in [start] + actions:
                assert tok not in recoverable, f"k=depth leaked the token {tok}"

    assert REDACTION_MEANINGFUL is True

    # --- AMENDMENT 6: step corruption, under BOTH formats --------------------------------
    n_corrupt = 20
    for _ in range(n_corrupt):
        depth = rng.choice(DEPTHS + extra_depths)
        seed = rng.randrange(10**9)
        group = groups[rng.randrange(len(groups))]
        cseed = rng.randrange(10**9)
        for k in sorted({1, max(1, depth // 2), depth}):
            per_format = {}
            for fmt in FORMATS:
                inst = generate(depth, seed, group=group, format=fmt)
                n = inst.meta["n"]
                gold = inst.steps[k - 1]
                true = inst.states[k - 1]
                prev = inst.meta["start"] if k == 1 else inst.states[k - 2]
                action = inst.meta["actions"][k - 1]

                text, bad = corrupt_step(inst, k, cseed)
                per_format[fmt] = (text, bad)

                assert bad != true, "corrupted state == true state"
                assert len(bad) == n and sorted(bad) == sorted(true), (
                    "corrupted state is not an arrangement of the same items"
                )
                assert sum(a != b for a, b in zip(bad, true)) == 2, (
                    "corruption is not a single transposition"
                )
                assert text != gold, "corrupted step text == gold step text"

                # task-specific structure: the step line still parses as a step line of this
                # format, and every field except the reported state is the gold one.
                if fmt == "published":
                    m = re.fullmatch(rf"step {k}: (\d{{{n}}}) -> (\d{{{n}}})", text)
                    assert m, f"corrupted step does not match the published step shape: {text!r}"
                    assert m.group(1) == action, "corruption changed the action"
                    assert m.group(2) == bad, "step text does not report the corrupted state"
                else:
                    m = re.fullmatch(
                        rf"step {k}: apply (\d{{{n}}}) to (\d{{{n}}}) -> take positions "
                        rf"([1-9](?:,[1-9])*) of (\d{{{n}}}) -> ((?:\d )*\d) -> (\d{{{n}}})",
                        text,
                    )
                    assert m, f"corrupted step does not match the ergonomic step shape: {text!r}"
                    assert m.group(1) == action and m.group(4) == prev, (
                        "corruption changed the action or the operand"
                    )
                    assert m.group(2) == prev, "corruption changed the operand"
                    assert m.group(3) == ",".join(action), (
                        "corruption changed the take-positions list"
                    )
                    assert m.group(5).replace(" ", "") == bad == m.group(6), (
                        "spaced digits and final arrangement disagree with the corrupted state"
                    )
                # only the trailing state field(s) differ from gold
                tail = 1 if fmt == "published" else 2
                assert text.split(" -> ")[:-tail] == gold.split(" -> ")[:-tail], (
                    "corruption rewrote more than the reported state"
                )
                assert corrupt_step(inst, k, cseed) == (text, bad), "corrupt_step is not deterministic"

            assert per_format["published"][1] == per_format["ergonomic"][1], (
                "corrupted state depends on the format"
            )
            assert per_format["published"][0] != per_format["ergonomic"][0], (
                "corrupted step text is identical across formats"
            )


    print(f"selftest: {n_cases} instances over groups S5/S3/A5/Zn, "
          f"depths {sorted(set(DEPTHS + extra_depths))}")
    print("  solve()==answer, check(gold)==True, check(wrong)==False, determinism, "
          "len(steps)==len(states)==depth: OK")
    print(f"  exemplars(3, 0): OK (exemplar 0 = published trace, answer {ex[0].answer})")
    print("  format='ergonomic': same instance/answer/states/depth/solve/check as "
          "'published', prompt+steps differ; exemplars() forwards format: OK")
    print(f"  redact_prompt: {n_redact} instances x 2 formats x k in "
          f"{{0, 1, depth//2, depth}} "
          f"(k=0 no-op, one '{_PLACEHOLDER}' per removed item, rule+question intact): OK")
    print(f"  corrupt_step: {n_corrupt} instances x 2 formats x k in "
          "{1, depth//2, depth} (one transposition of the reported "
          "arrangement, action text intact, deterministic): OK")
    print(f"  {_check_published_file()}")
    print("selftest PASS")
    return 0


def _render(inst: Instance, header: str) -> None:
    print("=" * 78)
    print(header)
    print("--- prompt ---")
    print(inst.prompt)
    print("--- format_cot (gold) ---")
    print(format_cot(inst))
    print()


def _demo() -> int:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            seed = 1000 + i
            inst = generate(depth, seed)
            _render(inst, f"group={inst.meta['group']}  depth={depth}  seed={seed}")

    # AMENDMENT 5: the same instance in each format, side by side.
    print("#" * 78)
    print("# AMENDMENT 5: one instance per format (identical instance, identical answer)")
    print("#" * 78)
    print()
    for fmt in FORMATS:
        inst = generate(6, 7, format=fmt)
        _render(
            inst,
            f"format={fmt}  group={inst.meta['group']}  depth={inst.depth}  seed=7  "
            f"answer={inst.answer}",
        )
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        sys.exit(_demo())
    else:
        print("usage: python3 task.py [--selftest|--demo]")
        sys.exit(2)
