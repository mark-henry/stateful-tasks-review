"""
dyck — close a Dyck-k prefix correctly, with the BBH stack-configuration scratchpad.

Published format: BIG-Bench-Hard `dyck_languages` 3-shot CoT prompt
(suzgunmirac/BIG-Bench-Hard, cot-prompts/dyck_languages.txt, commit
9ee07bd481feebf959a6b59d61ea57bdcf30964d; Suzgun et al. 2022, arXiv:2210.09261).

See README.md for citations, license, the format decision, depth semantics and caveats.
Pure python + stdlib only.
"""

from dataclasses import dataclass, field
import json
import os
import random
import re
import sys


# ---------------------------------------------------------------- interface types

@dataclass
class Instance:
    prompt: str
    steps: list
    states: list
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)


# The four bracket pairs used by BBH dyck_languages, in the order they appear in the
# canonical prompt file.
TYPES = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
_OPEN = [o for o, _ in TYPES]
_CLOSE_OF = {o: c for o, c in TYPES}
_ALL_BRACKETS = set("()[]{}<>")

ANSWER_FORMAT = "a sequence of closing brackets separated by single spaces, e.g. ] } )"

# depth = number of input symbols in the sequence to be completed = number of stack updates.
# BBH's own fixed set runs 2..99 symbols (median 13); 4 is trivial and 48 is past the
# long tail of the published set. See README "Depth semantics".
DEPTHS = [4, 8, 16, 24, 32, 48]

KNOBS = {
    "bracket_types": (4, "number of distinct bracket types the sequence may use, 1..4, drawn in "
                         "order from ( ) / [ ] / { } / < >; BBH uses all 4"),
    "max_nesting": (8, "cap on the stack depth (nesting level) reached anywhere in the sequence; "
                       "effectively clamped to `depth`. Held FIXED across the depth sweep so that "
                       "sequence length is the only thing varying. 8 was picked to track the "
                       "vendored BBH set's own mean max-nesting per length bucket (2.9 at len<=4, "
                       "6.4 at len<=16, 10.5 at len<=48; BBH has no cap)"),
    "max_answer_len": (4, "upper bound on the final stack size, i.e. on the number of closing "
                          "brackets in the answer. The final size has the same parity as `depth` "
                          "(each symbol moves the stack by +/-1), so the sampled length is drawn "
                          "from {v <= max_answer_len : v == depth (mod 2)}. BBH's fixed set only "
                          "ever shows 1, 2 or 3"),
}

_PREAMBLE = (
    "Let's think step by step.\n"
    "We should process each input one by one and keep track of the stack configuration.\n"
    "0: empty stack"
)

_PROMPT_TEMPLATE = (
    "Complete the rest of the sequence, making sure that the parentheses are closed properly. "
    "Input: {seq}"
)


# ---------------------------------------------------------------- rendering helpers

def _render_stack(stack):
    """The stack as the BBH step line shows it: space separated, or the word 'empty'."""
    return " ".join(stack) if stack else "empty"


def _state(stack):
    """Short canonical state string: stack contents bottom-to-top, no spaces; '-' when empty."""
    return "".join(stack) if stack else "-"


def _quoted(chars):
    return ", ".join('"%s"' % c for c in chars)


def _build(symbols, meta=None):
    """Build an Instance from a list of bracket symbols (the input sequence)."""
    stack = []
    steps = []
    states = []
    for i, sym in enumerate(symbols):
        if sym in _CLOSE_OF:
            stack.append(sym)
        else:
            if not stack or _CLOSE_OF[stack[-1]] != sym:
                raise ValueError("input sequence is not a valid Dyck prefix at position %d" % (i + 1))
            stack.pop()
        steps.append("%d: %s ; stack: %s" % (i + 1, sym, _render_stack(stack)))
        states.append(_state(stack))

    if not stack:
        raise ValueError("input sequence is already balanced; the answer would be empty")

    closers = [_CLOSE_OF[c] for c in reversed(stack)]
    answer = " ".join(closers)
    m = {"sequence": " ".join(symbols), "final_stack": " ".join(stack)}
    if meta:
        m.update(meta)
    return Instance(
        prompt=_PROMPT_TEMPLATE.format(seq=" ".join(symbols)),
        steps=steps,
        states=states,
        answer=answer,
        depth=len(symbols),
        meta=m,
    )


def _coda(inst):
    """The three closing lines of the published format."""
    stack = inst.meta["final_stack"].split()
    closers = [_CLOSE_OF[c] for c in reversed(stack)]
    return (
        'Now, we have reached the end. The final stack is "%s".\n'
        'We will need to pop out %s one by one in that order.\n'
        'So, we need %s. So the answer is %s.'
        % (" ".join(stack), _quoted(list(reversed(stack))), _quoted(closers), inst.answer)
    )


def format_cot(inst: Instance) -> str:
    """Gold trace in the published BBH format, plus the harness-imposed final Answer line."""
    return "%s\n%s\n%s\nAnswer: %s" % (
        _PREAMBLE, "\n".join(inst.steps), _coda(inst), inst.answer
    )


def step_spans(inst: Instance):
    """Char offsets (start, end) of each step line within format_cot(inst)."""
    spans = []
    pos = len(_PREAMBLE) + 1
    for s in inst.steps:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 1
    return spans


# ---------------------------------------------------------------- generator

def generate(depth: int, seed: int, **knobs) -> Instance:
    """
    Sample a Dyck-k prefix of exactly `depth` symbols whose stack is non-empty at the end.

    Pure function of (depth, seed, knobs). The walk is a stack process: at each step push a
    random bracket type or pop the top, restricted to moves from which the chosen final stack
    size is still reachable within the remaining steps and which stay within `max_nesting`.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    bracket_types = int(knobs.get("bracket_types", KNOBS["bracket_types"][0]))
    max_nesting = int(knobs.get("max_nesting", KNOBS["max_nesting"][0]))
    max_answer_len = int(knobs.get("max_answer_len", KNOBS["max_answer_len"][0]))
    if not 1 <= bracket_types <= 4:
        raise ValueError("bracket_types must be 1..4")
    if max_nesting < 1:
        raise ValueError("max_nesting must be >= 1")
    max_nesting = min(max_nesting, depth)

    rng = random.Random(seed)

    # Final stack size: same parity as depth (each symbol is a +/-1 move from 0), at least 1
    # (the answer must be non-empty), and no deeper than the nesting cap.
    hi = min(max_answer_len, max_nesting, depth)
    candidates = [v for v in range(1, hi + 1) if (depth - v) % 2 == 0]
    if not candidates:
        raise ValueError(
            "no feasible final stack size for depth=%d with max_nesting=%d, max_answer_len=%d "
            "(the final stack size must have the same parity as depth)"
            % (depth, max_nesting, max_answer_len)
        )
    target = rng.choice(candidates)

    stack = []
    symbols = []
    for i in range(depth):
        remaining = depth - i - 1
        moves = []
        if len(stack) < max_nesting and abs(target - (len(stack) + 1)) <= remaining:
            moves.append(1)
        if len(stack) > 0 and abs(target - (len(stack) - 1)) <= remaining:
            moves.append(-1)
        if not moves:  # pragma: no cover - excluded by the feasibility check above
            raise RuntimeError("stack walk became infeasible at step %d" % (i + 1))
        if rng.choice(moves) == 1:
            o = _OPEN[rng.randrange(bracket_types)]
            stack.append(o)
            symbols.append(o)
        else:
            symbols.append(_CLOSE_OF[stack.pop()])

    return _build(symbols, meta={
        "seed": seed,
        "bracket_types": bracket_types,
        "max_nesting": max_nesting,
        "max_answer_len": max_answer_len,
        "target_final_stack": target,
        "source": "generated",
    })


# ---------------------------------------------------------------- reference solver

def solve(inst: Instance) -> str:
    """
    Independent reference solver. Reads the input sequence back out of the PROMPT (not out of
    meta/steps/states) and reduces it by repeatedly deleting adjacent matched pairs until a
    fixed point; whatever survives is an all-open stack, and the answer is its mirror image.
    Deliberately a different algorithm from the incremental push/pop used by generate().
    """
    seq = inst.prompt.split("Input:", 1)[1]
    s = "".join(ch for ch in seq if ch in _ALL_BRACKETS)

    pairs = ["()", "[]", "{}", "<>"]
    while True:
        before = s
        for p in pairs:
            s = s.replace(p, "")
        if s == before:
            break

    if any(ch not in _CLOSE_OF for ch in s):
        raise ValueError("input is not a valid Dyck prefix: unmatched closing bracket")
    return " ".join(_CLOSE_OF[ch] for ch in reversed(s))


# ---------------------------------------------------------------- answer checking

def _extract(completion: str) -> str:
    """Last `Answer:` line, else the last non-empty line."""
    lines = [ln for ln in completion.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if "Answer:" in ln:
            return ln.split("Answer:", 1)[1]
    return lines[-1] if lines else ""


def _normalize(text: str) -> str:
    """Bracket characters only: tolerates '] } ]', ']}]', 'So the answer is ] } ].'"""
    return "".join(ch for ch in text if ch in _ALL_BRACKETS)


def check(inst: Instance, completion: str) -> bool:
    got = _normalize(_extract(completion))
    want = _normalize(inst.answer)
    return bool(want) and got == want


# ---------------------------------------------------------------- prompt redaction

# AMENDMENT 4: prompt blinding. The only state here is the stack, and it starts empty and is
# never written in the prompt, so there is no initial state to remove: the operators are the
# input symbols themselves. Redacting k means dropping the first k symbols. The symbols are not
# individually delimited beyond the spaces, so the removed run is replaced by ONE placeholder.
REDACTION_MEANINGFUL = True
REDACTION_PLACEHOLDER = "[…]"


def redact_prompt(inst: Instance, k: int) -> str:
    """
    The problem statement with the first `k` input symbols replaced by a single `[…]` span.

    The symbols consumed by steps 1..k are exactly what a model would need to recompute the stack
    after step k; the symbols for steps k+1..depth, and the question, are left intact. k=0 returns
    `inst.prompt` unchanged; k=depth leaves only the placeholder (the initial stack is empty and
    never appears in the prompt, so nothing else is state).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    k = min(k, inst.depth)
    if k == 0:
        return inst.prompt
    symbols = inst.meta["sequence"].split()
    return _PROMPT_TEMPLATE.format(seq=" ".join([REDACTION_PLACEHOLDER] + symbols[k:]).strip())


# ---------------------------------------------------------------- step corruption

# AMENDMENT 6: mistake propagation. The whole state of this task is the stack, and each step
# line reports it after the `; stack: ` marker. A corrupted step keeps the action half of the
# line (`k: <symbol>`) byte-for-byte and rewrites only the reported stack, to a stack that is a
# minimal edit away from the true one: replace the top symbol with a different opening bracket,
# drop the top symbol, or push one extra opening bracket. An empty true stack cannot be edited
# down, so it is corrupted upwards to a one-symbol stack.

_STACK_MARKER = " ; stack: "


def _stack_alphabet(inst: Instance):
    """The opening brackets this instance actually uses, so the wrong stack stays plausible."""
    used = [o for o in _OPEN if o in inst.meta["sequence"]]
    return used or list(_OPEN)


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple:
    """
    Step `k` (1-based) rewritten to report a plausible WRONG stack, plus that stack as a state
    string in the same canonical form as `inst.states` ('-' when empty).

    Deterministic in (inst, k, seed). The returned stack is never equal to `inst.states[k-1]`,
    is always a legal stack (openers only, drawn from the brackets the instance uses), and the
    `k: <symbol>` half of the line is untouched — only the reported result changes.
    """
    if not 1 <= k <= inst.depth:
        raise ValueError("k must be in 1..depth (got %d, depth=%d)" % (k, inst.depth))

    true_state = inst.states[k - 1]
    stack = [] if true_state == "-" else list(true_state)
    alphabet = _stack_alphabet(inst)
    rng = random.Random("dyck|corrupt|%d|%d|%s" % (seed, k, inst.meta["sequence"]))

    if not stack:
        # Nothing to edit away; report a spurious single symbol instead.
        bad = [rng.choice(alphabet)]
    else:
        alts = [o for o in alphabet if o != stack[-1]] or [o for o in _OPEN if o != stack[-1]]
        moves = ["replace", "drop", "push"] if alts else ["drop", "push"]
        move = rng.choice(moves)
        if move == "replace":
            bad = stack[:-1] + [rng.choice(alts)]
        elif move == "drop":
            bad = stack[:-1]
        else:
            bad = stack + [rng.choice(alphabet)]

    head = inst.steps[k - 1].split(_STACK_MARKER, 1)[0]
    return head + _STACK_MARKER + _render_stack(bad), _state(bad)


# ---------------------------------------------------------------- vendored fixed set

_HERE = os.path.dirname(os.path.abspath(__file__))
_BBH_JSON = os.path.join(_HERE, "vendor", "BIG-Bench-Hard", "bbh", "dyck_languages.json")


def bbh_instances():
    """The 250 canonical BBH dyck_languages examples, wrapped in the common interface.

    Fixed set: contaminated / canary-tagged. Use generate() for fresh instances.
    """
    with open(_BBH_JSON) as fh:
        data = json.load(fh)
    out = []
    for i, ex in enumerate(data["examples"]):
        symbols = ex["input"].split("Input:", 1)[1].split()
        inst = _build(symbols, meta={"source": "bbh", "index": i, "bbh_target": ex["target"]})
        out.append(inst)
    return out


# The first few-shot exemplar of the published BBH CoT prompt, verbatim (see
# published_trace.txt). exemplars()[0] reconstructs it.
_PUBLISHED_EXEMPLAR_INPUT = "[ { ["


def exemplars(k: int, seed: int):
    """Few-shot exemplars. [0] is the verbatim published BBH exemplar; the rest are generated."""
    out = [_build(_PUBLISHED_EXEMPLAR_INPUT.split(), meta={"source": "published_trace"})]
    rng = random.Random(seed)
    depths = [10, 16, 12, 20, 14, 8]
    i = 0
    while len(out) < k:
        d = depths[i % len(depths)]
        out.append(generate(d, rng.randrange(1 << 30)))
        i += 1
    return out[:k]


# ---------------------------------------------------------------- selftest / demo

def _published_trace_completion():
    """The 'A:' portion of published_trace.txt, for a verbatim comparison in --selftest."""
    path = os.path.join(_HERE, "published_trace.txt")
    if not os.path.exists(path):
        return None
    body = []
    started = False
    for ln in open(path).read().splitlines():
        if ln.startswith("A: "):
            started = True
            body.append(ln[3:])
        elif started:
            body.append(ln)
    return "\n".join(body).rstrip("\n") if started else None


def _selftest(n=200):
    failures = []

    # 1. The published exemplar must render byte-for-byte as published.
    pub = _published_trace_completion()
    ex0 = exemplars(1, 0)[0]
    if pub is not None:
        ours = format_cot(ex0)
        ours_no_answer = ours.rsplit("\nAnswer: ", 1)[0]
        if ours_no_answer != pub:
            failures.append("published exemplar mismatch:\n--ours--\n%s\n--published--\n%s"
                            % (ours_no_answer, pub))
    if ex0.prompt != _PROMPT_TEMPLATE.format(seq="[ { ["):
        failures.append("published exemplar prompt mismatch: %r" % ex0.prompt)
    if ex0.answer != "] } ]":
        failures.append("published exemplar answer mismatch: %r" % ex0.answer)

    # 2. The vendored BBH fixed set: our solver must reproduce all 250 published targets.
    try:
        fixed = bbh_instances()
    except OSError as e:
        failures.append("could not read vendored BBH data: %s" % e)
        fixed = []
    for inst in fixed:
        if inst.answer != inst.meta["bbh_target"] or solve(inst) != inst.meta["bbh_target"]:
            failures.append("BBH example %d: got %r / %r want %r"
                            % (inst.meta["index"], inst.answer, solve(inst), inst.meta["bbh_target"]))
    if fixed:
        print("BBH fixed set: %d/%d published targets reproduced by solve()"
              % (sum(1 for i in fixed if solve(i) == i.meta["bbh_target"]), len(fixed)))

    # 3. 200 random (depth, seed, knobs) triples.
    rng = random.Random(20260916)
    tested = 0
    while tested < n:
        depth = rng.randint(1, 60)
        seed = rng.randrange(1 << 30)
        knobs = {
            "bracket_types": rng.randint(1, 4),
            "max_nesting": rng.randint(1, 10),
            "max_answer_len": rng.randint(1, 5),
        }
        try:
            inst = generate(depth, seed, **knobs)
        except ValueError:
            continue  # infeasible knob combination (parity), not a failure
        tested += 1
        tag = "depth=%d seed=%d %s" % (depth, seed, knobs)

        if solve(inst) != inst.answer:
            failures.append("%s: solve()=%r != answer=%r" % (tag, solve(inst), inst.answer))
        if not check(inst, format_cot(inst)):
            failures.append("%s: check(format_cot) is False" % tag)

        wrong_first = next(c for c in ")]}>" if c != inst.answer.split()[0])
        wrong = " ".join([wrong_first] + inst.answer.split()[1:])
        if check(inst, "Answer: " + wrong):
            failures.append("%s: check accepted wrong answer %r" % (tag, wrong))
        if check(inst, "Answer: "):
            failures.append("%s: check accepted an empty answer" % tag)

        again = generate(depth, seed, **knobs)
        if (again.prompt, again.answer, again.steps, again.states) != \
           (inst.prompt, inst.answer, inst.steps, inst.states):
            failures.append("%s: generate is not deterministic" % tag)

        if not (len(inst.steps) == len(inst.states) == inst.depth == depth):
            failures.append("%s: len(steps)=%d len(states)=%d depth=%d"
                            % (tag, len(inst.steps), len(inst.states), inst.depth))
        if len(inst.answer.split()) > knobs["max_answer_len"]:
            failures.append("%s: answer longer than max_answer_len" % tag)
        if max(len(s) if s != "-" else 0 for s in inst.states) > min(knobs["max_nesting"], depth):
            failures.append("%s: nesting exceeded max_nesting" % tag)
        if len(set(inst.meta["sequence"].replace(" ", "")) & set(_OPEN)) > knobs["bracket_types"]:
            failures.append("%s: used more bracket types than allowed" % tag)

        cot = format_cot(inst)
        for (a, b), s in zip(step_spans(inst), inst.steps):
            if cot[a:b] != s:
                failures.append("%s: step_spans misaligned" % tag)
                break

    # 4. exemplars renders.
    exs = exemplars(3, 0)
    if len(exs) != 3:
        failures.append("exemplars(3, 0) returned %d instances" % len(exs))
    for e in exs:
        if not check(e, format_cot(e)) or solve(e) != e.answer:
            failures.append("exemplar does not round-trip: %r" % e.prompt)

    # 5. Redaction (AMENDMENT 4): 20 instances, k in {0, 1, depth//2, depth}.
    rrng = random.Random(20260918)
    redacted = 0
    for _ in range(20):
        depth = rrng.randint(2, 40)
        seed = rrng.randrange(1 << 30)
        try:
            inst = generate(depth, seed)
        except ValueError:
            continue
        redacted += 1
        symbols = inst.meta["sequence"].split()
        for k in sorted({0, 1, inst.depth // 2, inst.depth}):
            r = redact_prompt(inst, k)
            tag = "depth=%d seed=%d k=%d" % (depth, seed, k)
            if k == 0:
                if r != inst.prompt:
                    failures.append("%s: redact_prompt(inst, 0) != inst.prompt" % tag)
                continue
            if r == inst.prompt:
                failures.append("%s: redacted prompt is unchanged" % tag)
            if REDACTION_PLACEHOLDER not in r:
                failures.append("%s: redacted prompt lacks the placeholder" % tag)
            if r.count(REDACTION_PLACEHOLDER) != 1:
                failures.append("%s: expected exactly one placeholder span" % tag)
            # Task-specific: with the placeholder removed, the only bracket characters left are
            # the symbols for steps k+1..depth, in order. At k=depth none survive.
            body = r.split("Input:", 1)[1].replace(REDACTION_PLACEHOLDER, "")
            kept = "".join(ch for ch in body if ch in _ALL_BRACKETS)
            if kept != "".join(symbols[k:]):
                failures.append("%s: kept symbols %r != expected %r" % (tag, kept, "".join(symbols[k:])))
            if k == inst.depth and kept != "":
                failures.append("%s: initial-state/operator tokens survived full redaction" % tag)
            if not r.startswith(_PROMPT_TEMPLATE.split("Input:", 1)[0]):
                failures.append("%s: question text was damaged by redaction" % tag)
    if not REDACTION_MEANINGFUL:
        failures.append("dyck redaction is meaningful; REDACTION_MEANINGFUL must be True")
    try:
        redact_prompt(exemplars(1, 0)[0], -1)
        failures.append("redact_prompt accepted k=-1")
    except ValueError:
        pass

    # 6. Step corruption (AMENDMENT 6): 20 instances, k in {1, depth//2, depth}.
    step_re = re.compile(r"^(\d+): (\S) ; stack: (empty|[(\[{<](?: [(\[{<])*)$")
    crng = random.Random(20260919)
    corrupted = 0
    for _ in range(20):
        depth = crng.randint(2, 40)
        seed = crng.randrange(1 << 30)
        try:
            inst = generate(depth, seed)
        except ValueError:
            continue
        corrupted += 1
        symbols = inst.meta["sequence"].split()
        for k in sorted({1, max(1, inst.depth // 2), inst.depth}):
            tag = "depth=%d seed=%d k=%d" % (depth, seed, k)
            text, bad = corrupt_step(inst, k, seed)

            if bad == inst.states[k - 1]:
                failures.append("%s: corrupted_state equals the true state %r" % (tag, bad))
            if text == inst.steps[k - 1]:
                failures.append("%s: corrupted step text is unchanged" % tag)

            # Task-specific structure: still an `<index>: <symbol> ; stack: ...` line, with the
            # index and the action symbol untouched, and a stack of opening brackets only.
            m = step_re.match(text)
            if not m:
                failures.append("%s: corrupted step does not match the step-line structure: %r"
                                % (tag, text))
                continue
            if int(m.group(1)) != k or m.group(2) != symbols[k - 1]:
                failures.append("%s: corrupted step altered the action half: %r" % (tag, text))
            if text.split(_STACK_MARKER, 1)[0] != inst.steps[k - 1].split(_STACK_MARKER, 1)[0]:
                failures.append("%s: corrupted step altered the action half: %r" % (tag, text))

            # The reported stack and the returned state must agree, and the state must be canonical.
            shown = [] if m.group(3) == "empty" else m.group(3).split()
            if _state(shown) != bad:
                failures.append("%s: reported stack %r != corrupted_state %r" % (tag, m.group(3), bad))
            if bad != "-" and (" " in bad or any(ch not in _OPEN for ch in bad)):
                failures.append("%s: corrupted_state is not canonical: %r" % (tag, bad))

            # Plausible shape: a minimal edit of the true stack (length within 1, openers only).
            true_stack = [] if inst.states[k - 1] == "-" else list(inst.states[k - 1])
            if abs(len(shown) - len(true_stack)) > 1:
                failures.append("%s: corrupted stack is not a minimal edit: %r vs %r"
                                % (tag, bad, inst.states[k - 1]))

            if corrupt_step(inst, k, seed) != (text, bad):
                failures.append("%s: corrupt_step is not deterministic" % tag)
            if corrupt_step(generate(depth, seed), k, seed) != (text, bad):
                failures.append("%s: corrupt_step is not a pure function of (inst, k, seed)" % tag)

        # The corrupted step must splice into the gold trace exactly where the real one sat
        # (this mirrors what the batch-3 harness does to build the `mistake` prefix).
        kk = max(1, inst.depth // 2)
        text, _ = corrupt_step(inst, kk, seed)
        head = "%s\n%s" % (_PREAMBLE, "\n".join(inst.steps[:kk - 1]))
        full = format_cot(inst)
        j = full.find(inst.steps[kk - 1], len(head))
        if j < 0:
            failures.append("depth=%d seed=%d: step %d not locatable in the trace" % (depth, seed, kk))
        else:
            spliced = (full[:j] + text).splitlines()
            if spliced[-1] != text or spliced[:-1] != head.splitlines():
                failures.append("depth=%d seed=%d: corrupted step does not splice cleanly"
                                % (depth, seed))
    for bad_k in (0, -1, 10 ** 6):
        try:
            corrupt_step(exemplars(1, 0)[0], bad_k, 0)
            failures.append("corrupt_step accepted out-of-range k=%d" % bad_k)
        except ValueError:
            pass

    print("selftest: %d random instances, %d exemplars, %d BBH examples, %d redaction instances, "
          "%d corruption instances"
          % (tested, len(exs), len(fixed), redacted, corrupted))
    if failures:
        print("FAIL (%d):" % len(failures))
        for f in failures[:10]:
            print("  " + f)
        return 1
    print("PASS")
    return 0


def _demo():
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for seed in range(3):
            inst = generate(depth, seed)
            print("=" * 78)
            print("depth=%d seed=%d  (bracket_types=%d max_nesting=%d)"
                  % (depth, seed, inst.meta["bracket_types"], inst.meta["max_nesting"]))
            print("-" * 78)
            print("Q: " + inst.prompt)
            print("A: " + format_cot(inst))
            print()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        _demo()
    else:
        print(__doc__.strip())
        print("\nusage: python3 task.py --selftest | --demo")
