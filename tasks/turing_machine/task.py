"""
turing_machine — step-by-step simulation of an m-tag system (TMBench).

Primary source (per desk.json):
    Wu, Han, Zhou, Huang, Zhang, "Computational Reasoning of Large Language Models"
    (Turing Machine Bench / TMBench), arXiv:2504.20771 (2025).
    Repo: https://github.com/HaitaoWuTJU/Turing-Machine-Bench
          commit 9e04fcaec05664aa28fc86f20d9a2bb0e17551ea (vendored in vendor/TMBench-repo)

MACHINE MODEL: an *m-tag system* (Post tag system with deletion number m), which is what TMBench
actually implements despite its name. There is no head-state register and no tape-position pointer.
The machine state is a single queue of symbols; each step reads the head symbol, appends that
symbol's production to the tail, and deletes m symbols from the head. It halts when the queue is
shorter than m. See README.md.

Pure python + stdlib. No network at runtime.
"""

from dataclasses import dataclass, field
import argparse
import importlib.util
import os
import random
import re
import string
import sys


# --------------------------------------------------------------------------------------------
# interface objects
# --------------------------------------------------------------------------------------------

@dataclass
class Instance:
    prompt: str          # problem statement only, published wording
    steps: list          # gold trace, one element per serial step, published format (multi-line)
    states: list         # tracked state after each step (len == len(steps))
    answer: str          # exact-match target, e.g. "[C A B]"
    depth: int           # number of tag-system rewrite steps requested
    meta: dict = field(default_factory=dict)


# The final queue state, in TMBench's bracketed space-separated notation.
ANSWER_FORMAT = "the final queue state as a bracketed, space-separated symbol list, e.g. [A B C]"

# AMENDMENT 4 (prompt redaction). The only per-instance state in the prompt is the `Init:` queue;
# the transition rules, the alphabet, m and the step budget are static material every remaining
# step needs. So redaction is all-or-nothing at k >= 1. See README.md :: Redaction.
REDACTION_MEANINGFUL = True
REDACTION_PLACEHOLDER = "[…]"

# Justified in README.md. 30 is TMBench's own published ceiling (max_step: 31 -> 30 transitions).
DEPTHS = [2, 4, 8, 12, 20, 30]

KNOBS = {
    "m": (2, "deletion number: symbols removed from the head each step, and the halt threshold "
             "(TMBench default delete_count=2)"),
    "alphabet_size": (5, "number of distinct symbols; the alphabet is the first N uppercase letters "
                         "(TMBench default symbol_set=['A','B','C','D','E'])"),
    "rule_min_len": (1, "minimum production length for a symbol's rewrite rule (TMBench default 1)"),
    "rule_max_len": (5, "maximum production length for a symbol's rewrite rule (TMBench default 5)"),
    "init_min_len": (2, "minimum length of the initial queue; clamped up to m (TMBench default "
                        "str_min_length = delete_count = 2)"),
    "init_max_len": (9, "maximum length of the initial queue (TMBench default "
                        "str_max_length = delete_count + 7 = 9)"),
}

_MAX_ATTEMPTS = 20000


def _knob(knobs, name):
    if name in knobs and knobs[name] is not None:
        return knobs[name]
    return KNOBS[name][0]


# --------------------------------------------------------------------------------------------
# published-format renderers  (verbatim TMBench layout; see published_trace.txt)
# --------------------------------------------------------------------------------------------

def _fmt_queue(q: str) -> str:
    return "[" + " ".join(q) + "]"


def _init_block(init: str) -> str:
    return ("### step 0:\n"
            "   - Action: Init\n"
            "   - Queue State: " + _fmt_queue(init))


def _step_block(n: int, head: str, appended: str, deleted: str, queue: str, halted: bool) -> str:
    return ("### step %d:\n"
            "   - Head Symbol: %s\n"
            "   - Action: Append %s to the end of the queue. Remove %s from the head.\n"
            "   - Queue State: %s%s"
            % (n, head, " ".join(appended), " ".join(deleted),
               _fmt_queue(queue), " <halt>" if halted else ""))


def _render_prompt(m: int, alphabet: str, init: str, rule: dict, depth: int) -> str:
    """Problem statement only: TMBench's machine definition + the instance parameters.

    The scratchpad instruction ("provide the queue's state at each step", "Simulation steps:")
    and the worked example are deliberately omitted -- the harness assembles those.
    """
    alpha = "{" + ", ".join(sorted(alphabet)) + "}"
    rules = "\n".join("%s : %s" % (k, " ".join(rule[k])) for k in sorted(rule))
    return (
        "Simulate a m-tag system. Stop upon reaching the halt condition or %d steps.\n"
        "\n"
        "## Rules for Simulation:\n"
        "1. In each transition, the machine performs the following steps:\n"
        "   - If the queue length is less than m, halt\n"
        "   - Read the head symbol of queue\n"
        "   - Append symbols to the tail based on the head symbol and the corresponding transition rule\n"
        "   - Delete m symbols from the head of the queue\n"
        "\n"
        "2. The machine halt if:\n"
        "   - The queue's length is less than m.\n"
        "\n"
        "## The Only Problem to Solve:\n"
        "m: %d\n"
        "Alphabet: %s\n"
        "Init: %s\n"
        "Transition Rules:\n"
        "%s"
        % (depth, m, alpha, _fmt_queue(init), rules)
    )


# --------------------------------------------------------------------------------------------
# generate()
# --------------------------------------------------------------------------------------------

def _simulate(init: str, rule: dict, m: int, depth: int):
    """Run `depth` steps. Returns a list of (head, appended, deleted, queue_after, halted_flag),
    or None if the machine halts before completing `depth` steps."""
    q = init
    out = []
    for _ in range(depth):
        if len(q) < m:
            return None                     # halt condition reached early
        head = q[0]
        if head not in rule:
            return None
        appended = rule[head]
        deleted = q[:m]
        q = q[m:] + appended
        out.append((head, appended, deleted, q, len(q) < m))
    return out


def _build(m: int, alphabet: str, init: str, rule: dict, depth: int, meta_extra: dict) -> Instance:
    trace = _simulate(init, rule, m, depth)
    if trace is None:
        raise ValueError("machine halts before %d steps" % depth)
    steps = [_step_block(i + 1, h, a, d, q, halted) for i, (h, a, d, q, halted) in enumerate(trace)]
    states = [q for (_h, _a, _d, q, _halt) in trace]
    answer = _fmt_queue(states[-1])
    meta = {"m": m, "alphabet": alphabet, "init": init, "rule": dict(rule),
            "init_block": _init_block(init), "halted": trace[-1][4]}
    meta.update(meta_extra)
    return Instance(
        prompt=_render_prompt(m, alphabet, init, rule, depth),
        steps=steps,
        states=states,
        answer=answer,
        depth=depth,
        meta=meta,
    )


def generate(depth: int, seed: int, **knobs) -> Instance:
    """Pure function of (depth, seed, knobs).

    depth = number of tag-system rewrite steps actually executed. Instances that would halt
    before `depth` steps are rejected and resampled, so len(steps) == len(states) == depth always.
    TMBench's own reject_sampling (no two consecutive identical queue states) is applied too.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    m = int(_knob(knobs, "m"))
    alphabet_size = int(_knob(knobs, "alphabet_size"))
    rule_min_len = int(_knob(knobs, "rule_min_len"))
    rule_max_len = int(_knob(knobs, "rule_max_len"))
    init_min_len = max(int(_knob(knobs, "init_min_len")), m)
    init_max_len = max(int(_knob(knobs, "init_max_len")), init_min_len)
    if m < 1:
        raise ValueError("m must be >= 1")
    if not 1 <= alphabet_size <= 26:
        raise ValueError("alphabet_size must be in 1..26")

    alphabet = string.ascii_uppercase[:alphabet_size]
    rng = random.Random(seed)

    for attempt in range(_MAX_ATTEMPTS):
        # same sampling shape as TMBench src/tag_generate.py random_rule/random_str
        rule = {c: "".join(rng.choices(alphabet, k=rng.randint(rule_min_len, rule_max_len)))
                for c in alphabet}
        init = "".join(rng.choices(alphabet, k=rng.randint(init_min_len, init_max_len)))

        trace = _simulate(init, rule, m, depth)
        if trace is None:
            continue
        # TMBench reject_sampling: reject fixed points / stalled trajectories
        seq = [init] + [t[3] for t in trace]
        if any(seq[i] == seq[i - 1] for i in range(1, len(seq))):
            continue
        return _build(m, alphabet, init, rule, depth,
                      {"seed": seed, "attempts": attempt + 1,
                       "knobs": {"m": m, "alphabet_size": alphabet_size,
                                 "rule_min_len": rule_min_len, "rule_max_len": rule_max_len,
                                 "init_min_len": init_min_len, "init_max_len": init_max_len}})

    raise RuntimeError("could not sample a non-halting instance for depth=%d after %d attempts "
                       "(knobs make survival to depth impossible?)" % (depth, _MAX_ATTEMPTS))


# --------------------------------------------------------------------------------------------
# solve() — independent reference, wrapping the vendored TMBench simulator
# --------------------------------------------------------------------------------------------

_VENDOR_TAG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vendor", "TMBench-repo", "src", "tag_generate.py")
_vendor_cache = []


class _FallbackTagSystem:
    """Used only if vendor/TMBench-repo is missing. Mirrors TMBench's mTagSystem.step()."""

    def __init__(self, initial_string, rules, delete_count=2):
        self.string = initial_string
        self.rules = rules
        self.delete_count = delete_count

    def step(self):
        if not self.string:
            return False
        current_symbol = self.string[0]
        if len(self.string) < self.delete_count:
            return False
        if current_symbol in self.rules:
            add_string = self.rules[current_symbol]
            self.string = self.string[self.delete_count:]
            self.string += add_string
            return True
        return False


def _mtag_class():
    """Load TMBench's own mTagSystem out of vendor/ (module-level random.seed(1) is contained)."""
    if _vendor_cache:
        return _vendor_cache[0]
    cls = _FallbackTagSystem
    if os.path.exists(_VENDOR_TAG):
        state = random.getstate()
        try:
            spec = importlib.util.spec_from_file_location("_tmbench_tag_generate", _VENDOR_TAG)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls = mod.mTagSystem
        except Exception:
            cls = _FallbackTagSystem
        finally:
            random.setstate(state)          # vendor module calls random.seed(1) at import
    _vendor_cache.append(cls)
    return cls


def solve(inst: Instance) -> str:
    """Reference solver: re-runs the machine with TMBench's own simulator, independent of the
    trajectory generate() computed. Must equal inst.answer."""
    cls = _mtag_class()
    tm = cls(inst.meta["init"], dict(inst.meta["rule"]), inst.meta["m"])
    for _ in range(inst.depth):
        if not tm.step():
            break
    return _fmt_queue(tm.string)


# --------------------------------------------------------------------------------------------
# check() / format_cot()
# --------------------------------------------------------------------------------------------

_ANSWER_LINE = re.compile(r"^\s*answer\s*:(.*)$", re.IGNORECASE)


def _normalize_queue(text: str) -> str:
    t = text.strip()
    m = re.search(r"\[([^\]]*)\]", t)
    if m is not None:
        t = m.group(1)
    for ch in " \t,'\"":
        t = t.replace(ch, "")
    return t.upper()


def check(inst: Instance, completion: str) -> bool:
    """Extract the last `Answer:` line (else the last non-empty line) and exact-match after
    normalization (brackets/spaces/commas stripped, as TMBench's own acc.py does)."""
    if completion is None:
        return False
    cand = None
    lines = completion.splitlines()
    for ln in reversed(lines):
        m = _ANSWER_LINE.match(ln)
        if m is not None:
            cand = m.group(1)
            break
    if cand is None:
        for ln in reversed(lines):
            if ln.strip():
                cand = ln
                break
    if cand is None:
        return False
    return _normalize_queue(cand) == _normalize_queue(inst.answer)


def format_cot(inst: Instance) -> str:
    """Gold trace in TMBench's published format (### step 0 Init block + one block per step),
    then the harness-imposed final `Answer:` line."""
    body = inst.meta["init_block"] + "\n\n" + "\n\n".join(inst.steps)
    return body + "\n\nAnswer: " + inst.answer


def step_spans(inst: Instance):
    """Char offsets (start, end) of each element of inst.steps within format_cot(inst)."""
    spans = []
    pos = len(inst.meta["init_block"]) + 2      # init block + "\n\n"
    for s in inst.steps:
        spans.append((pos, pos + len(s)))
        pos += len(s) + 2                       # block + "\n\n"
    return spans


# --------------------------------------------------------------------------------------------
# redact_prompt()  -- AMENDMENT 4 (prompt blinding)
# --------------------------------------------------------------------------------------------

_INIT_LINE = re.compile(r"^Init:[^\n]*$", re.MULTILINE)


def redact_prompt(inst: Instance, k: int) -> str:
    """Hide what steps 1..k consumed, keep what steps k+1..depth need.

    For an m-tag system the *only* per-instance state in the prompt is the `Init:` queue.
    The transition rules, the alphabet, m and the step budget are static material that every
    remaining step consumes -- there is no per-step operator list to trim -- so the redaction is
    all-or-nothing: k == 0 returns the prompt unchanged, any k >= 1 replaces the initial queue
    with a single `[...]` placeholder and changes nothing else. k therefore matters only through
    the trace prefix the harness supplies alongside this prompt (AMENDMENT 4, task-specific note
    for turing_machine).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return inst.prompt
    redacted, n = _INIT_LINE.subn("Init: " + REDACTION_PLACEHOLDER, inst.prompt, count=1)
    if n != 1:
        raise ValueError("prompt has no `Init:` line to redact")
    return redacted


# --------------------------------------------------------------------------------------------
# corrupt_step()  -- AMENDMENT 6 (mistake propagation)
# --------------------------------------------------------------------------------------------

# The one line of a step block that reports state. The trailing " <halt>" marker (present when the
# resulting queue is shorter than m) is captured so it can be carried over unchanged: a corruption
# preserves queue length, so it never changes whether the machine halted.
_QUEUE_STATE_LINE = re.compile(r"^(   - Queue State: )\[[^\]]*\](.*)$", re.MULTILINE)


def _corruption_candidates(state: str, alphabet: str):
    """All minimal edits of `state` that stay inside `alphabet` and preserve its length.

    Two kinds, per AMENDMENT 6's "prefer minimal edits":
      ("sub",  i, c) -- replace the symbol at position i with a different alphabet symbol c
      ("swap", i)    -- exchange the adjacent symbols at positions i, i+1 (only when they differ)
    """
    subs = [("sub", i, c) for i in range(len(state)) for c in alphabet if c != state[i]]
    swaps = [("swap", i) for i in range(len(state) - 1) if state[i] != state[i + 1]]
    return subs, swaps


def _apply_corruption(state: str, edit) -> str:
    if edit[0] == "sub":
        _, i, c = edit
        return state[:i] + c + state[i + 1:]
    _, i = edit
    return state[:i] + state[i + 1] + state[i] + state[i + 2:]


def corrupt_step(inst: Instance, k: int, seed: int) -> tuple:
    """Return (step_text, corrupted_state) for step k (1-based), with a plausible wrong queue.

    The m-tag system's whole state is the queue, and the published step block reports it on exactly
    one line -- `   - Queue State: [A B C]`. So the corruption rewrites that line and nothing else:
    the `### step N:` header, the `Head Symbol:` line and the `Action:` line (which restate the
    operator that step applied) are byte-identical to `inst.steps[k-1]`, as AMENDMENT 6 (d) requires.

    The wrong queue is one minimal edit away from the true one: either one symbol replaced by a
    different alphabet symbol, or two adjacent symbols transposed. Both preserve length and stay
    inside the alphabet, so the result is indistinguishable in shape from a real queue -- including
    its halt status, so a trailing ` <halt>` marker is carried over verbatim.

    `corrupted_state` is returned in the same canonical form as `inst.states`: the bare symbol
    string (e.g. "ACBA"), not the bracketed rendering used by `answer`.

    Deterministic in (inst, k, seed): the RNG is keyed on the instance's defining parameters
    (m, alphabet, init, rules), k and seed -- not on object identity or `hash()`.

    This task has a single trace format (there is no `format` knob; TMBench's published layout is
    the only one), so AMENDMENT 6 (e) is satisfied trivially.

    Raises ValueError for k outside 1..depth, or when the alphabet has a single symbol (every
    same-length string over a 1-symbol alphabet is the true state, so no legal-looking wrong state
    exists; this only arises under `alphabet_size=1`, which is outside the published regime).
    """
    if not 1 <= k <= inst.depth:
        raise ValueError("k must be in 1..%d, got %d" % (inst.depth, k))

    state = inst.states[k - 1]
    alphabet = inst.meta["alphabet"]
    subs, swaps = _corruption_candidates(state, alphabet)
    if not subs and not swaps:
        raise ValueError("no legal-looking wrong queue exists for state %r over alphabet %r"
                         % (state, alphabet))

    key = "turing_machine|m=%d|alpha=%s|init=%s|rules=%s|k=%d|seed=%d" % (
        inst.meta["m"], alphabet, inst.meta["init"],
        ",".join("%s:%s" % (c, inst.meta["rule"][c]) for c in sorted(inst.meta["rule"])),
        k, seed)
    rng = random.Random(key)

    # Swaps are the scarcer, more "slip-like" error; take one half the time when available,
    # otherwise fall back to a substitution (and vice versa for length-1 queues).
    if swaps and (not subs or rng.random() < 0.5):
        edit = rng.choice(swaps)
    else:
        edit = rng.choice(subs)
    corrupted = _apply_corruption(state, edit)

    step_text, n = _QUEUE_STATE_LINE.subn(
        lambda mo: mo.group(1) + _fmt_queue(corrupted) + mo.group(2), inst.steps[k - 1], count=1)
    if n != 1:
        raise ValueError("step %d has no `Queue State:` line to corrupt" % k)
    return step_text, corrupted


# --------------------------------------------------------------------------------------------
# exemplars()
# --------------------------------------------------------------------------------------------

# The verbatim worked example baked into TMBench's official eval prompt
# (vendor/TMBench-repo/src/prompt.py, generate_prompt(); saved to published_trace.txt).
_PUBLISHED = {"m": 2, "alphabet": "ABC", "init": "BCA",
              "rule": {"A": "CAC", "B": "A", "C": "B"}, "depth": 4}

_EXEMPLAR_DEPTH = 4


def _published_exemplar() -> Instance:
    return _build(_PUBLISHED["m"], _PUBLISHED["alphabet"], _PUBLISHED["init"],
                  _PUBLISHED["rule"], _PUBLISHED["depth"],
                  {"source": "TMBench src/prompt.py generate_prompt() worked example "
                             "(commit 9e04fcaec05664aa28fc86f20d9a2bb0e17551ea)",
                   "published_exemplar": True})


def exemplars(k: int, seed: int) -> list:
    """exemplars(k, seed)[0] is the verbatim published TMBench exemplar; the rest are generated
    at a modest depth (4)."""
    if k <= 0:
        return []
    out = [_published_exemplar()]
    rng = random.Random(seed)
    while len(out) < k:
        out.append(generate(_EXEMPLAR_DEPTH, rng.randrange(1, 10 ** 9)))
    return out[:k]


# --------------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------------

_TRACE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "published_trace.txt")


def _published_trace_body():
    """The '### step 0:' .. end portion of published_trace.txt, or None if unavailable."""
    if not os.path.exists(_TRACE_FILE):
        return None
    with open(_TRACE_FILE, "r") as fh:
        txt = fh.read()
    i = txt.find("### step 0:")
    if i < 0:
        return None
    return txt[i:].rstrip("\n")


def _wrong_answer(inst: Instance) -> str:
    """A queue guaranteed different from the gold one."""
    q = inst.states[-1]
    alpha = inst.meta["alphabet"]
    other = alpha[(alpha.index(q[0]) + 1) % len(alpha)] if len(alpha) > 1 else "Z"
    return _fmt_queue(other + q[1:]) if len(alpha) > 1 else _fmt_queue(q + "Z")


def _selftest() -> int:
    rng = random.Random(20260916)
    failures = []

    # 1. verbatim reproduction of the published exemplar trace
    ex0 = _published_exemplar()
    rendered = ex0.meta["init_block"] + "\n\n" + "\n\n".join(ex0.steps)
    pub = _published_trace_body()
    if pub is None:
        print("WARN: published_trace.txt missing; skipping verbatim-format check")
    elif rendered != pub:
        failures.append("rendered exemplar trace does not match published_trace.txt verbatim")
        print("--- rendered ---\n%s\n--- published ---\n%s" % (rendered, pub))
    else:
        print("ok: exemplar trace reproduces published_trace.txt byte-for-byte (%d steps)"
              % len(ex0.steps))

    # 2. 200 random (depth, seed) pairs
    n = 200
    for i in range(n):
        depth = rng.randint(1, 30)
        seed = rng.randrange(1, 10 ** 9)
        inst = generate(depth, seed)

        if len(inst.steps) != depth or len(inst.states) != depth:
            failures.append("len(steps)=%d len(states)=%d != depth=%d (seed=%d)"
                            % (len(inst.steps), len(inst.states), depth, seed))
        got = solve(inst)
        if got != inst.answer:
            failures.append("solve()=%r != answer=%r (depth=%d seed=%d)"
                            % (got, inst.answer, depth, seed))
        if not check(inst, format_cot(inst)):
            failures.append("check(format_cot) False (depth=%d seed=%d)" % (depth, seed))
        if check(inst, "Answer: <wrong>"):
            failures.append("check accepted '<wrong>' (depth=%d seed=%d)" % (depth, seed))
        if check(inst, "Answer: " + _wrong_answer(inst)):
            failures.append("check accepted a wrong queue (depth=%d seed=%d)" % (depth, seed))
        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer) != \
           (inst.prompt, inst.steps, inst.states, inst.answer):
            failures.append("generate not deterministic (depth=%d seed=%d)" % (depth, seed))
        # steps must be locatable in the rendered CoT
        cot = format_cot(inst)
        for (a, b), s in zip(step_spans(inst), inst.steps):
            if cot[a:b] != s:
                failures.append("step_spans mismatch (depth=%d seed=%d)" % (depth, seed))
                break
        if failures:
            break
    print("ok: %d random instances (solve==answer, check, determinism, step counts)" % n)

    # 3. knob sweep still satisfies the contract
    for m in (1, 2, 3):
        for asz in (2, 3, 5, 8):
            inst = generate(6, 12345, m=m, alphabet_size=asz)
            if len(inst.steps) != 6 or solve(inst) != inst.answer:
                failures.append("knob sweep failed (m=%d alphabet_size=%d)" % (m, asz))
    print("ok: knob sweep m in {1,2,3} x alphabet_size in {2,3,5,8}")

    # 4. exemplars render
    exs = exemplars(3, 0)
    if len(exs) != 3:
        failures.append("exemplars(3,0) returned %d" % len(exs))
    else:
        for e in exs:
            if not e.prompt or not format_cot(e) or not check(e, format_cot(e)):
                failures.append("exemplar did not render/check")
        if not exs[0].meta.get("published_exemplar"):
            failures.append("exemplars(3,0)[0] is not the published exemplar")
    print("ok: exemplars(3, 0) renders (%d exemplars, [0] = published TMBench example)" % len(exs))

    # 5. redaction (AMENDMENT 4)
    rng_r = random.Random(20260918)
    n_red = 20
    for _ in range(n_red):
        depth = rng_r.randint(2, 30)
        inst = generate(depth, rng_r.randrange(1, 10 ** 9))
        init_q = _fmt_queue(inst.meta["init"])
        if redact_prompt(inst, 0) != inst.prompt:
            failures.append("redact_prompt(inst, 0) != inst.prompt (depth=%d)" % depth)
        for k in (1, depth // 2, depth):
            red = redact_prompt(inst, k)
            if red == inst.prompt:
                failures.append("redact_prompt k=%d did not change the prompt (depth=%d)"
                                % (k, depth))
            if REDACTION_PLACEHOLDER not in red:
                failures.append("redact_prompt k=%d has no placeholder (depth=%d)" % (k, depth))
            # the initial queue is gone; the static material (rules, alphabet, m, budget) stays
            if init_q in red:
                failures.append("redact_prompt k=%d still shows the initial queue (depth=%d)"
                                % (k, depth))
            if "Init: " + REDACTION_PLACEHOLDER not in red:
                failures.append("redact_prompt k=%d did not blank the Init line (depth=%d)"
                                % (k, depth))
            for c in sorted(inst.meta["rule"]):
                if "%s : %s" % (c, " ".join(inst.meta["rule"][c])) not in red:
                    failures.append("redact_prompt k=%d dropped a transition rule (depth=%d)"
                                    % (k, depth))
                    break
            if "m: %d" % inst.meta["m"] not in red or "or %d steps" % depth not in red:
                failures.append("redact_prompt k=%d dropped m or the step budget (depth=%d)"
                                % (k, depth))
        # k = depth: nothing of the initial state survives -- the Init line carries only the
        # placeholder, and no bracketed queue appears anywhere in the prompt.
        red_full = redact_prompt(inst, depth)
        brackets = [b for b in re.findall(r"\[[^\]]*\]", red_full)
                    if b != REDACTION_PLACEHOLDER]
        if brackets:
            failures.append("redact_prompt k=depth leaks a bracketed queue %r (depth=%d)"
                            % (brackets[0], depth))
        if redact_prompt(inst, 1) != red_full:
            failures.append("redaction is not all-or-nothing across k (depth=%d)" % depth)
    try:
        redact_prompt(generate(4, 11), -1)
    except ValueError:
        pass
    else:
        failures.append("redact_prompt accepted k < 0")
    print("ok: redaction on %d instances (k in {0, 1, depth//2, depth}; rules/m/budget kept)"
          % n_red)

    # 6. step corruption (AMENDMENT 6)
    step_block_re = re.compile(
        r"^### step (\d+):\n"
        r"   - Head Symbol: ([A-Z])\n"
        r"   - Action: Append ([A-Z ]+) to the end of the queue\. Remove ([A-Z ]+) from the head\.\n"
        r"   - Queue State: \[([A-Z ]*)\]( <halt>)?$")
    rng_c = random.Random(20260918)
    n_cor = 20
    for _ in range(n_cor):
        depth = rng_c.randint(2, 30)
        inst = generate(depth, rng_c.randrange(1, 10 ** 9))
        seed_c = rng_c.randrange(1, 10 ** 9)
        for k in (1, depth // 2, depth):
            step_text, corrupted = corrupt_step(inst, k, seed_c)
            true_state = inst.states[k - 1]
            true_step = inst.steps[k - 1]
            if corrupted == true_state:
                failures.append("corrupt_step k=%d returned the true state (depth=%d)" % (k, depth))
            if step_text == true_step:
                failures.append("corrupt_step k=%d returned the true step text (depth=%d)"
                                % (k, depth))
            # structure: still a well-formed published step block, same step number
            mo = step_block_re.match(step_text)
            if mo is None:
                failures.append("corrupted step k=%d is not a well-formed step block (depth=%d)"
                                % (k, depth))
                break
            if int(mo.group(1)) != k:
                failures.append("corrupted step k=%d renumbered the block (depth=%d)" % (k, depth))
            # the reported queue is exactly `corrupted`, and the state is legal-looking
            if mo.group(5).replace(" ", "") != corrupted:
                failures.append("corrupted step k=%d disagrees with corrupted_state (depth=%d)"
                                % (k, depth))
            if len(corrupted) != len(true_state):
                failures.append("corrupted state k=%d changed the queue length (depth=%d)"
                                % (k, depth))
            if any(c not in inst.meta["alphabet"] for c in corrupted):
                failures.append("corrupted state k=%d left the alphabet (depth=%d)" % (k, depth))
            # minimal edit: one substitution, or one adjacent transposition
            diff = [i for i in range(len(true_state)) if true_state[i] != corrupted[i]]
            one_sub = len(diff) == 1
            one_swap = (len(diff) == 2 and diff[1] == diff[0] + 1
                        and true_state[diff[0]] == corrupted[diff[1]]
                        and true_state[diff[1]] == corrupted[diff[0]])
            if not (one_sub or one_swap):
                failures.append("corrupted state k=%d is not a minimal edit (depth=%d)"
                                % (k, depth))
            # (d) only the state line moved: header/Head Symbol/Action lines are byte-identical,
            # and so is the <halt> marker (corruption preserves length, hence halt status)
            true_lines, cor_lines = true_step.split("\n"), step_text.split("\n")
            if len(true_lines) != len(cor_lines) or true_lines[:3] != cor_lines[:3]:
                failures.append("corrupt_step k=%d touched a non-state line (depth=%d)"
                                % (k, depth))
            if (" <halt>" in true_step) != (" <halt>" in step_text):
                failures.append("corrupt_step k=%d changed the halt marker (depth=%d)" % (k, depth))
            # determinism
            if corrupt_step(inst, k, seed_c) != (step_text, corrupted):
                failures.append("corrupt_step not deterministic (k=%d depth=%d)" % (k, depth))
            if corrupt_step(generate(depth, inst.meta["seed"]), k, seed_c) != (step_text, corrupted):
                failures.append("corrupt_step not a pure function of (inst, k, seed) "
                                "(k=%d depth=%d)" % (k, depth))
        if failures:
            break
    for bad_k in (0, -1, inst.depth + 1):
        try:
            corrupt_step(inst, bad_k, 1)
        except ValueError:
            pass
        else:
            failures.append("corrupt_step accepted k=%d" % bad_k)
    print("ok: corruption on %d instances (k in {1, depth//2, depth}; minimal edit, block "
          "structure, action lines untouched, determinism)" % n_cor)

    # 7. DEPTHS are all generable
    for d in DEPTHS:
        inst = generate(d, 7)
        if len(inst.steps) != d or solve(inst) != inst.answer:
            failures.append("DEPTHS entry %d failed" % d)
    print("ok: every entry of DEPTHS=%s generates and solves" % (DEPTHS,))

    if failures:
        print("\nFAILED (%d):" % len(failures))
        for f in failures[:20]:
            print("  - " + f)
        return 1
    print("\nSELFTEST PASSED")
    return 0


def _demo() -> int:
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            seed = depth * 1000 + i
            inst = generate(depth, seed)
            print("=== depth=%d instance=%d seed=%d ===" % (depth, i, seed))
            print("--- prompt ---")
            print(inst.prompt)
            print()
            print("--- gold CoT ---")
            print(format_cot(inst))
            print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if args.demo:
        return _demo()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
