"""
multiplication — multi-digit long multiplication with the Faith-and-Fate scratchpad.

Primary source: Dziri et al. 2023, "Faith and Fate: Limits of Transformers on
Compositionality" (arXiv:2305.18654, NeurIPS 2023 Spotlight). The gold chain of thought
reproduces that paper's published scratchpad format EXACTLY, by wrapping the paper's own
generator:

    vendor/faith-and-fate/multiplication/generate_scratchpads.py :: generate_prompt(x, y)
    commit 1e90edb54b4ed0fa150259a72c994b0fee90d388, MIT, (c) 2022 Nouha Dziri

See README.md for the format decision, depth semantics, and caveats.
Pure python + stdlib only (the vendored module's single third-party import, tqdm, is
stubbed at load time so nothing outside the stdlib is actually required).
"""

from dataclasses import dataclass, field
import importlib.util
import os
import random
import re
import sys
import types


# --------------------------------------------------------------------------------------
# Instance
# --------------------------------------------------------------------------------------

@dataclass
class Instance:
    prompt: str          # problem statement only, published wording
    steps: list          # gold trace, one element per numbered step of the published format
    states: list         # tracked state after each step (len == len(steps))
    answer: str          # exact-match target: the product, as a decimal integer string
    depth: int           # number of serial (numbered) steps
    meta: dict = field(default_factory=dict)


ANSWER_FORMAT = "a single integer (the product), written in plain decimal digits with no commas or spaces"

# Operand shapes 1x1, 2x1, 2x2, 3x2, 3x3, 4x4, 5x5 (digits_x x digits_y), the lower-triangular
# grid Dziri et al. sweep. depth = digits_y * (digits_x + 1); see README for the rationale.
DEPTHS = [2, 3, 6, 8, 12, 20, 30]

KNOBS = {
    "digits": (
        None,
        "if set, both operands get exactly this many digits and the step count is derived as "
        "digits * (digits + 1), overriding the shape implied by the `depth` argument",
    ),
    "digits_x": (
        None,
        "if set, the left operand x gets exactly this many digits (overrides the shape implied "
        "by `depth` and by `digits`)",
    ),
    "digits_y": (
        None,
        "if set, the right operand y gets exactly this many digits; y is the operand whose digits "
        "drive the partial products, so this is the number of partial products (overrides `depth`/`digits`)",
    ),
}

# Hard ceiling inherited from the published generator: its place-name table runs
# "ones" .. "billions" (10 entries), and it names the place of every digit of BOTH operands,
# so neither operand may exceed 10 digits. Well beyond anything a 7B will survive.
MAX_DIGITS = 10

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_GEN = os.path.join(
    _HERE, "vendor", "faith-and-fate", "multiplication", "generate_scratchpads.py"
)
_PUBLISHED_TRACE = os.path.join(_HERE, "published_trace.txt")

# The published exemplar's operands (Dziri et al. 2023, Figure 9 / Appendix A.1).
PUBLISHED_EXEMPLAR = (35, 90)


# --------------------------------------------------------------------------------------
# Thin wrapper around the vendored (published) scratchpad generator
# --------------------------------------------------------------------------------------

_vendor_mod = None


def _vendor():
    """Import the vendored Faith-and-Fate scratchpad generator, stdlib-only.

    The vendored file does `from tqdm import tqdm` at module scope purely for its CLI main();
    generate_prompt() never touches it. We inject a stub module so that task.py has no
    third-party dependency, and so the published generator is used verbatim rather than
    re-typed (which is what keeps format_cot() byte-identical to the paper's format).
    """
    global _vendor_mod
    if _vendor_mod is not None:
        return _vendor_mod
    if "tqdm" not in sys.modules:
        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda it, *a, **k: it  # noqa: E731
        sys.modules["tqdm"] = stub
    spec = importlib.util.spec_from_file_location("_ff_multiplication", _VENDOR_GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _vendor_mod = mod
    return mod


_NUMBERED = re.compile(r"^(\d+)\. ")


def _split_published_trace(completion: str):
    """Split the published completion text into (steps, closing).

    A "step" is one numbered item ("1. ", "2. ", ...) of the published format, carrying any
    section-header scaffolding that immediately precedes it ("Now, let's multiply 35 by the
    digit in the tens place of 90, which is 9.") and any blank line that follows it. The
    unnumbered final summation paragraph is the `closing`, per AMENDMENT 3's "published
    closing" rule. Reconstruction is exact:

        "\\n".join(steps) + "\\n" + closing == completion
    """
    lines = completion.split("\n")
    blocks = []
    pending = []
    for line in lines:
        if _NUMBERED.match(line):
            lead = 0
            while lead < len(pending) and pending[lead] == "":
                lead += 1
            if blocks:
                # Blank lines before a step belong to the step before it, so that no block
                # starts with a newline. (Before the first block there is nothing to hang
                # them on; the published format never starts with a blank line anyway.)
                blocks[-1] += "\n" * lead
            blocks.append("\n".join(pending[lead:] + [line]))
            pending = []
        else:
            pending.append(line)

    lead = 0
    while lead < len(pending) and pending[lead] == "":
        lead += 1
    if blocks:
        blocks[-1] += "\n" * lead
    closing = "\n".join(pending[lead:])
    return blocks, closing


def _sample_operand(rng, n_digits):
    """A genuine n-digit number: leading digit 1-9, the rest 0-9."""
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(n_digits - 1)]
    val = first
    for d in rest:
        val = val * 10 + d
    return val


def _shape_for_depth(depth):
    """Canonical (digits_x, digits_y) for a requested depth.

    depth == digits_y * (digits_x + 1). Among all valid factorizations we take the most
    square one (minimising |digits_x - digits_y|), tie-broken toward the wider left operand.
    Depths whose only factorization needs digits_y == 1 give a k-digit x 1-digit multiply.

    Both digit counts are capped at MAX_DIGITS (see that constant): the published generator's
    place-name table ("ones", "tens", ..., "billions") runs out past 10 digits. Depths with no
    factorization inside the cap (e.g. 13, whose only shapes are 1x12 and 13x0) do not exist
    as instances of this format; feasible_depths() enumerates the ones that do.
    """
    best = None
    for dy in range(1, min(depth, MAX_DIGITS) + 1):
        if depth % dy:
            continue
        dx = depth // dy - 1
        if dx < 1 or dx > MAX_DIGITS:
            continue
        key = (abs(dx - dy), -dx)
        if best is None or key < best[0]:
            best = (key, (dx, dy))
    if best is None:
        raise ValueError(
            f"depth={depth} is not achievable: depth must equal digits_y * (digits_x + 1) "
            f"for integers 1 <= digits_x, digits_y <= {MAX_DIGITS}. Use feasible_depths() "
            f"to enumerate achievable depths."
        )
    return best[1]


def feasible_depths(lo=2, hi=110):
    """Depths in [lo, hi] that are achievable as digits_y * (digits_x + 1)."""
    out = []
    for d in range(max(2, lo), hi + 1):
        try:
            _shape_for_depth(d)
        except ValueError:
            continue
        out.append(d)
    return out


def _instance_from_operands(x, y, seed=None, requested_depth=None):
    ff = _vendor()
    completion, question, product = ff.generate_prompt(x, y)

    # Strip the two fine-tuning delimiters of the published data format: the leading space
    # that only exists because the completion is concatenated after "###\n\n", and the
    # trailing " ###" stop marker. Nothing else about the published text is touched.
    if completion.startswith(" "):
        completion = completion[1:]
    assert completion.endswith(" ###"), "vendored generator changed its stop marker"
    completion = completion[: -len(" ###")]

    steps, closing = _split_published_trace(completion)

    # States, recomputed from the task definition (not scraped out of the prose), in the same
    # order the published format emits its numbered steps.
    digits_x = ff.digits(x)
    digits_y = ff.digits(y)
    states = []
    for j, dy in enumerate(digits_y):
        carry = 0
        for i, dx in enumerate(digits_x):
            total = dx * dy + carry
            if i < len(digits_x) - 1:
                residual, carry = total % 10, total // 10
            else:
                residual, carry = total, 0
            states.append(f"{residual},{carry}")
        states.append(f"{chr(ord('A') + j)}={dy * x}")

    assert len(states) == len(steps), "state/step mismatch against the published format"

    prompt = question.split("\n", 1)[0].strip()  # "What is 35 times 90?"

    return Instance(
        prompt=prompt,
        steps=steps,
        states=states,
        answer=str(product),
        depth=len(steps),
        meta={
            "x": x,
            "y": y,
            "digits_x": len(digits_x),
            "digits_y": len(digits_y),
            "closing": closing,
            "seed": seed,
            "requested_depth": requested_depth,
        },
    )


# --------------------------------------------------------------------------------------
# Common interface
# --------------------------------------------------------------------------------------

def generate(depth: int, seed: int, **knobs) -> Instance:
    """Generate a long-multiplication instance whose published scratchpad has `depth`
    numbered steps. Pure function of (depth, seed, knobs)."""
    digits = knobs.get("digits")
    dx = knobs.get("digits_x")
    dy = knobs.get("digits_y")

    if dx is None or dy is None:
        if digits is not None:
            base_x = base_y = int(digits)
        else:
            base_x, base_y = _shape_for_depth(int(depth))
        dx = base_x if dx is None else int(dx)
        dy = base_y if dy is None else int(dy)
    dx, dy = int(dx), int(dy)
    if not (1 <= dx <= MAX_DIGITS) or not (1 <= dy <= MAX_DIGITS):
        raise ValueError(f"digit counts must be in 1..{MAX_DIGITS} (got {dx} x {dy})")

    # Seeded from a string so the stream is stable across runs and python builds.
    rng = random.Random(f"multiplication|{seed}|{depth}|{dx}|{dy}")
    x = _sample_operand(rng, dx)
    y = _sample_operand(rng, dy)
    return _instance_from_operands(x, y, seed=seed, requested_depth=depth)


def solve(inst: Instance) -> str:
    """Independent reference solver.

    Reads the operands back out of the published problem statement (so it does not trust
    generate()'s bookkeeping) and multiplies them with a schoolbook digit-array algorithm:
    a column accumulator of single-digit products followed by one carry-propagation pass.
    It never multiplies the full operands, and shares no code with the vendored partial-
    product generator that produced inst.steps / inst.answer.
    """
    m = re.search(r"What is (\d+) times (\d+)\?", inst.prompt)
    if not m:
        raise ValueError(f"cannot parse operands from prompt: {inst.prompt!r}")
    a_digits = [int(c) for c in reversed(m.group(1))]
    b_digits = [int(c) for c in reversed(m.group(2))]

    acc = [0] * (len(a_digits) + len(b_digits))
    for i, da in enumerate(a_digits):
        for j, db in enumerate(b_digits):
            acc[i + j] += da * db

    out = []
    carry = 0
    for v in acc:
        v += carry
        out.append(v % 10)
        carry = v // 10
    while carry:
        out.append(carry % 10)
        carry //= 10

    s = "".join(str(d) for d in reversed(out)).lstrip("0")
    return s or "0"


def format_cot(inst: Instance) -> str:
    """Gold trace in the published Faith-and-Fate format, plus the harness's Answer line."""
    return (
        "\n".join(inst.steps)
        + "\n"
        + inst.meta["closing"]
        + "\nAnswer: "
        + inst.answer
    )


def step_spans(inst: Instance):
    """Char offsets (start, end) of each step within format_cot(inst)."""
    spans = []
    pos = 0
    for i, step in enumerate(inst.steps):
        spans.append((pos, pos + len(step)))
        pos += len(step) + 1  # the "\n" used by the join
    return spans


_INT = re.compile(r"[-+]?\d[\d,]*")


def _extract(completion: str):
    """AMENDMENT 3 extraction rule: last 'Answer:' line, else last non-empty line; then the
    last integer-looking token on it (which also picks up the published closing's
    '... = 3150.' form when no Answer line is present)."""
    lines = completion.splitlines()
    target = None
    for line in reversed(lines):
        if "Answer:" in line:
            target = line.split("Answer:")[-1]
            break
    if target is None:
        for line in reversed(lines):
            if line.strip():
                target = line
                break
    if target is None:
        return None
    hits = _INT.findall(target.replace("###", " "))
    if not hits:
        return None
    tok = hits[-1].replace(",", "").lstrip("+")
    try:
        return str(int(tok))
    except ValueError:
        return None


def check(inst: Instance, completion: str) -> bool:
    got = _extract(completion)
    return got is not None and got == str(int(inst.answer))


def exemplars(k: int, seed: int) -> list:
    """Few-shot exemplars. exemplars(k, seed)[0] is the paper's own published exemplar
    (Dziri et al. 2023 Figure 9, operands 35 x 90, reproduced by the vendored generator);
    the rest are freshly generated at a modest depth (6 == 2-digit x 2-digit, the same
    shape as the published exemplar)."""
    out = []
    if k >= 1:
        out.append(_instance_from_operands(*PUBLISHED_EXEMPLAR, seed=None, requested_depth=6))
    for i in range(1, k):
        out.append(generate(6, seed * 1000 + i))
    return out[:k]


# --------------------------------------------------------------------------------------
# selftest / demo
# --------------------------------------------------------------------------------------

def _published_completion():
    """The completion half of published_trace.txt, with the ### delimiters stripped the same
    way _instance_from_operands does."""
    raw = open(_PUBLISHED_TRACE, encoding="utf-8").read()
    body = raw.split("===\n", 1)[1]
    completion = body.split("\n\n###\n\n", 1)[1].rstrip("\n")
    if completion.startswith(" "):
        completion = completion[1:]
    if completion.endswith(" ###"):
        completion = completion[: -len(" ###")]
    return completion


def _selftest():
    rng = random.Random(20260916)
    fails = []

    # 1) Byte-exact agreement with the published trace.
    pub = exemplars(1, 0)[0]
    rebuilt = "\n".join(pub.steps) + "\n" + pub.meta["closing"]
    if rebuilt != _published_completion():
        fails.append("published trace mismatch: format_cot does not reproduce published_trace.txt")
    if pub.depth != 6 or pub.answer != "3150":
        fails.append(f"published exemplar wrong: depth={pub.depth} answer={pub.answer}")

    # 2) 200 random (depth, seed) pairs.
    depths = sorted(set(DEPTHS) | set(feasible_depths(2, 30)))
    n = 200
    for t in range(n):
        depth = rng.choice(depths)
        seed = rng.randrange(10 ** 6)
        inst = generate(depth, seed)

        if solve(inst) != inst.answer:
            fails.append(f"solve mismatch at depth={depth} seed={seed}: "
                         f"{solve(inst)} != {inst.answer}")
        if not check(inst, format_cot(inst)):
            fails.append(f"check(format_cot) False at depth={depth} seed={seed}")
        wrong = str(int(inst.answer) + 1 + rng.randrange(9))
        if check(inst, f"Answer: {wrong}"):
            fails.append(f"check accepted a wrong answer at depth={depth} seed={seed}")
        again = generate(depth, seed)
        if (again.prompt, again.steps, again.states, again.answer, again.depth) != (
                inst.prompt, inst.steps, inst.states, inst.answer, inst.depth):
            fails.append(f"non-deterministic at depth={depth} seed={seed}")
        if not (len(inst.steps) == len(inst.states) == inst.depth == depth):
            fails.append(f"depth mismatch at depth={depth} seed={seed}: "
                         f"steps={len(inst.steps)} states={len(inst.states)} depth={inst.depth}")
        # steps must reconstruct format_cot exactly, and spans must land on them
        cot = format_cot(inst)
        for (s, e), step in zip(step_spans(inst), inst.steps):
            if cot[s:e] != step:
                fails.append(f"step_spans misaligned at depth={depth} seed={seed}")
                break
        if not cot.endswith("\nAnswer: " + inst.answer):
            fails.append(f"format_cot missing Answer line at depth={depth} seed={seed}")
        # the published closing must actually state the product
        if inst.answer not in inst.meta["closing"]:
            fails.append(f"closing does not contain the answer at depth={depth} seed={seed}")

    # 3) knobs
    kn = generate(6, 1, digits=3)
    if (kn.meta["digits_x"], kn.meta["digits_y"], kn.depth) != (3, 3, 12):
        fails.append(f"digits knob broken: {kn.meta['digits_x']}x{kn.meta['digits_y']} depth={kn.depth}")
    kn = generate(6, 1, digits_x=5, digits_y=2)
    if (kn.meta["digits_x"], kn.meta["digits_y"], kn.depth) != (5, 2, 12):
        fails.append(f"digits_x/digits_y knobs broken: depth={kn.depth}")
    if solve(kn) != kn.answer:
        fails.append("solve mismatch under knob override")

    # 4) exemplars render
    ex = exemplars(3, 0)
    if len(ex) != 3:
        fails.append(f"exemplars(3, 0) returned {len(ex)} instances")
    for e in ex:
        if not format_cot(e).strip():
            fails.append("exemplar rendered empty")
        if solve(e) != e.answer:
            fails.append("exemplar solve mismatch")

    # 5) every DEPTHS value is achievable
    for d in DEPTHS:
        inst = generate(d, 7)
        if inst.depth != d:
            fails.append(f"DEPTHS value {d} not achievable (got {inst.depth})")

    if fails:
        print("SELFTEST FAILED")
        for f in fails[:20]:
            print("  -", f)
        print(f"  ({len(fails)} failure(s))")
        return 1
    print(f"multiplication selftest: {n} random instances OK")
    print(f"  solve()==answer, check(format_cot)==True, check(wrong)==False, deterministic, "
          f"len(steps)==len(states)==depth")
    print(f"  format_cot reproduces published_trace.txt (Dziri et al. 2023 Fig. 9, 35 x 90) byte-for-byte")
    print(f"  knobs digits/digits_x/digits_y OK; exemplars(3, 0) OK; DEPTHS={DEPTHS} all achievable")
    print("PASS")
    return 0


def _demo():
    for depth in (DEPTHS[0], DEPTHS[-1]):
        for i in range(3):
            inst = generate(depth, 100 + i)
            print("=" * 78)
            print(f"depth={inst.depth}  ({inst.meta['digits_x']}-digit x {inst.meta['digits_y']}-digit)"
                  f"  seed={inst.meta['seed']}")
            print("=" * 78)
            print("--- prompt ---")
            print(inst.prompt)
            print("--- gold CoT (format_cot) ---")
            print(format_cot(inst))
            print("--- states ---")
            print(" | ".join(inst.states))
            print()
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    elif "--demo" in sys.argv:
        sys.exit(_demo())
    else:
        print("usage: python3 task.py [--selftest | --demo]")
        sys.exit(2)
