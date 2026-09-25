"""Stub task for harness tests: mod-10 counter. Satisfies the AMENDMENT 3 contract."""
from dataclasses import dataclass, field
import random, re, sys

@dataclass
class Instance:
    prompt: str
    steps: list
    states: list
    answer: str
    depth: int
    meta: dict = field(default_factory=dict)

ANSWER_FORMAT = "a single digit 0-9"
DEPTHS = [2, 4, 8]
KNOBS = {"mod": (10, "modulus")}

def generate(depth, seed, mod=10):
    r = random.Random(seed)
    start = r.randrange(mod)
    ops = [r.randrange(1, mod) for _ in range(depth)]
    cur = start; steps = []; states = []
    for k, o in enumerate(ops, 1):
        nxt = (cur + o) % mod
        steps.append(f"{k}. {cur} + {o} = {nxt} (mod {mod})")
        states.append(str(nxt)); cur = nxt
    prompt = f"Start at {start}. Add {', '.join(map(str, ops))} in turn, each time taking the result mod {mod}. What is the final value?"
    return Instance(prompt, steps, states, str(cur), depth, {"start": start, "ops": ops, "mod": mod})

def solve(inst):
    m = inst.meta["mod"]; c = inst.meta["start"]
    for o in inst.meta["ops"]: c = (c + o) % m
    return str(c)

def _extract(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for l in reversed(lines):
        m = re.match(r"(?i)answer:\s*(.*)$", l)
        if m: return m.group(1).strip().rstrip(".")
    return lines[-1] if lines else ""

def check(inst, completion):
    return _extract(completion) == inst.answer

def format_cot(inst):
    return "\n".join(inst.steps) + f"\nSo the final value is {inst.answer}.\nAnswer: {inst.answer}"

def exemplars(k, seed):
    return [generate(3, 10_000 + seed * 100 + i) for i in range(k)]

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        for i in range(200):
            d = random.Random(i).choice(DEPTHS); inst = generate(d, i)
            assert solve(inst) == inst.answer and check(inst, format_cot(inst)) and not check(inst, "Answer: x")
            assert generate(d, i) == inst and len(inst.steps) == len(inst.states) == d
        print("selftest ok")
