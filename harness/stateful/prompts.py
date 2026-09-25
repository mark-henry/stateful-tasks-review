"""Prompt assembly for the batch-1 conditions. The task supplies the problem statement,
the published-format gold trace, and ANSWER_FORMAT; the harness supplies instructions and few-shot turns."""
from __future__ import annotations
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant
from .registry import SKIP_PUBLISHED_EXEMPLAR

COT_SYSTEM = (
    "Solve the problem. Show your work in exactly the same style as the worked examples, "
    "then finish with one final line of the form `Answer: <answer>`."
)
NOCOT_SYSTEM = (
    "Solve the problem. Do not show any work or reasoning. Reply with a single line of the form "
    "`Answer: <answer>`, where <answer> is {answer_format}, and nothing else."
)

def answer_line(answer: str) -> str:
    return f"Answer: {answer}"

def _exemplars(task, k, seed, knobs):
    """Forward knobs (e.g. format=) to exemplars() when the task accepts them; older tasks take (k, seed) only."""
    try:
        return task.exemplars(k, seed, **knobs) if knobs else task.exemplars(k, seed)
    except TypeError:
        return task.exemplars(k, seed)

def build_messages(task, inst, condition: str, shots: int, seed: int, knobs: dict | None = None):
    knobs = knobs or {}
    """condition in {"cot", "nocot"}. Few-shot exemplars are the same instances in both conditions;
    only the assistant turns differ (full published-format trace vs. bare answer line)."""
    slug = task.__name__.removeprefix("stateful_task_")
    if shots <= 0:
        ex = []
    elif slug in SKIP_PUBLISHED_EXEMPLAR:
        ex = _exemplars(task, shots + 1, seed, knobs)[1:]
    else:
        ex = _exemplars(task, shots, seed, knobs)
    if condition == "cot":
        msgs = [ChatMessageSystem(content=COT_SYSTEM)]
        for e in ex:
            msgs += [ChatMessageUser(content=e.prompt), ChatMessageAssistant(content=task.format_cot(e))]
    elif condition == "nocot":
        msgs = [ChatMessageSystem(content=NOCOT_SYSTEM.format(answer_format=task.ANSWER_FORMAT))]
        for e in ex:
            msgs += [ChatMessageUser(content=e.prompt), ChatMessageAssistant(content=answer_line(e.answer))]
    else:
        raise ValueError(condition)
    msgs.append(ChatMessageUser(content=inst.prompt))
    return msgs
