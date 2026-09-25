"""Batch 1: acc(CoT), acc(no-CoT), CoT gap, swept over depth.

    inspect eval stateful/batch1.py -T slug=s5_composition -T condition=cot --model openrouter/qwen/qwen-2.5-7b-instruct

Sample ids are "<slug>/<condition>/d<depth>/s<seed>" so logs from the two conditions pair up exactly:
the SAME instances (same depth, seed) are used in both conditions.
"""
from __future__ import annotations
import json
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import GenerateConfig, ContentReasoning
from inspect_ai.scorer import scorer, Score, Target, accuracy, stderr, grouped, CORRECT, INCORRECT
from inspect_ai.solver import generate, TaskState

from .registry import load_task
from .prompts import build_messages

def instance_seed(seed: int, depth: int, i: int) -> int:
    return seed * 1_000_003 + depth * 1_009 + i

def build_dataset(slug: str, condition: str, depths, n: int, seed: int, shots: int, knobs: dict):
    t = load_task(slug)
    depths = list(depths) if depths else list(t.DEPTHS)
    samples = []
    for d in depths:
        for i in range(n):
            s = instance_seed(seed, d, i)
            inst = t.generate(d, s, **knobs)
            samples.append(Sample(
                id=f"{slug}/{condition}/d{d}/s{s}",
                input=build_messages(t, inst, condition, shots, seed, knobs),
                target=inst.answer,
                metadata={"slug": slug, "condition": condition, "depth": d, "inst_seed": s,
                          "knobs": knobs, "n_steps": len(inst.steps),
                          "gold_cot_chars": len(t.format_cot(inst))},
            ))
    return MemoryDataset(samples, name=f"{slug}-{condition}")

@scorer(metrics=[accuracy(), stderr(), grouped(accuracy(), "depth")])
def task_check():
    """Delegates to the task's own check(); records leak/format flags in metadata."""
    async def score(state: TaskState, target: Target) -> Score:
        md = state.metadata
        t = load_task(md["slug"])
        inst = t.generate(md["depth"], md["inst_seed"], **md.get("knobs", {}))
        completion = state.output.completion or ""
        ok = bool(t.check(inst, completion))
        lines = [ln for ln in completion.strip().splitlines() if ln.strip()]
        has_answer_line = any(ln.strip().lower().startswith("answer:") for ln in lines)
        msg = state.output.message if state.output else None
        hidden = bool(msg and isinstance(msg.content, list) and any(
            isinstance(c, ContentReasoning) and (c.reasoning or "").strip() for c in msg.content))
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=completion[-200:],
            metadata={
                "has_answer_line": has_answer_line,
                "hidden_reasoning": hidden,   # model reasoned in a hidden channel: invalidates the arm
                "n_lines": len(lines),
                "leaked_reasoning": (md["condition"] == "nocot" and len(lines) > 1),
                "truncated": (state.output.stop_reason == "max_tokens") if state.output else None,
                "output_tokens": (state.output.usage.output_tokens if state.output and state.output.usage else None),
                "input_tokens": (state.output.usage.input_tokens if state.output and state.output.usage else None),
            },
        )
    return score

@task
def batch1(slug: str, condition: str = "cot", depths: str | list[int] | None = None, n: int = 50,
           seed: int = 0, shots: int = 3, knobs: str | dict = "{}",
           max_tokens_cot: int = 0, max_tokens_nocot: int = 0, cache: bool = True) -> Task:
    """max_tokens_*: 0 = size from the data. cot: 2x the longest gold trace (chars/2, generous for
    spaced-digit formats) with a 1024 floor; nocot: longest target chars/2 + 16, floor 24."""
    if isinstance(depths, str):
        depths = [int(x) for x in depths.split(",") if x]
    if isinstance(knobs, str):
        knobs = json.loads(knobs) if knobs else {}
    dataset = build_dataset(slug, condition, depths, n, seed, shots, knobs)
    if condition == "cot":
        mt = max_tokens_cot or max(1024, 2 * max(s.metadata["gold_cot_chars"] for s in dataset) // 2)
    else:
        mt = max_tokens_nocot or max(24, max(len(s.target) for s in dataset) // 2 + 16)
    cfg = GenerateConfig(temperature=0.0, max_tokens=mt)
    return Task(
        dataset=dataset,
        solver=[generate(cache=cache)],
        scorer=task_check(),
        config=cfg,
        name=f"batch1-{slug}-{condition}",
        metadata={"slug": slug, "condition": condition, "shots": shots, "seed": seed, "max_tokens": mt},
    )
