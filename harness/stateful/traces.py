"""Load each instance's own batch-1 CoT trace from the Inspect logs, keyed by (slug, depth, inst_seed)."""
from __future__ import annotations
from functools import lru_cache
from inspect_ai.log import list_eval_logs, read_eval_log

MODEL_LOGS = {   # model tag -> (inspect model string, log dirs in override order)
    "qwen9b": ("together/Qwen/Qwen3.5-9B", ["results/logs/batch1", "results/logs/batch1b-9b"]),
    "ds":     ("together/deepseek-ai/DeepSeek-V4-Flash-0731", ["results/logs/batch1-ds", "results/logs/batch1-ds-cap"]),
    "dspro":  ("together/deepseek-ai/DeepSeek-V4-Pro-0813", ["results/logs/batch1-dspro"]),
    "llama70b": ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", ["results/logs/batch1-70b"]),
}

@lru_cache(maxsize=None)
def cot_traces(model_tag: str, slug: str) -> dict:
    """(depth, inst_seed) -> dict(completion, output_tokens, correct)"""
    out = {}
    for d in MODEL_LOGS[model_tag][1]:
        for info in sorted(list_eval_logs(d), key=lambda i: i.name):
            log = read_eval_log(info, header_only=True)
            md = log.eval.metadata or {}
            if md.get("slug") != slug or md.get("condition") != "cot" or log.status != "success": continue
            log = read_eval_log(info)
            for s in log.samples:
                sc = s.scores.get("task_check") if s.scores else None
                if sc is None: continue
                out[(s.metadata["depth"], s.metadata["inst_seed"])] = {
                    "completion": s.output.completion or "", "correct": sc.value == "C",
                    "output_tokens": (sc.metadata or {}).get("output_tokens")}
    return out
