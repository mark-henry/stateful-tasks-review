"""Load tasks/<slug>/task.py by path. Tasks never import the harness."""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
from types import ModuleType

ROOT = Path(os.environ.get("STATEFUL_TASKS_ROOT", Path(__file__).resolve().parents[2]))
TASKS_DIR = ROOT / "tasks"

SURVIVORS = [
    "addition", "blocksworld", "boolean_expressions", "cellular_automaton", "cruxeval",
    "cup_shuffling", "dyck", "entity_tracking_boxes", "hanoi", "multiplication",
    "nested_arithmetic", "random_lookup_table", "s5_composition", "synthetic_program_trace",
    "threesum", "turing_machine",
]

# Dropped after batch 1 (kept implemented; excluded from sweeps by default).
DROPPED = {
    "multiplication": "Qwen3.5-9B: acc(no-CoT)=1.0 through 3x3 digits, scratchpad only hurts; no CoT gap to study",
}

# Markers. `†` = the source taught the trace format by fine-tuning / from-scratch training (the format was
# never prompted). `*` = no published CoT trace exists at all (the format is ours).
DAGGER = {
    "addition": "Nye 2021: fine-tuned on the scratchpad format",
    "synthetic_program_trace": "Nye 2021: fine-tuned on the trace format",
    "s5_composition": "belindal 2025: fine-tuned; composition convention never prompted",
    "random_lookup_table": "Ramesh 2023: trained from scratch on the token format",
    "threesum": "Pfau 2024: trained from scratch on the token format",
}
ASTERISK = {
    "cellular_automaton": "no published CoT trace; format is ours",
    "entity_tracking_boxes": "no published CoT trace; format is ours",
}

# Per-slug knob overrides for the sweep, with the reason. Empty = published defaults.
SWEEP_KNOBS = {
    # 28-42% of published-default instances have the answer "nothing"; an always-"nothing" baseline
    # would eat the CoT gap. The task ships a policy that queries a non-empty box instead.
    "entity_tracking_boxes": {"query_policy": "most_changed_nonempty"},
}

# Slugs whose published exemplar (exemplars()[0]) has a different prompt shape from generated
# instances; the sweep drops it and uses exemplars(k+1)[1:] so the few-shot prefix is homogeneous.
SKIP_PUBLISHED_EXEMPLAR = {"random_lookup_table"}   # published instance lists 15 per-step tables

REQUIRED = ["Instance", "generate", "solve", "check", "format_cot", "exemplars",
            "ANSWER_FORMAT", "DEPTHS", "KNOBS"]

_cache: dict[str, ModuleType] = {}

def load_task(slug: str, path: Path | None = None) -> ModuleType:
    if slug in _cache:
        return _cache[slug]
    task_dir = path or (TASKS_DIR / slug)
    if not task_dir.exists() and (Path(__file__).resolve().parents[1] / "tests" / slug).exists():
        task_dir = Path(__file__).resolve().parents[1] / "tests" / slug   # harness test fixtures
    file = task_dir / "task.py"
    if not file.exists():
        raise FileNotFoundError(f"{file} (task not implemented yet?)")
    # run with cwd-independent vendor access: tasks read vendor/ relative to __file__
    spec = importlib.util.spec_from_file_location(f"stateful_task_{slug}", file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    old = os.getcwd()
    try:
        os.chdir(task_dir)          # some tasks may open vendor/ relative to cwd
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old)
    missing = [k for k in REQUIRED if not hasattr(mod, k)]
    if missing:
        raise AttributeError(f"{slug}: task.py missing {missing}")
    _cache[slug] = mod
    return mod

def implemented(slugs=SURVIVORS, include_dropped=False) -> list[str]:
    return [s for s in slugs if (TASKS_DIR / s / "task.py").exists() and (include_dropped or s not in DROPPED)]

def display_name(slug: str) -> str:
    return slug + ("†" if slug in DAGGER else "") + ("*" if slug in ASTERISK else "")
