"""Contract check for one or more tasks: registry load, the task's own --selftest, and a mock Inspect eval of
both conditions (the gold trace must score 1.0 through the harness; the no-CoT path must run). No API key needed.

    uv run python check_task.py s5_composition dyck
    uv run python check_task.py --all
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from inspect_ai import eval
from inspect_ai.model import get_model, ModelOutput
from stateful.registry import load_task, implemented, TASKS_DIR
from stateful.batch1 import batch1, build_dataset

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("slugs", nargs="*", help="task slugs (directories under tasks/)")
ap.add_argument("--all", action="store_true", help="every implemented task, including dropped ones")
a = ap.parse_args()
slugs = implemented(include_dropped=True) if a.all else a.slugs
if not slugs:
    ap.error("give one or more slugs, or --all")

failed = []
with tempfile.TemporaryDirectory() as log_dir:
    for slug in slugs:
        t = load_task(slug)
        r = subprocess.run([sys.executable, "task.py", "--selftest"], cwd=TASKS_DIR / slug, capture_output=True, text=True, timeout=600)
        out = (r.stdout or r.stderr).strip()
        print(f"[{slug}] selftest rc={r.returncode} :: {out.splitlines()[-1][:150] if out else ''}")
        d = t.DEPTHS
        ds = build_dataset(slug, "cot", [d[0], d[-1]], 2, 0, 3, {})
        outs = [ModelOutput.from_content("mockllm/model", t.format_cot(t.generate(s.metadata["depth"], s.metadata["inst_seed"]))) for s in ds]
        lg = eval([batch1(slug=slug, condition="cot", depths=f"{d[0]},{d[-1]}", n=2, shots=3, cache=False)], model=get_model("mockllm/model", custom_outputs=outs), log_dir=log_dir, display="none", log_samples=True)[0]
        acc = lg.results.scores[0].metrics["accuracy"].value if lg.status == "success" else None
        lg2 = eval([batch1(slug=slug, condition="nocot", depths=f"{d[0]},{d[-1]}", n=2, shots=3, cache=False)], model="mockllm/model", log_dir=log_dir, display="none")[0]
        ptoks = sum(len(m.content) for m in ds[-1].input) // 4
        print(f"[{slug}] cot-gold acc={acc} ({lg.status}) nocot {lg2.status} | DEPTHS={d} | ANSWER_FORMAT={t.ANSWER_FORMAT!r} | ~prompt chars/4 at max depth={ptoks} | gold cot chars at max depth={ds[-1].metadata['gold_cot_chars']}")
        if r.returncode != 0 or acc != 1.0 or lg2.status != "success":
            failed.append(slug)
print(f"[check_task] {len(slugs) - len(failed)}/{len(slugs)} ok" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
sys.exit(1 if failed else 0)
