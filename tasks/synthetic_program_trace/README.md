# synthetic_program_trace

Tiny integer Python programs (2-3 variables, `=` / `+=` / `-=` / small `*=`, bounded `while`
loops) traced line-by-line in the Nye et al. (2021) scratchpad format, answering with the final
value of `output`.

## Source and citation

Primary source (and the published format reproduced here):

> Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David
> Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, Augustus
> Odena. **"Show Your Work: Scratchpads for Intermediate Computation with Language Models."**
> arXiv:2112.00114 (2021). Section 5.1 and Appendix C.

The trace format is taken from Appendix C, "Example few-shot prompt for synthetic Python
experiments" (pp. 13-14), transcribed verbatim into `published_trace.txt`. Nye et al. say of the
underlying corpus (Section 5.1): *"We use a dataset of synthetic Python programs modified from
Bieber et al. (2020). These programs include small integers (0, 1, and 2), simple while loops,
and if statements."* No code or data was ever released for the paper (see `SOURCING.md` for the
search record), so the generator here is written from scratch to that description.

Secondary references for the operation vocabulary and prompt framing:

- Wojciech Zaremba, Ilya Sutskever. "Learning to Execute." arXiv:1410.4615 (2014) — the original
  tiny-program-execution task; its generator's primitives (assignment, add/subtract, small
  multiply, bounded loop, if) are the vocabulary reused here.
- David Bieber, Charles Sutton, Hugo Larochelle, Daniel Tarlow. "Learning to Execute Programs
  with Instruction Pointer Attention Graph Neural Networks." NeurIPS 2020 — the synthetic-program
  corpus Nye et al. modified.
- BIG-bench `auto_debugging` (Apache 2.0) — precedent for the "what is the value of X" phrasing.

## Vendored vs written

`vendor/` holds sourcing-phase artifacts only. **Nothing in `vendor/` is read at runtime**;
`task.py` is pure python + stdlib and has no data dependency.

| path | what | commit | license |
| --- | --- | --- | --- |
| `vendor/learning_to_execute/` | Zaremba & Sutskever 2014 reference code (Torch7/Lua) | `dc6baf6835337f0acea76c8e8233f767c50b058f` | Apache 2.0 |
| `vendor/BIG-bench-tasks/` | sparse checkout of `auto_debugging`, `simple_arithmetic`, `modified_arithmetic`, `cs_algorithms` | `092b196c1f8f14a54bbc62f24759d43bde46dd3b` | Apache 2.0 |
| `published_trace.txt` | transcription of Nye et al. 2021 Appendix C (arXiv) | — | arXiv non-exclusive license; quoted for research |

Everything in `task.py` — the program generator, the structured-program tracer, the independent
AST interpreter used as the reference solver, the answer parser — is written from scratch. The
Lua code was a vocabulary reference only; no line of it was ported.

## Format decision

The gold trace reproduces the published Appendix C format character-for-character:

```
[BEGIN]

state: {}
line: def f(v0):
state: {"f": "<callable_object f>"}
line: output = f(6)
state: {"v0": 6}
line:   v0 += 0
state: {"v0": 6}
...
line:   return v0
state: {"f": "<callable_object f>", "output": 24}

[DONE]
```

Every executed source line is echoed verbatim with its original indentation after `line: `, then
the **full** current variable dict after `state: ` as JSON (not just the variable that changed).
Loops are fully unrolled, so the `while` header reappears once per condition check (T+1 times for
T trips) and each body line once per iteration. Inside the call the state shows the callee frame
(`{"v0": ..., "v4": ...}`); the `def` line and the `return` line show the module frame, where `f`
renders as `"<callable_object f>"` and `output` gets bound at the end.

`--selftest` asserts that `exemplars(3, seed)[0]` — the reconstructed Appendix C example
(`f(6)`, `v0 *= 2` twice, answer 24) — is byte-identical to the trace in `published_trace.txt`,
so format drift fails the test.

Three deviations, all required by AMENDMENT 3 or documented here:

1. **The prompt's last line.** The published prompt ends `What is the execution trace?`; this
   task's prompt ends `What is the value of output?`. AMENDMENT 3 requires `Instance.prompt` to
   be the problem statement only with no scratchpad instruction (the harness supplies that), and
   the no-CoT arm has to be answerable at all. Everything above that line — `Consider the
   following Python function:`, the function body, the `output = f(K)` call line — is the
   published wording unchanged.
2. **`Answer: <x>` after `[DONE]`.** The single harness-imposed normalization mandated by
   AMENDMENT 3. The published closing `[DONE]` is emitted first, then the answer line.
3. **A step is two lines of text.** `steps[i]` is the pair `line: ...\nstate: ...`, because one
   state update in this format occupies a `line:`/`state:` couplet. AMENDMENT 3 explicitly allows
   multi-line steps. `states[i]` is just the JSON state dict.

One generator extension beyond the single published example: the right-hand side of `+=` / `-=`
is sometimes another variable (`v0 += v4`), not only a constant. The Appendix C example only
shows constant operands, but the corpus it comes from (Bieber et al. 2020) uses variable
operands, and they are what force the model to keep *more than one* variable live. Set
`p_var_operand=0.0` to reproduce constants-only programs.

## Depth semantics

**depth = the number of executed statements, loops unrolled = `len(steps)` = `len(states)`,
exactly.** This is the paper's own step granularity (Figure 1 caption: "all loops are unrolled
across time") and is what `desk.json`'s `knob` field records.

Three of those statements are fixed overhead present in every instance: `def f(v0):`,
`output = f(K)`, and `return v0`. So `depth - 3` statement executions are available for the
function body, and `depth >= 4` is required (`generate` raises below that). The published
Appendix C example is exactly depth 12.

Cost accounting the generator uses when it places a block:

| block | executed statements |
| --- | --- |
| one simple statement | 1 |
| `while` loop, trip count T, body of B statements | `1` (counter init) + `T+1` (condition checks) + `T*B` (body) |
| `if` block, body of B statements (`allow_if`) | `1` + `B` if the condition holds, else `1` |

The generator fills the budget exactly: loops are only placed when a feasible `(T, B)` fits in
the remaining budget, and the remainder is always fillable with cost-1 statements. Minimum loop
cost is 5 (T=1, B=2), which is why depth 6 instances are straight-line only.

## ANSWER_FORMAT

`ANSWER_FORMAT = "a single integer, the final value of output (e.g. 24)"`.

The published trace's terminal state line is `state: {"f": "<callable_object f>", "output": 24}`,
i.e. the paper's synthetic-program task ends by binding `output`, and its direct-prediction
baseline (Table 2: 11% few-shot / 20% fine-tuned, vs 26.5% / 41.5% with the scratchpad) predicts
that value without the trace. So the answer is the final value of `output` — a plain integer:
exact-match comparable, no judge, and a coherent question in the no-CoT condition.

The rejected alternative was answering with the whole final state dict. It would have made the
answer a paraphrase of the trace format itself (awkward to match, and it leaks the scratchpad
into the no-CoT arm), and it is not what the paper's direct-prediction baseline predicts.

`check()` takes the last line matching `Answer:` (tolerating `**Answer:** 24.`, bullets, and
`=` instead of `:`), else the last non-empty line, strips markup and thousands separators, and
compares the integer to the target.

## DEPTHS

`DEPTHS = [6, 12, 20, 32, 48, 72]`

- **6** — floor anchor: three straight-line body statements, no loop fits. Both conditions should
  ace it; if no-CoT misses here, something is wrong with the prompt, not the model.
- **12** — exactly the length of the published Appendix C example (one loop, one or two variables
  live). The comparison point against the literature.
- **20, 32** — one to three unrolled loops, 6-11 writes to `v0`. This is where the direct-answer
  arm should start to fall apart; Nye et al.'s 137B model managed only 11% few-shot without a
  scratchpad on this family.
- **48, 72** — 16 and 25 writes to `v0` across ~4-6 loops. Expected floor for no-CoT on a 7B, and
  where CoT itself should begin to break. Cost stays modest: ~50 characters per step, so the gold
  trace is ~3.8k characters (roughly 1.3k tokens) at depth 72.

Answer guessability falls off across the sweep — over 300 seeds the most common answer takes
11.3% of instances at depth 6 but only 2.3% at depth 72 — so the no-CoT arm cannot ride a
majority class.

## Knobs

`depth` is not a knob. `KNOBS` (name — default — effect):

- `n_vars` — 3 — distinct variables including `v0` and loop counters (>= 2). Names are drawn from
  `v0..v9`, with `v0` always the parameter and the returned variable.
- `const_max` — 2 — constants for `=` / `+=` / `-=` come from `0..const_max` ("small integers
  (0, 1, and 2)").
- `mul_max` — 2 — multipliers for `*=` come from `2..mul_max`; the published example only uses
  `*= 2`.
- `max_trips` — 4 — maximum while-loop trip count.
- `max_body` — 3 — maximum statements in a loop body, counting the counter decrement (minimum 2).
- `p_loop` — 0.5 — chance of opening a loop at each block position where one fits.
- `max_loops` — 6 — cap on loops per program.
- `p_target_v0` — 0.5 — chance a statement targets `v0`.
- `p_var_operand` — 0.35 — chance a `+=` / `-=` right-hand side is a variable rather than a
  constant.
- `value_cap` — 1000 — generation-time magnitude gate (see caveats).
- `allow_if` — False — also emit `if <var> <cmp> <const>:` blocks.
- `p_if` — 0.3 — chance of opening an `if` block, when `allow_if`.
- `arg_max` — 9 — the call argument `K` in `output = f(K)` comes from `0..arg_max`.

## Redaction

`redact_prompt(inst, k)` (SPEC AMENDMENT 4) blanks the call argument `K` in `output = f(K)` with
a single `[…]` placeholder for every `k >= 1`; `redact_prompt(inst, 0)` returns `inst.prompt`
byte-identical. `REDACTION_MEANINGFUL = True`.

**The program text is kept on purpose.** AMENDMENT 4: *"where the 'initial state' is a static
object needed for every step (a rule table, a lookup table, the program text, ...), keep it — it
is not state."* The source is the rule table of this task, not its state: it is the same text at
every step, its statements are not consumed one per step (loops are unrolled, so one source line
drives many steps and the number of steps a `while` contributes is itself a function of the
runtime state), and steps `k+1..depth` are simply unexecutable without it. The state a model
could recompute is the *trace so far*, which is the program applied to `K` — so `K` is the whole
of the recomputable input, and removing it is exactly what prevents re-executing from scratch
while leaving a model able to continue from the last `state:` line of the trace prefix it is
given. This is the same call as AMENDMENT 4 makes for cruxeval (redact the input argument, keep
the code); here the analogue is exact rather than approximate, because the argument is the only
per-instance input the prompt carries.

The redaction is therefore binary in `k`, not graded: `k = 1`, `k = depth//2` and `k = depth` all
produce the same string, and at `k = depth` nothing of the initial state survives while the
program text and the question do. Two consequences for the harness:

- Step 2 of the published trace is `line: output = f(K)`, so for every `k >= 2` the trace prefix
  restates the argument. That is expected and allowed ("the redaction is of the PROMPT only") —
  what it removes is the *prompt-side* copy a model could read instead of carrying state.
- `k = 1` is degenerate: the prefix stops after `line: def f(v0):`, before the call line, so the
  argument is genuinely unavailable and no model can continue. Use `k >= 2` for this task.

Rendered example, `generate(12, 0)` at `k = depth//2 = 6` (this seed reproduces the published
Appendix C program; answer 24):

```
Consider the following Python function:

def f(v0):
  v4 = 2
  while v4 > 0:
    v4 -= 1
    v0 *= 2
  v0 -= 0
  return v0

output = f([…])

What is the value of output?
```

The state it must be continued from lives in the trace prefix, whose step 6 is
`line:     v0 *= 2` / `state: {"v0": 12, "v4": 1}`.

## Corruption

`corrupt_step(inst, k, seed)` (SPEC AMENDMENT 6) returns `(step_text, corrupted_state)` for step
`k` (1-based). A step here is two lines — `line: <source>` and `state: {...}` — so only the
`state:` line is rewritten; the `line:` half comes back byte-identical, which keeps the action the
step performed intact and moves only the result it reports. The edit is minimal: exactly one
variable's value shifts by ±1, so the wrong state keeps the same keys in the same published order
with the same shapes, is a state the program could plausibly have been in, and is never equal to
`inst.states[k-1]`. `corrupted_state` is the same canonical `json.dumps` string used for `states`.

The variable picked is **the one the step just wrote** — the assignment's target, the callee's
`v0` for `output = f(K)`, `output` for `return v0` — whenever it is a live integer in the reported
state, since that is the corruption that actually propagates into later steps and the answer. A
`while`/`if` condition line writes nothing, so there one live integer variable is chosen
deterministically from the state instead. The task has a single trace format (the published Nye et
al. one; no `format` knob), so no per-format branching is needed.

Rendered example, `generate(12, 0)` (the published Appendix C program, answer 24) at
`k = depth//2 = 6`, `seed = 0` — the true step is

```
line:     v0 *= 2
state: {"v0": 12, "v4": 1}
```

and `corrupt_step(inst, 6, 0)` returns

```
line:     v0 *= 2
state: {"v0": 11, "v4": 1}
```

with `corrupted_state == '{"v0": 11, "v4": 1}'`. (`seed = 1` shifts the other way, to `13`; the
sign and, on condition lines, the variable are the only things the seed controls.) At `k = depth`
the same machinery corrupts `output` itself: `state: {"f": "<callable_object f>", "output": 23}`.

One degenerate point: step 1 is `line: def f(v0):` with `state: {"f": "<callable_object f>"}`,
which holds no integer to shift. It is corrupted instead into the premature-binding error a model
plausibly makes — `{"f": "<callable_object f>", "v0": 6}`, the parameter bound one step early —
which has the same shape as the final step's state (a callable plus an integer) and so is still a
legal state of this format.

## Caveats

- **State growth is gated, deliberately.** `desk.json` records `state_bounded: false` and flags
  that the published example doubles `v0` every iteration, so magnitudes grow with depth. The
  generator gates this: any statement that would push some variable past `value_cap` (default
  1000) is re-drawn, dropping `*=` first. Observed answers stay within roughly ±1000 at depth 72.
  This keeps each state a short, cheap-to-tokenize string at every depth, which is the point of
  the sweep — but it does mean `*=` gets throttled as depth grows. Raise `value_cap` (e.g. to
  `10**12`) to recover the ungated behaviour; `state_bounded` in `desk.json` describes the
  published task, not this gate, and was left as-is.
- **No plain re-assignment of a live variable**, apart from loop-counter initialisation. A
  statement like `v0 = 0` mid-program would discard everything the trace had accumulated and let
  a model skip straight to the last write — the opposite of what this benchmark is measuring.
  Plain assignment is therefore only used to *introduce* a variable or to initialise a loop
  counter, which is exactly what the published example does (`v4 = 2`).
- **Residual dead code.** Statements targeting a non-`v0` variable that is never read again do
  not affect the answer, and a sufficiently clever model could prune them. `p_target_v0` plus a
  rejection gate (at least 30% of body statements must write `v0`, up to 8 redraws) keeps most of
  the trace load-bearing; `meta["v0_writes"]` reports the actual count per instance (mean 3.7 at
  depth 12, 24.5 at depth 72). It is not zero, so treat depth as an upper bound on serial
  dependency length, not an exact one.
- **Flat loops only.** The paper says "simple while loops" and the example has one; nested loops
  are not generated. Nesting would be the natural next extension, and would change the cost
  arithmetic above.
- **`if` is off by default**, even though Nye et al. mention if statements, because the assigned
  operation vocabulary for this task is assign / `+` / `-` / small `*` / while. It is implemented
  and selftested; turn it on with `allow_if=True`.
- **The reference solver is independent of the generator.** `solve()` re-parses the program text
  back out of `Instance.prompt` with `ast` and runs it through a small tree-walking interpreter
  with a fuel budget; it shares no code with the generator's own simulation and never sees the
  structured program. It does not use `exec`.
- **`desk.json` was not modified**; no errors were found in it. Two notes for whoever collates it:
  its `tokens_per_step` (~73) was measured over the whole `published_trace.txt` file including the
  ~700-character attribution header, as its own `notes` field admits; the trace body itself runs
  about 50 characters (roughly 20-25 tokens) per step. And `contamination_risk: low` holds for
  generated instances, but the Appendix C example returned by `exemplars(k, seed)[0]` is a
  published few-shot prompt from a well-known paper — it is a prompt exemplar, never a scored
  instance.

## Running

```
python3 task.py --selftest        # 200 random (depth, seed) pairs + published-trace fidelity + redaction
python3 task.py --demo            # 3 instances at depth 6 and 3 at depth 72 -> examples.txt
```
