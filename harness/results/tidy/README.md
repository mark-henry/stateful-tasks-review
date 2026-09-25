# results/tidy — chart-ready CSVs

Regenerate with `cd harness && uv run python tidy.py` (about 1 minute; idempotent, it rewrites every CSV here). All
numbers come from the Inspect logs in `results/logs/`. The markdown tables (`SUMMARY.md`, `SECOND_CHANCE.md`,
`BENCH*.md`, per-dir `batch1.csv`) are only cross-checked, and every disagreement goes to `crosscheck.csv`.

| file | grain | rows |
|---|---|---|
| depth_sweep.csv | (slug, format, model, depth) | 431 |
| survival.csv | (slug, format, model) | 63 |
| batch2.csv | slug (batch-2 cell) | 12 |
| batch3.csv | (slug, condition, k) | 105 |
| batch3_summary.csv | slug | 10 |
| tasks.csv | slug (`registry.implemented(include_dropped=True)`) | 16 |
| master.csv | (slug, format), one row per format that passes on at least one model; a slug with no passing format gets one row in its batch-1 format | 19 |
| crosscheck.csv | one disagreement between the logs and a markdown/csv/desk source | 23 |

## Vocabularies

`model`: `qwen9b` = Qwen3.5-9B (thinking off), `llama70b` = Llama-3.3-70B-Instruct-Turbo, `ds` = DeepSeek-V4-Flash-0731
(thinking off), `dspro` = DeepSeek-V4-Pro-0813 (only s5_composition and threesum, published format).

`format`, and where each label's logs live:

| format | meaning | slugs | log dirs (qwen9b / llama70b / ds / dspro) |
|---|---|---|---|
| `published` | the batch-1 trace format taken from the source (for † tasks, the SFT or from-scratch token format). For hanoi this is the Apple paper's **plan** formulation (`moves = [...]`, depths 2,5,10,20,40,80) | all except the two below | `batch1`+`batch1b-9b` / `batch1-70b` / `batch1-ds`+`batch1-ds-cap` / `batch1-dspro` |
| `ours` | batch-1 format for tasks with no published trace: cellular_automaton one-row-per-generation (`rows`), entity_tracking_boxes | cellular_automaton, entity_tracking_boxes | same dirs as `published` |
| `ergonomic` | SPEC AMENDMENT 5 `format="ergonomic"`, the SFT-free variant | s5_composition, random_lookup_table, threesum, addition | `batch1-erg-9b` / — / `batch1-erg-ds` |
| `execute` | hanoi `formulation="execute"`: apply the given moves and report the configuration (`move k: [d,f,t] -> [[..],[..],[..]]`) | hanoi | `batch1-2nd-9b` / — / `batch1-2nd-ds` |
| `cells2` | cellular_automaton per-cell format, rev 2 (`cell i: 100 -> 0`, then `row:`) | cellular_automaton | `batch1-2nd-9b` / — / `batch1-2nd-ds` |
| `cells3` | cellular_automaton per-cell format, rev 3 (named neighbours plus a running `row so far`) | cellular_automaton | `batch1-2nd3-9b` / — / `batch1-2nd3-ds` |

How the mapping was verified (the script asserts all of these on every run):
- The model string in every log header matches the tag.
- `task_args.knobs` in every log is `{}`, except `{"query_policy": "most_changed_nonempty"}` for entity_tracking_boxes (registry `SWEEP_KNOBS`) and `{"format": "ergonomic"}` in the `-erg-` dirs.
- **The execute, cells2 and cells3 runs pass no knob.** They ran on changed task *defaults*, so their sample ids collide with batch-1 ids (for example `hanoi/cot/d4/s…` exists under both the plan and execute formulations). The script therefore never dedupes across format groups. It also checks the few-shot assistant turns of the first cot sample in every hanoi and cellular_automaton log for a format marker (`moves = [` = plan, `move 1:` = execute, `generation 1:` alone = rows, `cell 1:` = cells2, `row so far` = cells3) and asserts that the marker matches the label.
- batch 3: `plans/batch3.json` knobs give `ergonomic` for s5/rlt/threesum. hanoi and cellular ran on defaults at 2026-09-19 01:33 UTC, after rev 3, and the gold-trace prefill markers confirm `execute` and `cells3`.
- batch 2 reused batch-1 traces from `stateful/traces.py` `MODEL_LOGS` (batch-1 dirs only), so every batch-2 row is `published` (entity: `ours`). hanoi and cellular were not in batch 2.

## depth_sweep.csv

- **Dedupe:** within a (model, format) group, dirs are read in the order listed above. The collect.py rule applies: logs are sorted by name, only scored samples count, and a later dir overrides an earlier one on the same sample id. `sources` lists the dirs whose samples ended up in the cell. `superseded_sources` lists dirs whose samples for that cell were overridden. The overrides are:
  - 9B `batch1b-9b` (n=150) supersedes `batch1` (n=50) at blocksworld d4, boolean d4 and entity d26. These are the only 12 cells with n=150.
  - DeepSeek `batch1-ds-cap` (max_tokens 8192) supersedes `batch1-ds` (cap 1024) wholesale for hanoi, cellular_automaton, s5_composition and threesum.
- `gap = acc_cot − acc_nocot`: unpaired arm means, as in collect.py. `p_mcnemar` is the two-sided exact McNemar test on pairs matched by `(depth, inst_seed)`. The function is `mcnemar_exact`, taken from collect.py's own source via `ast`, so it is literally the same code. `b_cot_only`/`c_nocot_only` are the discordant counts.
- **Missing pairs:** no cell is missing an arm, so no `p_mcnemar` is blank. Three cells have unequal arms because samples failed with HTTP 402 "Credit limit exceeded" and were never scored (`n_unscored`). Their p uses only the paired instances:
  - cellular_automaton cells2 qwen9b d1: n_cot = 27, n_nocot = 50.
  - threesum published dspro d12: n_cot = 49.
  - threesum published dspro d22: n_cot = 49.
- `tok_per_step = tok_out_cot_mean / depth`. `depth` is the harness knob (`tasks.csv:harness_depth`), which is not always the DESK.md knob. This is the model's actual spend per requested step, not the format's cost. Where the model does not write the gold format, the two diverge. Two diagnostics show when:
  - `cot_lines_per_step`: mean non-empty completion lines divided by depth.
  - `cot_len_ratio_vs_gold`: mean completion characters divided by mean gold-trace characters.
- `hidden_reasoning_rate` covers both arms. It is 0 everywhere.
- Extra columns beyond the spec: `n_cot`, `n_nocot`, `b_cot_only`, `c_nocot_only`, `leak_rate_nocot`, `no_answer_rate_cot`, `n_unscored`, and the two length diagnostics.

## survival.csv

- **Rules:** gap ≥ 0.2, p < 0.05 and acc_cot ≥ 0.7 at the same depth, compared with a 1e-9 tolerance. Without the tolerance, 0.98 − 0.78 comes out as 0.19999999999999996 in floating point and fails the gap rule. summarize.py had exactly this bug (see the crosscheck section).
- `working_depth` = shallowest passing depth. `deepest_passing_depth` = deepest passing depth.
- `best_depth` = the passing depth with the largest gap. **Ties on the 2-dp gap go to the DEEPEST depth.** summarize2.py/SECOND_CHANCE.md and plans/batch2.json use this rule (rlt/ds → d24). summarize.py/SUMMARY.md took the shallowest.
- For failing rows, `best_*` describe the best-gap depth (same tie rule), and `fail_reason` is one of:
  - `no gap`: no depth has gap ≥ 0.2.
  - `acc<0.7`: some depth has a significant gap but low accuracy.
  - `p≥0.05`: a gap ≥ 0.2 that is not significant. This value never occurs, but summarize.py would have labelled it `acc<0.7`.
- `verdict_text` = SUMMARY.md's per-model phrasing. `second_chance_text` = SECOND_CHANCE.md's phrasing. `task_verdict` = SUMMARY.md's verdict column. It is computed over qwen9b, llama70b and ds for published/ours (dspro excluded, as in SUMMARY.md) and over qwen9b and ds for variant formats. It uses the current registry markers.

## batch2.csv

A copy of `results/batch2.csv` with these changes:
- `model` normalised. It was already `qwen9b`/`ds`.
- Added `format` (published, or ours for entity).
- Added `filler_recovery = filler − acc_nocot`.
- Added `cont_k`, the step the continuation was cut at (depth/2, floored).

It was recomputed from `logs/batch2`+`logs/batch2c2` with collect2.py's rule and agrees exactly.

## batch3.csv / batch3_summary.csv

- Long format with `condition ∈ {goldko, mistake, cont}` (`cont` = the uncorrupted continuation, `continue` in the logs), `k_frac = k/depth`, `se = sqrt(p(1−p)/n)`. n = 50 per cell.
- The summary k columns reproduce collect3.py exactly: `k_q1, k_half, k_q3 = max(1, round(d·x))` for x = .25, .5, .75, using Python's round-half-to-even. At small depths they repeat:
  - cellular_automaton d3: k = 1, 2, 2.
  - turing_machine d4: k = 1, 2, 3, and 3 is also d−1.
- `mean_propagation` and `cont_mean` are plain means over those three entries, repeats included, which matches BENCH3.md.
- `prop = cont − mistake` at the same k.

## tasks.csv

- **Desk values:** the `DESK.md` table is the curated, later-corrected table and is the primary source. `tasks/<slug>/desk.json` is used where DESK.md has `·`. Disagreements between the two are in crosscheck.csv:
  - blocksworld bits: 6.0 in DESK.md vs 6.97 in desk.json.
  - turing_machine bits: 10.2 in DESK.md vs null in desk.json.
  - turing_machine bounded: Y in DESK.md vs False in desk.json.
  - turing_machine answer space: 64 in DESK.md vs null in desk.json.
- `answer_space` null means ∞. `guess_floor` = 1/answer_space, or 0 for ∞.
- `source_short` names the task's primary source paper. `trace_source_short` names where the published trace format came from. They differ for s5_composition: the source is Liu et al. 2022, but the trace is from Li et al. 2025, "(How) Do Language Models Track State?", the belindal repo. For the two `*` tasks, `trace_source_short` is "none (format is ours)".
- `state_description` and `harness_depth` are hand-written from the task docstrings.
- `depth_knob` is DESK.md's "Knobs" name for the published knob. This is sometimes not what the harness sweeps: cup_shuffling's desk knob is n people, but the harness sweeps swaps. `harness_depth` gives what the harness actually sweeps.
- `surviving_format` is chosen in this order:
  1. The batch-3 format, if it passes somewhere.
  2. Otherwise, the passing format with the most models passing (ties go to the batch-1 format).
  3. Otherwise, the batch-1 format.
- `max_depth_swept` is taken over all formats and models.

## master.csv

- One row per passing (slug, format). `primary` marks the row that matches `tasks.csv:surviving_format`.
  - cellular_automaton has two rows, cells2 (ds only) and cells3.
  - s5_composition and random_lookup_table each have a `published` row (passes on ds only) and an `ergonomic` row.
  - multiplication passes nowhere and keeps a `published` row.
- `passes_<model>` is blank when that format was never run on that model. `models_passing` counts all four models, including `passes_dspro`, which is an extra column. dspro passes nothing.
- `tok_per_step_9b` and `tok_per_step_ds` are taken at that model's `best_depth`, and are blank if the model does not pass in that format.
- Batch-2 columns (`filler_recovery`, `cont_*`) and batch-3 columns (`goldko_dm1`, `mean_propagation`) are filled only when the batch ran in that row's format. For example, the s5 `ergonomic` row gets batch 3 but not batch 2, and the s5 `published` row gets batch 2 only.
- **`knockout_shape`:**
  - Rows with batch 2 use the classification from the vault post note ("2026/comparative review of stateful tasks post.md", Batch 2 bullet). **entity_tracking_boxes was never classified there and is `unclassified`**: its curve is 0.40/0.10/0.24/0.58 against CoT 0.82.
  - Rows without batch 2 but with batch 3 use `decided_at_end` if gold-knockout at 3d/4 ≤ no-CoT + 0.15. All five such rows qualify, which matches the note's "ten decided-at-the-end tasks".
  - `knockout_shape_source` says which rule applied.
- `self_checking`: yes for hanoi, weak for dyck, no for the other batch-3 rows, and n/a without batch 3.
- `published_format_passes_anywhere` is filled only on variant rows.

## Disagreements with the existing tables (all in crosscheck.csv)

- SUMMARY.md, **boolean_expressions / qwen9b**: the working depth is **d3** under the stated rules, not d4. At d3 the arms are 0.98 and 0.78 (gap 0.20, p = 0.006, n = 50). summarize.py computes 0.98 − 0.78 < 0.2 in floating point and drops it.
- SUMMARY.md, where only the tie-break differs. Same working depth and same best gap, but a different best depth, because SUMMARY picks the shallowest of equal gaps:
  - nested_arithmetic/qwen9b: SUMMARY d5 (1.00, 0.96), tidy d7 (0.96, 0.96). Batch 2 and batch 3 ran at d7.
  - dyck/ds: SUMMARY d16 (0.96), tidy d24 (0.86).
  - random_lookup_table/ds: SUMMARY d2, tidy d24. SECOND_CHANCE.md and batch 2 use d24.
  - Two failing cells: addition/qwen9b is d8 vs d24, threesum/llama70b is d4 vs d22.
- SUMMARY.md, stale markers: SUMMARY.md predates AMENDMENT 5. It marks addition, random_lookup_table, s5_composition, synthetic_program_trace and threesum with `*`, where the registry now uses `†`, and its threesum/cellular verdict wording is the old one.
- SECOND_CHANCE.md, BENCH2.md/batch2.csv, BENCH3.md, and every per-dir batch1.csv and BENCH1.md p-value: no disagreements.
- plans/batch3.json vs HARNESS.md: HARNESS.md says batch 3 ran "at each task's deepest passing 9B depth", but for 6 of 10 tasks it did not:

  | task | depth batch 3 ran at | deepest passing 9B depth |
  |---|---|---|
  | nested_arithmetic | 7 | 20 |
  | turing_machine | 4 | 5 |
  | synthetic_program_trace | 32 | 48 |
  | dyck | 8 | 16 |
  | cup_shuffling | 12 | 16 |
  | random_lookup_table (ergonomic) | 24 | 48 |

  Most of these are the batch-2 depths. Hanoi, cellular, s5 and threesum do match.
- DESK.md vs desk.json: blocksworld bits, and turing_machine bits/bounded/answer space (see tasks.csv above).
